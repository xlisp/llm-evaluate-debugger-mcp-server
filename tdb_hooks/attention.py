"""Attention-probability capture via SDPA / flash_attn monkey-patches.

PyTorch's ``F.scaled_dot_product_attention`` (and FA3 / flash_attn) deliberately
do not return attention weights. To get them for TDB-style inspection we
temporarily replace those functions with a manual implementation **only when
``capture.want_attn_probs`` is set**.

The current layer index is set by a ``forward_pre_hook`` on each attention
module (see attach.py); the wrapped function reads ``DEBUG['current_layer']``.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .core import DEBUG, LayerState


_PATCHED = {"orig_sdpa": None, "orig_fa3": None, "orig_fa_func": None}


def _record_probs(probs: torch.Tensor):
    """probs: (B, H, T, T)."""
    cap = DEBUG.get("capture")
    layer = DEBUG.get("current_layer")
    if cap is None or layer is None or not cap.want_attn_probs:
        return
    if not cap.wants(layer):
        return
    p = probs.detach().float().cpu()
    cap.attn_probs[layer] = p
    eps = 1e-9
    ent = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(-1).mean(dim=(0, 2))
    st = cap.layer_states.setdefault(layer, LayerState(layer))
    st.attn_entropy_per_head = [float(v) for v in ent.tolist()]


def _manual_attention(q, k, v, *, is_causal: bool, window_left: int | None = None):
    """q, k, v in (B, H, T, D) layout. Returns (out, probs)."""
    # GQA: expand kv heads if needed
    h_q, h_kv = q.size(1), k.size(1)
    if h_q != h_kv and h_q % h_kv == 0:
        rep = h_q // h_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(qf.size(-1))
    scores = qf @ kf.transpose(-1, -2) * scale
    T = scores.size(-1)
    if is_causal:
        mask = torch.ones(T, T, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    if window_left is not None and window_left >= 0:
        i = torch.arange(T, device=scores.device).view(-1, 1)
        j = torch.arange(T, device=scores.device).view(1, -1)
        scores = scores.masked_fill((i - j) > window_left, float("-inf"))
    probs = scores.softmax(dim=-1)
    out = probs @ vf
    return out.to(q.dtype), probs


def _wrapped_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, **kwargs):
    """Drop-in replacement for F.scaled_dot_product_attention.

    Only diverges from the original when capture is asking for probs;
    otherwise calls the original implementation untouched.
    """
    cap = DEBUG.get("capture")
    if cap is None or not cap.want_attn_probs:
        return _PATCHED["orig_sdpa"](query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal,
                                     scale=scale, **kwargs)
    out, probs = _manual_attention(query, key, value, is_causal=is_causal)
    _record_probs(probs)
    return out


def _wrapped_fa_func(q, k, v, *args, **kwargs):
    """For nanochat's flash_attn.flash_attn_func (B, T, H, D layout, causal=True)."""
    cap = DEBUG.get("capture")
    if cap is None or not cap.want_attn_probs:
        return _PATCHED["orig_fa_func"](q, k, v, *args, **kwargs)
    causal = kwargs.get("causal", args[0] if args else True)
    window = kwargs.get("window_size", None)
    window_left = window[0] if (window is not None and window[0] is not None and window[0] >= 0) else None
    # (B, T, H, D) -> (B, H, T, D)
    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out, probs = _manual_attention(q_, k_, v_, is_causal=bool(causal), window_left=window_left)
    _record_probs(probs)
    return out.transpose(1, 2)  # back to (B, T, H, D)


@contextmanager
def patched_attention():
    """Install monkey-patches for the duration of the block.

    Always safe — the wrappers no-op back to the original when capture is off.
    But to avoid any overhead we only install when active.
    """
    _PATCHED["orig_sdpa"] = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _wrapped_sdpa
    # nanochat-style flash_attn module (optional)
    try:
        from nanochat.flash_attention import flash_attn  # type: ignore
        _PATCHED["orig_fa_func"] = flash_attn.flash_attn_func
        flash_attn.flash_attn_func = _wrapped_fa_func
        installed_fa = flash_attn
    except Exception:
        installed_fa = None
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _PATCHED["orig_sdpa"]
        if installed_fa is not None and _PATCHED["orig_fa_func"] is not None:
            installed_fa.flash_attn_func = _PATCHED["orig_fa_func"]
        _PATCHED["orig_sdpa"] = None
        _PATCHED["orig_fa_func"] = None
