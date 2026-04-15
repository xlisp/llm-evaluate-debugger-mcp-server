"""Attach TDB hooks to an arbitrary nn.Module via standard PyTorch forward hooks.

No source-code edits required in the target project. The user only has to tell
us *where* the blocks / attn / mlp / lm_head live (or pick a preset).
"""
from __future__ import annotations

import contextlib
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .core import DEBUG, LayerState, _norm
from .attention import patched_attention, _wrapped_sdpa, _wrapped_fa_func, _PATCHED
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Path resolution helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get(obj: Any, dotted: str) -> Any:
    """Resolve a dotted attribute path: 'transformer.h' -> obj.transformer.h."""
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def _try_get(obj: Any, dotted: str, default=None):
    try:
        return _get(obj, dotted)
    except AttributeError:
        return default


def _autodetect(model: nn.Module) -> dict:
    """Heuristics to find blocks / attn / mlp / lm_head on common GPT layouts.

    Recognises:
      - nanoGPT / nanochat / MathGPT      (transformer.h, attn, mlp, lm_head)
      - CodeChat / minGPT-style flat       (blocks, attn, mlp, head)
      - HuggingFace GPT-2                  (transformer.h, attn, mlp, lm_head)
      - LLaMA-style                        (model.layers, self_attn, mlp, lm_head)
    """
    candidates = [
        dict(blocks="transformer.h", attn="attn", mlp="mlp", lm_head="lm_head"),
        dict(blocks="blocks",         attn="attn", mlp="mlp", lm_head="head"),
        dict(blocks="blocks",         attn="attn", mlp="mlp", lm_head="lm_head"),
        dict(blocks="model.layers",   attn="self_attn", mlp="mlp", lm_head="lm_head"),
        dict(blocks="layers",         attn="self_attn", mlp="mlp", lm_head="lm_head"),
        dict(blocks="h",              attn="attn", mlp="mlp", lm_head="lm_head"),
    ]
    for c in candidates:
        blk = _try_get(model, c["blocks"])
        if blk is None or not hasattr(blk, "__len__") or len(blk) == 0:
            continue
        first = blk[0]
        if not hasattr(first, c["attn"]) or not hasattr(first, c["mlp"]):
            continue
        if not hasattr(model, c["lm_head"]):
            continue
        return c
    raise RuntimeError(
        "tdb_hooks: could not auto-detect model structure. Pass blocks=... attn=... mlp=... lm_head=... explicitly, or use preset=...."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Handle returned to the user
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AttachHandle:
    model: nn.Module
    layout: dict
    handles: list = field(default_factory=list)   # torch hook handles
    _patch_ctx: Optional[contextlib.AbstractContextManager] = None

    @property
    def n_layer(self) -> int:
        return len(_get(self.model, self.layout["blocks"]))

    @property
    def n_head(self) -> int:
        return self.layout.get("n_head") or _layer_attr(self, "n_head") or _cfg_attr(self.model, "n_head") or 1

    @property
    def n_embd(self) -> int:
        return self.layout.get("n_embd") or _cfg_attr(self.model, "n_embd") or _cfg_attr(self.model, "hidden_size") or 0

    @property
    def vocab_size(self) -> int:
        return self.layout.get("vocab_size") or _cfg_attr(self.model, "vocab_size") or 0

    @property
    def lm_head(self) -> nn.Module:
        return _get(self.model, self.layout["lm_head"])

    def detach(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        # also turn off the SDPA monkey-patch if it leaked
        if _PATCHED["orig_sdpa"] is not None:
            F.scaled_dot_product_attention = _PATCHED["orig_sdpa"]
            _PATCHED["orig_sdpa"] = None


def _cfg_attr(model, name):
    for cfg_name in ("config", "cfg"):
        cfg = getattr(model, cfg_name, None)
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    return None


def _layer_attr(handle, name):
    blk = _get(handle.model, handle.layout["blocks"])
    if len(blk) == 0:
        return None
    attn = getattr(blk[0], handle.layout["attn"], None)
    return getattr(attn, name, None) if attn is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# Hook factories
# ──────────────────────────────────────────────────────────────────────────────
def _attn_pre_hook(layer_idx: int):
    def hook(_mod, _inp):
        if DEBUG["enabled"]:
            DEBUG["current_layer"] = layer_idx
    return hook


def _attn_fwd_hook(layer_idx: int):
    """Capture attn write magnitude + apply ablation."""
    def hook(_mod, _inp, output):
        # output may be a tensor or a tuple (HF style). Keep it consistent.
        out_tensor, *rest = output if isinstance(output, tuple) else (output,)
        cap = DEBUG.get("capture")
        if cap is not None and cap.wants(layer_idx):
            st = cap.layer_states.setdefault(layer_idx, LayerState(layer_idx))
            st.delta_attn_norm = _norm(out_tensor)
            if cap.direction is not None:
                d = cap.direction.to(out_tensor.device, out_tensor.dtype)
                st.direct_effect_attn = float((out_tensor[:, -1] @ d).mean().item())
        abl = DEBUG.get("ablate")
        if abl is not None and layer_idx in abl.attn_layers:
            out_tensor = torch.zeros_like(out_tensor)
        if isinstance(output, tuple):
            return (out_tensor, *rest)
        return out_tensor
    return hook


def _mlp_fwd_hook(layer_idx: int):
    def hook(_mod, _inp, output):
        out = output[0] if isinstance(output, tuple) else output
        cap = DEBUG.get("capture")
        if cap is not None and cap.wants(layer_idx):
            st = cap.layer_states.setdefault(layer_idx, LayerState(layer_idx))
            st.delta_mlp_norm = _norm(out)
            if cap.direction is not None:
                d = cap.direction.to(out.device, out.dtype)
                st.direct_effect_mlp = float((out[:, -1] @ d).mean().item())
        abl = DEBUG.get("ablate")
        if abl is not None and layer_idx in abl.mlp_layers:
            out = torch.zeros_like(out)
        if isinstance(output, tuple):
            return (out, *output[1:])
        return out
    return hook


def _block_pre_hook(layer_idx: int):
    def hook(_mod, inp):
        cap = DEBUG.get("capture")
        if cap is None or not cap.wants(layer_idx):
            return
        x = inp[0]
        st = cap.layer_states.setdefault(layer_idx, LayerState(layer_idx))
        st.resid_pre_norm = _norm(x)
    return hook


def _block_fwd_hook(layer_idx: int):
    def hook(_mod, _inp, output):
        cap = DEBUG.get("capture")
        if cap is None or not cap.wants(layer_idx):
            return
        x = output[0] if isinstance(output, tuple) else output
        st = cap.layer_states.setdefault(layer_idx, LayerState(layer_idx))
        st.resid_post_mlp_norm = _norm(x)
        cap.residual_post_block[layer_idx] = x.detach().float().cpu()
    return hook


def _root_pre_hook(_mod, _inp):
    """Install the SDPA / flash_attn monkey-patches at the start of forward
    when capture is enabled and probs are wanted."""
    cap = DEBUG.get("capture")
    if cap is None or not cap.want_attn_probs:
        return
    if _PATCHED["orig_sdpa"] is not None:
        return  # already patched
    _PATCHED["orig_sdpa"] = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _wrapped_sdpa
    try:
        from nanochat.flash_attention import flash_attn  # type: ignore
        _PATCHED["orig_fa_func"] = flash_attn.flash_attn_func
        flash_attn.flash_attn_func = _wrapped_fa_func
    except Exception:
        pass


def _root_post_hook(_mod, _inp, output):
    """Record final logits + residual + uninstall SDPA patch."""
    # logits could be a Tensor, a tuple (logits, loss), or HF output
    logits = None
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        logits = output[0]
    elif hasattr(output, "logits"):
        logits = output.logits
    cap = DEBUG.get("capture")
    if cap is not None and logits is not None and logits.dim() >= 2:
        cap.last_logits = logits[..., -1, :].detach().float().cpu()
    # uninstall SDPA monkey-patch
    if _PATCHED["orig_sdpa"] is not None:
        F.scaled_dot_product_attention = _PATCHED["orig_sdpa"]
        _PATCHED["orig_sdpa"] = None
    if _PATCHED["orig_fa_func"] is not None:
        try:
            from nanochat.flash_attention import flash_attn  # type: ignore
            flash_attn.flash_attn_func = _PATCHED["orig_fa_func"]
        except Exception:
            pass
        _PATCHED["orig_fa_func"] = None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def attach(
    model: nn.Module,
    blocks: Optional[str] = None,
    attn: Optional[str] = None,
    mlp: Optional[str] = None,
    lm_head: Optional[str] = None,
    preset: Optional[str] = None,
    n_head: Optional[int] = None,
    n_embd: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> AttachHandle:
    """Wire TDB hooks onto ``model``.

    Either pass a ``preset`` name (see ``tdb_hooks.PRESETS``) or override
    individual paths (``blocks="transformer.h"`` etc.). With no arguments,
    auto-detection is attempted.
    """
    from .adapters import PRESETS
    layout: dict = {}
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset '{preset}'. known: {list(PRESETS)}")
        layout.update(PRESETS[preset])
    overrides = dict(blocks=blocks, attn=attn, mlp=mlp, lm_head=lm_head,
                     n_head=n_head, n_embd=n_embd, vocab_size=vocab_size)
    layout.update({k: v for k, v in overrides.items() if v is not None})
    if not all(layout.get(k) for k in ("blocks", "attn", "mlp", "lm_head")):
        layout = {**_autodetect(model), **layout}

    blk_list = _get(model, layout["blocks"])
    handle = AttachHandle(model=model, layout=layout)

    for i, block in enumerate(blk_list):
        attn_mod = getattr(block, layout["attn"])
        mlp_mod = getattr(block, layout["mlp"])
        handle.handles.append(attn_mod.register_forward_pre_hook(_attn_pre_hook(i)))
        handle.handles.append(attn_mod.register_forward_hook(_attn_fwd_hook(i)))
        handle.handles.append(mlp_mod.register_forward_hook(_mlp_fwd_hook(i)))
        handle.handles.append(block.register_forward_pre_hook(_block_pre_hook(i)))
        handle.handles.append(block.register_forward_hook(_block_fwd_hook(i)))
    handle.handles.append(model.register_forward_pre_hook(_root_pre_hook))
    handle.handles.append(model.register_forward_hook(_root_post_hook))
    return handle


def detach(handle: AttachHandle):
    handle.detach()
