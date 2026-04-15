"""Capture / Ablate context managers + per-layer state container.

Model-agnostic. Used together with ``attach.py`` which wires the actual
PyTorch ``register_forward_hook`` plumbing onto a target model.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


# Process-wide singleton. Hook callbacks read these to know what to record /
# whether to ablate. Nestable via the context managers below.
DEBUG: dict[str, Any] = {
    "enabled": False,
    "capture": None,   # type: Optional["Capture"]
    "ablate": None,    # type: Optional["Ablate"]
    "current_layer": None,  # int, set by attn pre-hook so SDPA monkey-patch
                            # can route attention probs to the right layer
}


@dataclass
class LayerState:
    layer_idx: int
    resid_pre_norm: float = 0.0
    resid_post_attn_norm: float = 0.0
    resid_post_mlp_norm: float = 0.0
    delta_attn_norm: float = 0.0
    delta_mlp_norm: float = 0.0
    attn_entropy_per_head: list[float] = field(default_factory=list)
    direct_effect_attn: float | None = None
    direct_effect_mlp: float | None = None


@dataclass
class Capture:
    layers: set[int] | str = "all"
    want_attn_probs: bool = False
    layer_states: dict[int, LayerState] = field(default_factory=dict)
    # attn_probs[layer_idx] -> tensor (B, H, T, T) on cpu
    attn_probs: dict[int, torch.Tensor] = field(default_factory=dict)
    # snapshot of post-block residual stream, fp32 cpu
    residual_post_block: dict[int, torch.Tensor] = field(default_factory=dict)
    final_resid: torch.Tensor | None = None
    last_logits: torch.Tensor | None = None
    direction: torch.Tensor | None = None  # unit vector in residual space

    def wants(self, layer_idx: int) -> bool:
        return self.layers == "all" or layer_idx in self.layers


@dataclass
class Ablate:
    attn_layers: set[int] = field(default_factory=set)
    mlp_layers: set[int] = field(default_factory=set)
    attn_head_mask: dict[int, set[int]] = field(default_factory=dict)


@contextlib.contextmanager
def capture(layers="all", attn_probs: bool = False,
            direction: torch.Tensor | None = None):
    layer_set = layers if layers == "all" else set(layers)
    cap = Capture(layers=layer_set, want_attn_probs=attn_probs, direction=direction)
    prev = DEBUG.copy()
    DEBUG.update(enabled=True, capture=cap)
    try:
        yield cap
    finally:
        DEBUG["capture"] = prev["capture"]
        DEBUG["enabled"] = (DEBUG["capture"] is not None) or (DEBUG["ablate"] is not None)


@contextlib.contextmanager
def ablate(attn_layers=(), mlp_layers=(), attn_head_mask=None):
    abl = Ablate(
        attn_layers=set(attn_layers),
        mlp_layers=set(mlp_layers),
        attn_head_mask={k: set(v) for k, v in (attn_head_mask or {}).items()},
    )
    prev = DEBUG.copy()
    DEBUG.update(enabled=True, ablate=abl)
    try:
        yield abl
    finally:
        DEBUG["ablate"] = prev["ablate"]
        DEBUG["enabled"] = (DEBUG["capture"] is not None) or (DEBUG["ablate"] is not None)


def is_active() -> bool:
    return DEBUG["enabled"]


def _norm(x: torch.Tensor) -> float:
    try:
        return float(x.detach().float().norm().item())
    except Exception:
        return float("nan")
