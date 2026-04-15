"""tdb_hooks — pip-installable TDB-style debugger for any nanoGPT-like model.

Quickstart::

    import tdb_hooks
    handle = tdb_hooks.attach(model)               # auto-detect
    # or:  handle = tdb_hooks.attach(model, preset="mathgpt")
    # or:  handle = tdb_hooks.attach(model, blocks="transformer.h",
    #                                attn="attn", mlp="mlp", lm_head="lm_head")

    with tdb_hooks.capture(attn_probs=True) as cap:
        out = model(idx)
    cap.layer_states[3].delta_attn_norm

    with tdb_hooks.ablate(attn_layers=[5]):
        out = model(idx)

    handle.detach()  # remove all hooks
"""
from .core import Capture, Ablate, LayerState, capture, ablate, is_active
from .attach import attach, detach, AttachHandle
from .adapters import PRESETS, register_preset

__all__ = [
    "attach", "detach", "AttachHandle",
    "capture", "ablate", "is_active",
    "Capture", "Ablate", "LayerState",
    "PRESETS", "register_preset",
]

__version__ = "0.1.0"
