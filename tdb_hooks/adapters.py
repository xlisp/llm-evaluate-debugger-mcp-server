"""Named layout presets for known projects.

Add your own with ``tdb_hooks.register_preset('myproj', dict(...))``.
"""
from __future__ import annotations
from typing import Dict


PRESETS: Dict[str, dict] = {
    # MathGPT (forked from nanochat)
    "mathgpt": dict(
        blocks="transformer.h",
        attn="attn",
        mlp="mlp",
        lm_head="lm_head",
    ),
    "nanochat": dict(
        blocks="transformer.h",
        attn="attn",
        mlp="mlp",
        lm_head="lm_head",
    ),
    # CodeChat (minGPT-style flat layout)
    "codechat": dict(
        blocks="blocks",
        attn="attn",
        mlp="mlp",
        lm_head="head",
    ),
    # nanoGPT (Karpathy)
    "nanogpt": dict(
        blocks="transformer.h",
        attn="attn",
        mlp="mlp",
        lm_head="lm_head",
    ),
    # HuggingFace GPT-2 style
    "hf_gpt2": dict(
        blocks="transformer.h",
        attn="attn",
        mlp="mlp",
        lm_head="lm_head",
    ),
    # LLaMA-style
    "llama": dict(
        blocks="model.layers",
        attn="self_attn",
        mlp="mlp",
        lm_head="lm_head",
    ),
}


def register_preset(name: str, layout: dict):
    """Register or override a preset.

    layout must contain at least: blocks, attn, mlp, lm_head. May also include
    n_head, n_embd, vocab_size to override config sniffing.
    """
    required = {"blocks", "attn", "mlp", "lm_head"}
    missing = required - set(layout)
    if missing:
        raise ValueError(f"preset '{name}' missing keys: {missing}")
    PRESETS[name] = dict(layout)
