"""Standalone example: debug MathGPT with tdb_hooks.

Usage::

    pip install -e /path/to/llm-evaluate-debugger-mcp   # installs tdb_hooks
    python examples/mathgpt_debug.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

# Make MathGPT importable
MATHGPT_ROOT = Path("/data/data/com.termux/files/home/pypro/MathGPT")
sys.path.insert(0, str(MATHGPT_ROOT))
from math_gpt.compat import apply; apply()

import tdb_hooks
from nanochat.checkpoint_manager import load_model

# 1. load a checkpoint -- pass model_tag/step that exist in your nanochat_base
device = torch.device("cpu")
model, tokenizer, meta = load_model("base", device, "eval")

# 2. one-line attach -- preset='mathgpt' matches transformer.h / attn / mlp / lm_head
handle = tdb_hooks.attach(model, preset="mathgpt")
print(f"attached: n_layer={handle.n_layer} n_head={handle.n_head}")

# 3. forward with capture
prompt = "Question: 3+4=?\nAnswer:"
ids = tokenizer.encode(prompt)
idx = torch.tensor([ids], dtype=torch.long, device=device)

with torch.no_grad():
    with tdb_hooks.capture(attn_probs=True) as cap:
        _ = model(idx)

for i in range(handle.n_layer):
    st = cap.layer_states[i]
    ent = sum(st.attn_entropy_per_head) / len(st.attn_entropy_per_head)
    print(f"L{i:<2}  |Δattn|={st.delta_attn_norm:6.2f}  "
          f"|Δmlp|={st.delta_mlp_norm:6.2f}  H̄={ent:.3f}")

# 4. ablate one attention layer and compare
with torch.no_grad():
    base, _ = (lambda o: (o, None) if torch.is_tensor(o) else o)(model(idx))
    with tdb_hooks.ablate(attn_layers=[5]):
        ab, _ = (lambda o: (o, None) if torch.is_tensor(o) else o)(model(idx))
print("baseline argmax:", base[0, -1].argmax().item(),
      "  ablated argmax:", ab[0, -1].argmax().item())

handle.detach()
