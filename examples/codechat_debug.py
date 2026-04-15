"""Standalone example: debug CodeChat with tdb_hooks.

Usage::

    pip install -e /path/to/llm-evaluate-debugger-mcp
    python examples/codechat_debug.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

CODECHAT_ROOT = Path("/data/data/com.termux/files/home/pypro/CodeChat")
sys.path.insert(0, str(CODECHAT_ROOT))

import tdb_hooks
from codechat.gpt import GPT, make_config

# 1. random small model for the demo (replace with your real ckpt loader)
cfg = make_config("d20")
cfg.depth = 4
cfg.n_embd = 128
cfg.n_head = 4
cfg.block_size = 64
cfg.vocab_size = 1024
cfg.grad_checkpoint = False
model = GPT(cfg).to("cpu").eval()

# 2. attach with preset='codechat' (blocks / attn / mlp / head)
handle = tdb_hooks.attach(model, preset="codechat")
print(f"attached: n_layer={handle.n_layer} n_head={handle.n_head}")

# 3. forward + capture
idx = torch.randint(0, cfg.vocab_size, (1, 16))
with torch.no_grad():
    with tdb_hooks.capture(attn_probs=True) as cap:
        logits, _ = model(idx)

for i in range(handle.n_layer):
    st = cap.layer_states[i]
    ent = sum(st.attn_entropy_per_head) / len(st.attn_entropy_per_head)
    print(f"L{i}  |Δattn|={st.delta_attn_norm:.2f}  "
          f"|Δmlp|={st.delta_mlp_norm:.2f}  H̄={ent:.3f}  "
          f"top-attn-shape={tuple(cap.attn_probs[i].shape)}")

# 4. attention distribution at layer 0, head 0, last token
p = cap.attn_probs[0][0, 0, -1]
print("L0 H0 last-token attn:", [round(v, 3) for v in p.tolist()])

# 5. ablate MLP at layer 2 and compare top-1
with torch.no_grad():
    base, _ = model(idx)
    with tdb_hooks.ablate(mlp_layers=[2]):
        ab, _ = model(idx)
print("baseline argmax:", base[0, -1].argmax().item(),
      "  ablated argmax:", ab[0, -1].argmax().item())

handle.detach()
