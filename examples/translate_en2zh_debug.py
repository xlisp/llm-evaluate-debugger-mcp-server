"""English → Chinese translation demo, debugged with tdb_hooks.

This example is **self-contained**: it defines a small nanoGPT-style
decoder-only transformer, trains it for a few hundred steps on a toy
parallel corpus, and then runs the full 7-step TDB debug walkthrough
on a real translation forward pass.

Why this example exists
-----------------------
The other examples (``mathgpt_debug.py`` / ``codechat_debug.py``) require
external project checkpoints. This file runs end-to-end on a CPU with
nothing but ``torch`` and ``tdb_hooks`` installed, so it:

1. makes ``tdb_hooks`` easy to try for a newcomer,
2. exercises every public API of the package (attach / capture /
   ablate / direction / attn_probs / residual snapshots),
3. gives the MCP server something real to point at when we iterate on
   ``llm_debugger.py`` — run ``load_random(project="translate_en2zh")``
   style flows against a model whose behaviour you can actually inspect.

Usage::

    pip install -e .
    python examples/translate_en2zh_debug.py

Pass ``--skip-train`` to skip training and go straight to the debugger
on a randomly-initialized model (useful when hacking on tdb_hooks
itself — forward pass still exercises every hook).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make tdb_hooks importable when the script is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tdb_hooks


# ──────────────────────────────────────────────────────────────────────────────
# 1. Tiny parallel corpus — enough for the model to memorize patterns
# ──────────────────────────────────────────────────────────────────────────────
PAIRS: list[tuple[str, str]] = [
    ("hello", "你好"),
    ("good morning", "早上好"),
    ("good night", "晚安"),
    ("thank you", "谢谢"),
    ("i love you", "我爱你"),
    ("how are you", "你好吗"),
    ("i am fine", "我很好"),
    ("what is your name", "你叫什么名字"),
    ("my name is claude", "我叫克劳德"),
    ("where are you from", "你来自哪里"),
    ("i am from china", "我来自中国"),
    ("do you speak english", "你会说英语吗"),
    ("yes i do", "是的我会"),
    ("no i do not", "不我不会"),
    ("see you tomorrow", "明天见"),
    ("goodbye", "再见"),
    ("this is a cat", "这是一只猫"),
    ("this is a dog", "这是一只狗"),
    ("i like apples", "我喜欢苹果"),
    ("i drink water", "我喝水"),
    ("i eat rice", "我吃米饭"),
    ("it is cold today", "今天很冷"),
    ("it is hot today", "今天很热"),
    ("the sky is blue", "天空是蓝色的"),
    ("the sun is bright", "太阳很亮"),
    ("i am a student", "我是学生"),
    ("i am a teacher", "我是老师"),
    ("open the door", "打开门"),
    ("close the window", "关上窗户"),
    ("read a book", "读一本书"),
]

SRC_TAG = "<en>"
TGT_TAG = "<zh>"
EOS = "<eos>"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Character-level tokenizer that handles ASCII + CJK
# ──────────────────────────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self, pairs: list[tuple[str, str]]):
        chars: set[str] = set()
        for en, zh in pairs:
            chars.update(en)
            chars.update(zh)
        specials = [SRC_TAG, TGT_TAG, EOS, "<pad>"]
        vocab = specials + sorted(chars)
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, text: str, *, specials: list[str] = ()) -> list[int]:
        ids = [self.stoi[s] for s in specials]
        for ch in text:
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok in (SRC_TAG, TGT_TAG, EOS, "<pad>"):
                continue
            out.append(tok)
        return "".join(out)

    def format_pair(self, en: str, zh: str) -> list[int]:
        ids = [self.stoi[SRC_TAG]]
        ids += [self.stoi[c] for c in en]
        ids += [self.stoi[TGT_TAG]]
        ids += [self.stoi[c] for c in zh]
        ids += [self.stoi[EOS]]
        return ids


# ──────────────────────────────────────────────────────────────────────────────
# 3. nanoGPT-style decoder transformer (matches the `nanogpt` preset)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Route through F.scaled_dot_product_attention so tdb_hooks'
        # monkey-patch can intercept the attention probabilities.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # tied embeddings

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        return logits, loss


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training
# ──────────────────────────────────────────────────────────────────────────────
def build_batch(tok: CharTokenizer, pairs, block_size: int, device):
    xs, ys = [], []
    for en, zh in pairs:
        ids = tok.format_pair(en, zh)
        ids = ids[: block_size + 1]
        x = ids[:-1]
        y = ids[1:]
        # Only learn to predict the Chinese side + EOS.
        tgt_start = x.index(tok.stoi[TGT_TAG]) + 1
        y_masked = [-100] * tgt_start + y[tgt_start:]
        pad = block_size - len(x)
        x = x + [tok.pad_id] * pad
        y_masked = y_masked + [-100] * pad
        xs.append(x)
        ys.append(y_masked)
    return (torch.tensor(xs, device=device),
            torch.tensor(ys, device=device))


def train(model, tok, pairs, *, steps=400, lr=3e-3, device="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x, y = build_batch(tok, pairs, model.config.block_size, device)
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"  step {step:4d}  loss={loss.item():.4f}")
    model.eval()


@torch.no_grad()
def translate(model, tok, english: str, max_new: int = 20) -> str:
    device = next(model.parameters()).device
    ids = [tok.stoi[SRC_TAG]] + [tok.stoi[c] for c in english] + [tok.stoi[TGT_TAG]]
    idx = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits = model(idx[:, -model.config.block_size:])
        if isinstance(logits, tuple):
            logits = logits[0]
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    out = idx[0].tolist()
    tgt_start = out.index(tok.stoi[TGT_TAG]) + 1
    return tok.decode(out[tgt_start:])


# ──────────────────────────────────────────────────────────────────────────────
# 5. TDB-style 7-step debug walkthrough
# ──────────────────────────────────────────────────────────────────────────────
def attention_entropy_ascii(entropies: list[list[float]]) -> str:
    """entropies[layer][head] -> ASCII heatmap of attention entropy.

    Low entropy (sharp head) = ``#``; high entropy (diffuse) = ``.``.
    """
    if not entropies:
        return "(no entropies)"
    flat = [e for row in entropies for e in row]
    lo, hi = min(flat), max(flat)
    rng = (hi - lo) or 1.0
    ramp = " .:-=+*#%@"
    rows = ["      " + "".join(f"H{h:<2}" for h in range(len(entropies[0])))]
    for i, row in enumerate(entropies):
        cells = []
        for e in row:
            idx = int((1.0 - (e - lo) / rng) * (len(ramp) - 1))
            cells.append(f" {ramp[idx]} ")
        rows.append(f"L{i:<3} " + "".join(cells))
    return "\n".join(rows)


def tdb_walkthrough(model: GPT, tok: CharTokenizer, handle, *,
                    english: str, target_char: str, distractor_char: str):
    print("\n" + "=" * 70)
    print(f"TDB WALKTHROUGH — translate: {english!r}")
    print(f"  target     = {target_char!r}  (expected first Chinese char)")
    print(f"  distractor = {distractor_char!r}")
    print("=" * 70)

    device = next(model.parameters()).device
    prompt_ids = [tok.stoi[SRC_TAG]] + [tok.stoi[c] for c in english] + [tok.stoi[TGT_TAG]]
    idx = torch.tensor([prompt_ids], device=device)

    # ── Step 1: tokenize ─────────────────────────────────────────────────────
    print("\n[1] tokenize")
    for i, t in enumerate(prompt_ids):
        print(f"    {i:2d}  id={t:<3}  tok={tok.itos[t]!r}")

    # ── Step 2: forward + per-layer write magnitudes ─────────────────────────
    print("\n[2] run_forward  (|Δattn|, |Δmlp|, entropy per layer)")
    with torch.no_grad(), tdb_hooks.capture(attn_probs=True) as cap:
        logits = model(idx)
        if isinstance(logits, tuple):
            logits = logits[0]

    for i in range(handle.n_layer):
        st = cap.layer_states[i]
        entropy_mean = (sum(st.attn_entropy_per_head) / len(st.attn_entropy_per_head)
                        if st.attn_entropy_per_head else float("nan"))
        print(f"    L{i}  |resid|={st.resid_pre_norm:7.2f}  "
              f"|Δattn|={st.delta_attn_norm:7.2f}  "
              f"|Δmlp|={st.delta_mlp_norm:7.2f}  H̄={entropy_mean:.3f}")

    # ── Step 3: attention entropy heat-map ───────────────────────────────────
    print("\n[3] attention_entropy_map  (low=sharp #,  high=diffuse .)")
    ents = [cap.layer_states[i].attn_entropy_per_head for i in range(handle.n_layer)]
    print(attention_entropy_ascii(ents))

    # ── Step 4: sharpest head — which token does it attend to? ───────────────
    sharp_layer, sharp_head, sharp_val = 0, 0, float("inf")
    for L in range(handle.n_layer):
        for H, e in enumerate(cap.layer_states[L].attn_entropy_per_head):
            if e < sharp_val:
                sharp_val, sharp_layer, sharp_head = e, L, H
    probs = cap.attn_probs[sharp_layer][0, sharp_head, -1]
    topk = torch.topk(probs, k=min(5, probs.numel()))
    print(f"\n[4] attention_distribution  (sharpest head L{sharp_layer} H{sharp_head}, "
          f"entropy={sharp_val:.3f})")
    print(f"    last-token attends to:")
    for p, i in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"      pos {i:<2}  tok={tok.itos[prompt_ids[i]]!r:<8}  p={p:.3f}")

    # ── Step 5: direction of interest = W_U[target] - W_U[distractor] ────────
    W_U = handle.lm_head.weight  # (vocab, n_embd) — tied with wte
    tgt_id = tok.stoi[target_char]
    dst_id = tok.stoi[distractor_char]
    direction = (W_U[tgt_id] - W_U[dst_id]).detach().float()
    direction = direction / (direction.norm() + 1e-9)

    print("\n[5] direction_of_interest  (direct effect of each layer onto target−distractor)")
    with torch.no_grad(), tdb_hooks.capture(direction=direction) as cap2:
        _ = model(idx)
    for i in range(handle.n_layer):
        st = cap2.layer_states[i]
        de_a = st.direct_effect_attn if st.direct_effect_attn is not None else 0.0
        de_m = st.direct_effect_mlp if st.direct_effect_mlp is not None else 0.0
        arrow_a = "+" if de_a > 0 else "-"
        arrow_m = "+" if de_m > 0 else "-"
        print(f"    L{i}  attn→{arrow_a}{abs(de_a):6.3f}   mlp→{arrow_m}{abs(de_m):6.3f}")

    # ── Step 6: trace_upstream (grad of logit diff wrt each residual post-block) ─
    print("\n[6] trace_upstream  (estimated total effect via real backward)")
    with tdb_hooks.capture() as cap3:
        logits = model(idx)
        if isinstance(logits, tuple):
            logits = logits[0]
        last = logits[0, -1]
        logit_diff = last[tgt_id] - last[dst_id]
    # Re-run forward with grad enabled so residuals stay in the graph.
    model.zero_grad(set_to_none=True)
    acts: dict[int, torch.Tensor] = {}

    def _block_capture(i):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            acts[i] = t
        return hook

    tmp_handles = []
    for i, block in enumerate(model.transformer.h):
        tmp_handles.append(block.register_forward_hook(_block_capture(i)))
    try:
        logits = model(idx)
        if isinstance(logits, tuple):
            logits = logits[0]
        last = logits[0, -1]
        (last[tgt_id] - last[dst_id]).backward()
    finally:
        for h in tmp_handles:
            h.remove()
    for i in range(handle.n_layer):
        a = acts[i]
        g = a.grad
        ete = float((a[:, -1] * g[:, -1]).sum().item()) if g is not None else 0.0
        print(f"    L{i}  estimated_total_effect={ete:+.4f}")

    # ── Step 7: ablate the top-candidate layer, compare logit diff ───────────
    with torch.no_grad():
        logits = model(idx)
        if isinstance(logits, tuple):
            logits = logits[0]
        base_diff = float((logits[0, -1, tgt_id] - logits[0, -1, dst_id]).item())
    # Pick the layer whose MLP direct_effect most supports the target.
    best_layer = max(range(handle.n_layer),
                     key=lambda i: cap2.layer_states[i].direct_effect_mlp or 0.0)
    with torch.no_grad(), tdb_hooks.ablate(mlp_layers=[best_layer]):
        logits = model(idx)
        if isinstance(logits, tuple):
            logits = logits[0]
        abl_diff = float((logits[0, -1, tgt_id] - logits[0, -1, dst_id]).item())
    print(f"\n[7] ablate_node  (MLP L{best_layer} — top positive direct-effect)")
    print(f"    baseline logit_diff(target−distractor) = {base_diff:+.4f}")
    print(f"    ablated  logit_diff(target−distractor) = {abl_diff:+.4f}")
    print(f"    Δ = {abl_diff - base_diff:+.4f}  "
          f"{'(confirmed — ablating this layer hurts the target)' if abl_diff < base_diff else '(no drop — try another layer)'}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true",
                   help="skip training, debug a random-weights model")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--prompt", type=str, default="hello",
                   help="English phrase to translate and debug on")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cpu"

    tok = CharTokenizer(PAIRS)
    print(f"vocab_size = {tok.vocab_size}  "
          f"(specials + ascii + CJK chars from {len(PAIRS)} pairs)")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"n_embd={cfg.n_embd}  params={n_params:,}")

    if not args.skip_train:
        print(f"\n--- training {args.steps} steps on {len(PAIRS)} pairs ---")
        train(model, tok, PAIRS, steps=args.steps, device=device)
    else:
        print("\n--- skipping training (random weights) ---")

    print("\n--- sample translations ---")
    for en in ["hello", "thank you", "i love you", "goodbye", args.prompt]:
        print(f"  {en!r:<20} → {translate(model, tok, en)!r}")

    # Attach tdb_hooks — the model matches the `nanogpt` preset exactly.
    handle = tdb_hooks.attach(model, preset="nanogpt")
    print(f"\ntdb_hooks attached: n_layer={handle.n_layer} n_head={handle.n_head} "
          f"n_embd={handle.n_embd} vocab={handle.vocab_size}")

    # Pick a target/distractor pair from the training data so the signal is real.
    english = args.prompt if args.prompt in {e for e, _ in PAIRS} else "hello"
    expected_zh = next(zh for e, zh in PAIRS if e == english)
    target_char = expected_zh[0]
    # Distractor = first char of a *different* translation.
    distractor_char = next(zh for e, zh in PAIRS if e != english)[0]
    if distractor_char == target_char:
        distractor_char = next(zh for e, zh in PAIRS
                               if zh[0] != target_char)[0]

    tdb_walkthrough(model, tok, handle,
                    english=english,
                    target_char=target_char,
                    distractor_char=distractor_char)

    handle.detach()
    print("\ntdb_hooks detached — model is back to native state.")


if __name__ == "__main__":
    main()
