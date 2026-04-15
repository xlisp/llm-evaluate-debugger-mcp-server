"""English → Chinese translation demo, debugged with tdb_hooks.

Two modes in one script:

* **toy** (default): 30-pair corpus, 4-layer model, trains in seconds on CPU.
  Good for checking the debugger plumbing end-to-end.
* **--large**: real parallel corpus (HF ``Helsinki-NLP/opus-100`` en-zh by
  default, or a local TSV via ``--data-path``), 6-layer / 384-dim model,
  tuned for a single 1080-class GPU. Produces enough circuitry that the TDB
  walkthrough surfaces real layer specialization (not just memorization).

Why this example exists
-----------------------
The other examples (``mathgpt_debug.py`` / ``codechat_debug.py``) require
external project checkpoints. This file runs end-to-end with just ``torch``
and ``tdb_hooks``, so it:

1. lets a newcomer try tdb_hooks without cloning MathGPT or CodeChat,
2. exercises every public API (attach / capture / ablate / direction /
   attn_probs / residual snapshots),
3. gives the MCP server a reproducible target when iterating on
   ``llm_debugger.py``.

Usage
-----
::

    # quick smoke test (CPU, <1 min)
    python examples/translate_en2zh_debug.py

    # real training on OPUS-100 en-zh via HuggingFace datasets
    pip install datasets
    python examples/translate_en2zh_debug.py --large --save en2zh.pt

    # use an already-trained checkpoint, skip straight to the debugger
    python examples/translate_en2zh_debug.py --large --load en2zh.pt --skip-train

    # bring your own parallel TSV (two columns: english\\tchinese)
    python examples/translate_en2zh_debug.py --large --data-path corpus.tsv
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make tdb_hooks importable when the script is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tdb_hooks


# ──────────────────────────────────────────────────────────────────────────────
# 1. Toy parallel corpus — enough for the model to memorize patterns
# ──────────────────────────────────────────────────────────────────────────────
TOY_PAIRS: list[tuple[str, str]] = [
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
UNK = "<unk>"
PAD = "<pad>"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Corpus loaders
# ──────────────────────────────────────────────────────────────────────────────
def _looks_chinese(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def _clean_pair(en: str, zh: str) -> tuple[str, str] | None:
    en = en.strip().lower()
    zh = zh.strip()
    if not en or not zh:
        return None
    if not _looks_chinese(zh):
        return None
    return en, zh


def load_tsv(path: str, max_pairs: int | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cleaned = _clean_pair(parts[0], parts[1])
            if cleaned is None:
                continue
            pairs.append(cleaned)
            if max_pairs and len(pairs) >= max_pairs:
                break
    print(f"[data] loaded {len(pairs):,} pairs from {path}")
    return pairs


def load_hf_dataset(name: str, config: str, max_pairs: int) -> list[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "--large requires `pip install datasets` (or pass --data-path <tsv>)"
        ) from e
    split = f"train[:{max_pairs * 2}]"  # overshoot; we filter
    print(f"[data] loading {name}/{config} split={split} ...")
    ds = load_dataset(name, config, split=split)
    pairs: list[tuple[str, str]] = []
    for item in ds:
        tr = item["translation"]
        en = tr.get("en", "")
        zh = tr.get("zh", "")
        cleaned = _clean_pair(en, zh)
        if cleaned is None:
            continue
        # Length cap so char-level block_size stays tractable.
        if len(cleaned[0]) > 64 or len(cleaned[1]) > 32:
            continue
        pairs.append(cleaned)
        if len(pairs) >= max_pairs:
            break
    print(f"[data] kept {len(pairs):,} filtered pairs")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 3. Character-level tokenizer with frequency-capped vocab + <unk>
# ──────────────────────────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self, pairs: list[tuple[str, str]], max_vocab: int | None = None):
        counter: Counter = Counter()
        for en, zh in pairs:
            counter.update(en)
            counter.update(zh)
        specials = [SRC_TAG, TGT_TAG, EOS, UNK, PAD]
        if max_vocab is not None:
            keep = max_vocab - len(specials)
            chars = [c for c, _ in counter.most_common(keep)]
        else:
            chars = sorted(counter.keys())
        vocab = specials + chars
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def unk_id(self) -> int: return self.stoi[UNK]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]
    @property
    def src_id(self) -> int: return self.stoi[SRC_TAG]
    @property
    def tgt_id(self) -> int: return self.stoi[TGT_TAG]

    def ch(self, c: str) -> int:
        return self.stoi.get(c, self.unk_id)

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos.get(int(i), UNK)
            if tok in (SRC_TAG, TGT_TAG, EOS, PAD):
                continue
            out.append(tok if tok != UNK else "·")
        return "".join(out)

    def format_pair(self, en: str, zh: str) -> list[int]:
        ids = [self.src_id]
        ids += [self.ch(c) for c in en]
        ids += [self.tgt_id]
        ids += [self.ch(c) for c in zh]
        ids += [self.eos_id]
        return ids

    def state(self) -> dict:
        return {"stoi": self.stoi}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        tok = cls.__new__(cls)
        tok.stoi = dict(state["stoi"])
        tok.itos = {i: s for s, i in tok.stoi.items()}
        return tok


# ──────────────────────────────────────────────────────────────────────────────
# 4. nanoGPT-style decoder transformer (matches the `nanogpt` preset)
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
# 5. Dataset encoding + training
# ──────────────────────────────────────────────────────────────────────────────
def encode_corpus(pairs, tok: CharTokenizer, block_size: int
                  ) -> tuple[torch.Tensor, torch.Tensor, int]:
    xs, ys = [], []
    dropped = 0
    for en, zh in pairs:
        ids = tok.format_pair(en, zh)
        if len(ids) > block_size + 1:
            dropped += 1
            continue
        x = ids[:-1]
        y = ids[1:]
        # y[i] is the target for input x[i]. When x[i] == <zh>, y[i] is the
        # first Chinese char — so tgt_start must be the position of <zh>,
        # NOT one past it.
        tgt_start = x.index(tok.tgt_id)
        y_masked = [-100] * tgt_start + y[tgt_start:]
        pad = block_size - len(x)
        x = x + [tok.pad_id] * pad
        y_masked = y_masked + [-100] * pad
        xs.append(x)
        ys.append(y_masked)
    if dropped:
        print(f"[data] dropped {dropped} pairs that exceed block_size={block_size}")
    X = torch.tensor(xs, dtype=torch.long)
    Y = torch.tensor(ys, dtype=torch.long)
    return X, Y, len(xs)


def train(model: GPT, tok: CharTokenizer, X: torch.Tensor, Y: torch.Tensor,
          *, epochs: int, batch_size: int, lr: float, device: str,
          val_frac: float = 0.02, log_every: int = 50,
          sample_every: int = 500, sample_prompts: list[str] | None = None):
    N = X.size(0)
    n_val = max(1, int(N * val_frac)) if N > 200 else min(5, N)
    perm = torch.randperm(N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xval, Yval = X[val_idx], Y[val_idx]
    print(f"[train] N={N:,}  train={Xtr.size(0):,}  val={Xval.size(0):,}  "
          f"batch={batch_size}  epochs={epochs}  lr={lr}")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    steps_per_epoch = max(1, Xtr.size(0) // batch_size)
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    step = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(Xtr.size(0))
        for b in range(steps_per_epoch):
            sel = perm[b * batch_size:(b + 1) * batch_size]
            x = Xtr[sel].to(device, non_blocking=True)
            y = Ytr[sel].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step == 1 or step % log_every == 0 or step == total_steps:
                elapsed = time.time() - t0
                print(f"  ep {ep:3d}  step {step:6d}/{total_steps}  "
                      f"loss={loss.item():.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  "
                      f"{step/elapsed:.1f} it/s")
            if sample_every and sample_prompts and step % sample_every == 0:
                _print_samples(model, tok, sample_prompts, device)
        # end-of-epoch val
        with torch.no_grad():
            model.eval()
            val_losses = []
            for b in range(0, Xval.size(0), batch_size):
                x = Xval[b:b + batch_size].to(device)
                y = Yval[b:b + batch_size].to(device)
                _, vl = model(x, targets=y)
                val_losses.append(vl.item())
            vl_mean = sum(val_losses) / max(1, len(val_losses))
            model.train()
        print(f"  [epoch {ep}] val_loss={vl_mean:.4f}")
    model.eval()


def _print_samples(model, tok, prompts, device):
    model.eval()
    with torch.no_grad():
        for en in prompts:
            zh = translate(model, tok, en)
            print(f"    sample  {en!r:<30} → {zh!r}")
    model.train()


@torch.no_grad()
def translate(model: GPT, tok: CharTokenizer, english: str, max_new: int = 32) -> str:
    device = next(model.parameters()).device
    ids = [tok.src_id] + [tok.ch(c) for c in english.lower()] + [tok.tgt_id]
    idx = torch.tensor([ids], device=device)
    out_chars: list[int] = []
    for _ in range(max_new):
        logits = model(idx[:, -model.config.block_size:])
        if isinstance(logits, tuple):
            logits = logits[0]
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id:
            break
        out_chars.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out_chars)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint save / load
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(path: str, model: GPT, tok: CharTokenizer):
    torch.save({
        "cfg": asdict(model.config),
        "tok": tok.state(),
        "state_dict": model.state_dict(),
    }, path)
    print(f"[ckpt] saved → {path}")


def load_checkpoint(path: str, device: str) -> tuple[GPT, CharTokenizer]:
    blob = torch.load(path, map_location=device)
    cfg = GPTConfig(**blob["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    tok = CharTokenizer.from_state(blob["tok"])
    print(f"[ckpt] loaded ← {path}  "
          f"(n_layer={cfg.n_layer} n_embd={cfg.n_embd} vocab={cfg.vocab_size})")
    return model, tok


# ──────────────────────────────────────────────────────────────────────────────
# 7. TDB-style 7-step debug walkthrough
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
    prompt_ids = [tok.src_id] + [tok.ch(c) for c in english.lower()] + [tok.tgt_id]
    idx = torch.tensor([prompt_ids], device=device)

    # ── Step 1: tokenize ─────────────────────────────────────────────────────
    print("\n[1] tokenize")
    for i, t in enumerate(prompt_ids):
        print(f"    {i:2d}  id={t:<5}  tok={tok.itos.get(t, UNK)!r}")

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
        tok_str = tok.itos.get(prompt_ids[i], UNK)
        print(f"      pos {i:<2}  tok={tok_str!r:<10}  p={p:.3f}")

    # ── Step 5: direction of interest = W_U[target] - W_U[distractor] ────────
    W_U = handle.lm_head.weight  # (vocab, n_embd) — tied with wte
    tgt_id = tok.ch(target_char)
    dst_id = tok.ch(distractor_char)
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
# 8. main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--large", action="store_true",
                   help="scale model + use real parallel corpus (1080-class GPU)")
    p.add_argument("--hf-dataset", default="Helsinki-NLP/opus-100")
    p.add_argument("--hf-config", default="en-zh")
    p.add_argument("--data-path", default=None,
                   help="local TSV file (en<TAB>zh per line) — overrides HF")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="cap corpus size (default: 200000 in --large, all in toy)")
    p.add_argument("--max-vocab", type=int, default=None,
                   help="cap tokenizer vocab (default: 8192 in --large)")
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps", type=int, default=None,
                   help="toy-mode: total gradient steps on the tiny batch")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--save", default=None, help="save checkpoint to path")
    p.add_argument("--load", default=None, help="load checkpoint from path")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--prompt", default="hello",
                   help="English phrase to translate and debug on")
    p.add_argument("--device", default=None,
                   help="cpu / cuda / cuda:0 (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  "
          f"torch={torch.__version__}  "
          f"cuda={torch.cuda.is_available()}")

    # ── load or train ────────────────────────────────────────────────────────
    if args.load:
        model, tok = load_checkpoint(args.load, device)
        cfg = model.config
    else:
        # 1) corpus
        if args.data_path:
            pairs = load_tsv(args.data_path, args.max_pairs)
            if not pairs:
                raise SystemExit("no usable pairs from --data-path")
        elif args.large:
            max_pairs = args.max_pairs or 200_000
            pairs = load_hf_dataset(args.hf_dataset, args.hf_config, max_pairs)
        else:
            pairs = TOY_PAIRS
            print(f"[data] toy corpus: {len(pairs)} pairs")

        # 2) tokenizer
        max_vocab = args.max_vocab or (8192 if args.large else None)
        tok = CharTokenizer(pairs, max_vocab=max_vocab)
        print(f"[tok] vocab_size={tok.vocab_size}")

        # 3) model
        cfg = GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=args.block_size or (128 if args.large else 32),
            n_layer=args.n_layer or (6 if args.large else 4),
            n_head=args.n_head or (8 if args.large else 4),
            n_embd=args.n_embd or (384 if args.large else 128),
        )
        model = GPT(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] n_layer={cfg.n_layer} n_head={cfg.n_head} "
              f"n_embd={cfg.n_embd} block={cfg.block_size}  "
              f"params={n_params:,}")

        # 4) train
        if not args.skip_train:
            X, Y, N = encode_corpus(pairs, tok, cfg.block_size)
            if args.large:
                epochs = args.epochs or 3
                batch_size = args.batch_size or 64
                lr = args.lr or 3e-4
                sample_prompts = [en for en, _ in pairs[:5]]
                train(model, tok, X, Y,
                      epochs=epochs, batch_size=batch_size,
                      lr=lr, device=device,
                      sample_every=1000, sample_prompts=sample_prompts)
            else:
                # Toy mode: keep behaviour simple & fast — one batch, many steps
                steps = args.steps or 400
                lr = args.lr or 3e-3
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
                x = X.to(device); y = Y.to(device)
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
        else:
            print("[train] skipped")

        if args.save:
            save_checkpoint(args.save, model, tok)

    # ── sample translations ──────────────────────────────────────────────────
    print("\n--- sample translations ---")
    demo_prompts = ["hello", "thank you", "i love you", "goodbye"]
    if args.prompt and args.prompt not in demo_prompts:
        demo_prompts.append(args.prompt)
    for en in demo_prompts:
        print(f"  {en!r:<25} → {translate(model, tok, en)!r}")

    # ── attach debugger ──────────────────────────────────────────────────────
    handle = tdb_hooks.attach(model, preset="nanogpt")
    print(f"\ntdb_hooks attached: n_layer={handle.n_layer} n_head={handle.n_head} "
          f"n_embd={handle.n_embd} vocab={handle.vocab_size}")

    # Pick target/distractor from vocab in a model-agnostic way.
    english = args.prompt
    # If the prompt is in our demo set, use its known translation's first char
    # as the target. Otherwise, pick the model's own greedy prediction as the
    # target and a plausible distractor.
    target_char, distractor_char = _pick_target_distractor(model, tok, english)
    tdb_walkthrough(model, tok, handle,
                    english=english,
                    target_char=target_char,
                    distractor_char=distractor_char)

    handle.detach()
    print("\ntdb_hooks detached — model is back to native state.")


def _pick_target_distractor(model, tok, english: str) -> tuple[str, str]:
    """Pick a (target, distractor) pair from real Chinese characters in the
    tokenizer, using the model's own distribution if no ground truth.
    """
    # See if english is one of the toy pairs first.
    for e, zh in TOY_PAIRS:
        if e == english.strip().lower():
            tgt = zh[0]
            # distractor: first char of a different Chinese string
            for _, zh2 in TOY_PAIRS:
                if zh2[0] != tgt:
                    return tgt, zh2[0]
    # Otherwise: let the model choose. Take top-1 prediction as target,
    # a high-probability but different Chinese char as distractor.
    device = next(model.parameters()).device
    with torch.no_grad():
        ids = [tok.src_id] + [tok.ch(c) for c in english.lower()] + [tok.tgt_id]
        logits = model(torch.tensor([ids], device=device))
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = logits[0, -1].softmax(-1)
    order = probs.argsort(descending=True).tolist()
    tgt = None
    dst = None
    for i in order:
        ch = tok.itos.get(i, UNK)
        if len(ch) == 1 and "\u4e00" <= ch <= "\u9fff":
            if tgt is None:
                tgt = ch
            elif ch != tgt:
                dst = ch
                break
    return (tgt or "你"), (dst or "早")


if __name__ == "__main__":
    main()
