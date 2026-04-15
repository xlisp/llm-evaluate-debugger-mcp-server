"""
MathGPT × Transformer-Debugger MCP server.

A small FastMCP server that lets an LLM agent (Claude / Cursor / etc.) drive a
TDB-style layered debugging session against the MathGPT model living in
``/data/data/com.termux/files/home/pypro/MathGPT``.

Tools roughly mirror the OpenAI Transformer Debugger (TDB) workflow:

    1.  load_model              -- mount a checkpoint
    2.  tokenize                -- turn a prompt into token ids + readable tokens
    3.  run_forward             -- forward pass with debug hooks; returns a per-layer
                                   summary (write magnitudes, attn entropy, ...)
    4.  attention_distribution  -- attention probabilities for one (layer, head, query)
    5.  top_logits              -- top-k next-token candidates at the final position
    6.  direction_of_interest   -- target vs distractor logit diff and direct effects
    7.  ablate_node             -- zero out attn or mlp at chosen layers and rerun
    8.  trace_upstream          -- gradient w.r.t. earlier residual streams for a
                                   chosen logit difference (write magnitudes per layer)
    9.  attention_entropy_map   -- ascii heatmap of per-(layer,head) attention entropy
   10.  training_curve          -- read MathGPT training log/report and show loss curve
   11.  run_tdb_walkthrough     -- scripted demo on a math prompt

Additionally, generic filesystem tools (read_file / list_directory) are provided
in the spirit of ``filesystem.py`` so the agent can browse code / reports.

The model is kept on a small CPU-only forward path by default so this works in
Termux.  Any Cuda checkpoint will be downcast to float32 on load.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import textwrap
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# --- Paths --------------------------------------------------------------------
MATHGPT_ROOT = Path("/data/data/com.termux/files/home/pypro/MathGPT")
TDB_ROOT = Path("/data/data/com.termux/files/home/pypro/transformer-debugger")
DEFAULT_PROMPT = (
    "Question: A bag has 3 red balls and 4 blue balls. "
    "If I draw two balls without replacement, what is the probability "
    "they are both red?\nAnswer:"
)

# Make MathGPT importable.
if str(MATHGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(MATHGPT_ROOT))

mcp = FastMCP("llm-debugger")

# --- Lazy state ---------------------------------------------------------------
_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "meta": None,
    "checkpoint": None,
}


def _torch():
    import torch  # imported lazily so MCP startup is fast
    return torch


def _ensure_compat():
    """MathGPT ships a PyTorch 2.3 / Py 3.12 compat shim. Apply it once."""
    try:
        from math_gpt.compat import apply
        apply()
    except Exception:
        pass


def _require_model():
    if _state["model"] is None:
        raise RuntimeError(
            "No model loaded. Call load_model(...) first. "
            "Example: load_model(source='base', model_tag='d20')."
        )


def _layer_count() -> int:
    return _state["model"].config.n_layer


def _head_count() -> int:
    return _state["model"].config.n_head


# =============================================================================
# Filesystem helpers (stripped down versions of filesystem.py)
# =============================================================================
@mcp.tool()
async def read_file(file_path: str, max_chars: int = 80_000) -> str:
    """Read a text file. Useful to look at MathGPT source / reports / logs.

    Args:
        file_path: absolute or repo-relative path
        max_chars: truncate after this many characters
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return f"Error: not a file: {file_path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {file_path}: {e}"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


@mcp.tool()
async def list_directory(directory_path: str = ".") -> str:
    """List contents of a directory."""
    p = Path(directory_path)
    if not p.exists() or not p.is_dir():
        return f"Error: not a directory: {directory_path}"
    rows = []
    for item in sorted(p.iterdir()):
        kind = "DIR " if item.is_dir() else "FILE"
        size = item.stat().st_size if item.is_file() else 0
        rows.append(f"{kind} {item.name:<40} {size:>12,} bytes")
    return "\n".join(rows) if rows else "(empty)"


# =============================================================================
# Model loading / tokenization
# =============================================================================
@mcp.tool()
async def load_model(
    source: str = "base",
    model_tag: Optional[str] = None,
    step: Optional[int] = None,
    device: str = "cpu",
) -> str:
    """Load a MathGPT checkpoint.

    Args:
        source: 'base' | 'sft' | 'rl' (which checkpoint dir to look in)
        model_tag: e.g. 'd20'. None = pick largest depth available
        step: training step. None = pick last step
        device: 'cpu' | 'cuda' | 'mps'

    Returns a one-line summary of the loaded model.
    """
    _ensure_compat()
    torch = _torch()
    try:
        from nanochat.checkpoint_manager import load_model as _load
    except Exception as e:
        return f"Error importing nanochat.checkpoint_manager: {e}"
    dev = torch.device(device)
    try:
        model, tokenizer, meta = _load(source, dev, "eval", model_tag=model_tag, step=step)
    except FileNotFoundError as e:
        return (
            f"No checkpoint found ({e}). "
            f"Set NANOCHAT_BASE_DIR or train a model first. "
            f"For a smoke test you may also call load_random_model()."
        )
    except Exception as e:
        return f"Error loading checkpoint: {e}"
    _state.update(model=model, tokenizer=tokenizer, device=dev, meta=meta,
                  checkpoint=f"{source}/{model_tag}/step={step}")
    cfg = model.config
    return (
        f"Loaded {_state['checkpoint']} on {dev}. "
        f"n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd} "
        f"vocab={cfg.vocab_size} seq={cfg.sequence_len}"
    )


@mcp.tool()
async def load_random_model(n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
                            vocab_size: int = 1024, sequence_len: int = 128) -> str:
    """Build a tiny randomly-initialised MathGPT for smoke-testing the debugger.

    Useful when no real checkpoint is around (CI / first-time setup).
    """
    _ensure_compat()
    torch = _torch()
    try:
        from nanochat.gpt import GPT, GPTConfig
    except Exception as e:
        return f"Error importing nanochat.gpt: {e}"
    cfg = GPTConfig(
        sequence_len=sequence_len, vocab_size=vocab_size,
        n_layer=n_layer, n_head=n_head, n_kv_head=n_head,
        n_embd=n_embd, window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    _state.update(
        model=model,
        tokenizer=_DummyTokenizer(vocab_size),
        device=torch.device("cpu"),
        meta={"model_config": cfg.__dict__, "note": "random-init"},
        checkpoint="random",
    )
    return (
        f"Loaded random model: n_layer={cfg.n_layer} n_head={cfg.n_head} "
        f"n_embd={cfg.n_embd} vocab={cfg.vocab_size}"
    )


class _DummyTokenizer:
    """Tiny char-hash tokenizer for the random-model smoke test."""
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        return [hash(c) % self.vocab_size for c in text]

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)

    def get_vocab_size(self):
        return self.vocab_size


@mcp.tool()
async def tokenize(prompt: str = DEFAULT_PROMPT) -> str:
    """Tokenize a prompt with the loaded tokenizer.

    Returns a list of (position, token_id, token_string) lines.
    """
    _require_model()
    tok = _state["tokenizer"]
    ids = tok.encode(prompt) if hasattr(tok, "encode") else tok(prompt)
    if hasattr(ids, "ids"):
        ids = ids.ids
    out = [f"prompt={prompt!r}", f"n_tokens={len(ids)}", ""]
    for i, tid in enumerate(ids):
        try:
            piece = tok.decode([tid])
        except Exception:
            piece = "?"
        out.append(f"{i:>4}  id={tid:<6}  {piece!r}")
    return "\n".join(out)


def _tokenize_to_tensor(prompt: str):
    torch = _torch()
    tok = _state["tokenizer"]
    ids = tok.encode(prompt) if hasattr(tok, "encode") else tok(prompt)
    if hasattr(ids, "ids"):
        ids = ids.ids
    if len(ids) < 2:
        ids = ids + [0]  # GPT.forward asserts T > 1 in the smear path
    idx = torch.tensor([ids], dtype=torch.long, device=_state["device"])
    return idx, ids


# =============================================================================
# TDB-style introspection
# =============================================================================
def _direction_for_tokens(target_id: int, distractor_id: int):
    """direction = lm_head[target] - lm_head[distractor]   (TDB direction-of-interest)."""
    torch = _torch()
    W = _state["model"].lm_head.weight.detach().float()  # (V, C)
    d = (W[target_id] - W[distractor_id])
    d = d / (d.norm() + 1e-9)
    return d


@mcp.tool()
async def run_forward(prompt: str = DEFAULT_PROMPT, capture_attn: bool = True,
                      target_token: Optional[str] = None,
                      distractor_token: Optional[str] = None) -> str:
    """Forward pass with TDB hooks. Returns per-layer summary (write magnitudes,
    attention entropy, optional direct effects on target-vs-distractor direction).
    """
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh
    idx, ids = _tokenize_to_tensor(prompt)
    direction = None
    if target_token and distractor_token:
        tok = _state["tokenizer"]
        try:
            t_id = tok.encode(target_token)[0]
            d_id = tok.encode(distractor_token)[0]
            direction = _direction_for_tokens(t_id, d_id)
        except Exception as e:
            return f"Error encoding target/distractor: {e}"
    with torch.no_grad():
        with dh.capture(layers="all", attn_probs=capture_attn, direction=direction) as cap:
            _ = _state["model"](idx)
    rows = ["layer | |x|     |Δattn| |Δmlp|  H̄(attn)  DE_attn   DE_mlp"]
    rows.append("-" * 70)
    for i in range(_layer_count()):
        st = cap.layer_states.get(i)
        if st is None:
            continue
        ent = (sum(st.attn_entropy_per_head) / max(len(st.attn_entropy_per_head), 1)
               if st.attn_entropy_per_head else float("nan"))
        de_a = "    -   " if st.direct_effect_attn is None else f"{st.direct_effect_attn:+8.3f}"
        de_m = "    -   " if st.direct_effect_mlp is None else f"{st.direct_effect_mlp:+8.3f}"
        rows.append(
            f"{i:>5} | {st.resid_pre_norm:6.2f}  {st.delta_attn_norm:6.2f}  "
            f"{st.delta_mlp_norm:6.2f}  {ent:7.3f}  {de_a}  {de_m}"
        )
    rows.append("")
    rows.append(f"prompt_tokens={len(ids)}  capture_attn={capture_attn}  "
                f"direction={'yes' if direction is not None else 'no'}")
    return "\n".join(rows)


@mcp.tool()
async def attention_distribution(layer: int, head: int, query_pos: int = -1,
                                 prompt: str = DEFAULT_PROMPT, top_k: int = 8) -> str:
    """Show attention distribution at one (layer, head, query_pos).

    Returns top_k key positions with their probabilities and the decoded token.
    """
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh
    idx, ids = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with dh.capture(layers=[layer], attn_probs=True) as cap:
            _ = _state["model"](idx)
    if layer not in cap.attn_probs:
        return f"No attention captured at layer {layer}."
    p = cap.attn_probs[layer][0]  # (H, T, T)
    if head >= p.size(0):
        return f"head {head} out of range (have {p.size(0)} heads at layer {layer})"
    if query_pos < 0:
        query_pos = p.size(1) + query_pos
    row = p[head, query_pos]  # (T,)
    vals, top_idx = torch.topk(row, k=min(top_k, row.numel()))
    tok = _state["tokenizer"]
    out = [f"layer={layer} head={head} query_pos={query_pos} "
           f"(token={tok.decode([ids[query_pos]])!r})", ""]
    for v, ki in zip(vals.tolist(), top_idx.tolist()):
        try:
            piece = tok.decode([ids[ki]])
        except Exception:
            piece = "?"
        bar = "#" * int(v * 40)
        out.append(f"  key={ki:<4} p={v:6.3f} {bar} {piece!r}")
    return "\n".join(out)


@mcp.tool()
async def attention_entropy_map(prompt: str = DEFAULT_PROMPT) -> str:
    """ASCII heat-map of average attention entropy per (layer, head).

    Low entropy = sharp attention (often "name mover" / induction-style).
    High entropy = diffuse attention (often a no-op / averaging head).
    """
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh
    idx, _ids = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with dh.capture(layers="all", attn_probs=True) as cap:
            _ = _state["model"](idx)
    L, H = _layer_count(), _head_count()
    out = ["Attention entropy per (layer, head)  (lower = sharper)",
           "      " + "".join(f"h{h:<3}" for h in range(H))]
    glyphs = " .:-=+*#%@"
    all_vals = []
    for i in range(L):
        st = cap.layer_states.get(i)
        if st and st.attn_entropy_per_head:
            all_vals.extend(st.attn_entropy_per_head)
    if not all_vals:
        return "No attention probs captured."
    lo, hi = min(all_vals), max(all_vals) + 1e-9
    for i in range(L):
        st = cap.layer_states.get(i)
        ent = st.attn_entropy_per_head if st else []
        cells = []
        for h in range(H):
            if h < len(ent):
                v = (ent[h] - lo) / (hi - lo)
                g = glyphs[min(int(v * (len(glyphs) - 1)), len(glyphs) - 1)]
                cells.append(f" {g}  ")
            else:
                cells.append(" ?  ")
        out.append(f"L{i:<3}  " + "".join(cells))
    out.append(f"\nrange: low={lo:.3f}  high={hi:.3f}")
    return "\n".join(out)


@mcp.tool()
async def top_logits(prompt: str = DEFAULT_PROMPT, k: int = 10) -> str:
    """Top-k next-token logits at the final position."""
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh
    idx, _ids = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with dh.capture(layers=[]) as cap:
            _ = _state["model"](idx)
    logits = cap.last_logits[0]  # (V,)
    vals, top_idx = torch.topk(logits, k=min(k, logits.numel()))
    tok = _state["tokenizer"]
    out = [f"Top-{k} continuations after prompt:"]
    probs = torch.softmax(vals, dim=-1)
    for v, ki, pr in zip(vals.tolist(), top_idx.tolist(), probs.tolist()):
        try:
            piece = tok.decode([ki])
        except Exception:
            piece = "?"
        out.append(f"  id={ki:<6} logit={v:+7.3f} p≈{pr:5.3f}  {piece!r}")
    return "\n".join(out)


@mcp.tool()
async def direction_of_interest(prompt: str, target_token: str, distractor_token: str) -> str:
    """Measure 'why target rather than distractor' the TDB way.

    Reports the logit difference and per-layer direct effect on the
    target-minus-distractor unembedding direction.
    """
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh
    tok = _state["tokenizer"]
    try:
        t_id = tok.encode(target_token)[0]
        d_id = tok.encode(distractor_token)[0]
    except Exception as e:
        return f"Error encoding tokens: {e}"
    direction = _direction_for_tokens(t_id, d_id)
    idx, _ids = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with dh.capture(layers="all", attn_probs=False, direction=direction) as cap:
            _ = _state["model"](idx)
    last = cap.last_logits[0]
    diff = float(last[t_id] - last[d_id])
    out = [
        f"target={target_token!r} (id={t_id})  distractor={distractor_token!r} (id={d_id})",
        f"logit_diff = {diff:+.3f}   p(target)/p(distractor) ≈ {math.exp(diff):.3g}",
        "",
        "Per-layer direct effect on direction-of-interest:",
        "layer    DE(attn)    DE(mlp)",
    ]
    for i in range(_layer_count()):
        st = cap.layer_states.get(i)
        if st is None:
            continue
        a = "      -" if st.direct_effect_attn is None else f"{st.direct_effect_attn:+8.3f}"
        m = "      -" if st.direct_effect_mlp is None else f"{st.direct_effect_mlp:+8.3f}"
        out.append(f"  L{i:<3}   {a}    {m}")
    return "\n".join(out)


@mcp.tool()
async def ablate_node(prompt: str, attn_layers: str = "", mlp_layers: str = "",
                      target_token: Optional[str] = None,
                      distractor_token: Optional[str] = None) -> str:
    """Zero-ablate attention and/or MLP at chosen layers, then rerun.

    ``attn_layers`` and ``mlp_layers`` are comma-separated layer indices, e.g.
    ``"3,7"``. Reports the baseline vs ablated logit difference (if you
    supply target / distractor tokens) or just the top-1 prediction.
    """
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh

    def _parse(s):
        return [int(x) for x in s.split(",") if x.strip()]
    a_layers = _parse(attn_layers)
    m_layers = _parse(mlp_layers)
    idx, _ids = _tokenize_to_tensor(prompt)
    tok = _state["tokenizer"]

    def _summary(logits):
        if target_token and distractor_token:
            t = tok.encode(target_token)[0]
            d = tok.encode(distractor_token)[0]
            return f"logit_diff(target-distractor) = {float(logits[t] - logits[d]):+.3f}"
        v, i = torch.topk(logits, k=3)
        items = []
        for vv, ii in zip(v.tolist(), i.tolist()):
            try:
                piece = tok.decode([ii])
            except Exception:
                piece = "?"
            items.append(f"{piece!r}@{vv:+.2f}")
        return "top3=" + " ".join(items)

    with torch.no_grad():
        with dh.capture(layers=[]) as cap:
            _ = _state["model"](idx)
        base = cap.last_logits[0]
        with dh.ablate(attn_layers=a_layers, mlp_layers=m_layers):
            with dh.capture(layers=[]) as cap2:
                _ = _state["model"](idx)
        ab = cap2.last_logits[0]
    return (
        f"ablating attn_layers={a_layers} mlp_layers={m_layers}\n"
        f"  baseline:  {_summary(base)}\n"
        f"  ablated :  {_summary(ab)}"
    )


@mcp.tool()
async def trace_upstream(prompt: str, target_token: str, distractor_token: str) -> str:
    """Backprop the target-minus-distractor logit through the network and
    report each layer's act·grad on its post-block residual.

    This is the cheap "estimated total effect" TDB uses: positive = layer
    pushes toward the target, negative = pushes toward the distractor.
    """
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh
    tok = _state["tokenizer"]
    try:
        t_id = tok.encode(target_token)[0]
        d_id = tok.encode(distractor_token)[0]
    except Exception as e:
        return f"Error encoding tokens: {e}"
    idx, _ids = _tokenize_to_tensor(prompt)
    # Need grads -> can't use no_grad. Capture residuals first, but as
    # leaf tensors with grad enabled. Simplest: rerun with hooks that retain.
    model = _state["model"]
    # We rerun once, capture residuals (as grad-enabled tensors), then
    # backward from the logit difference.
    captured: dict[int, "torch.Tensor"] = {}
    handles = []
    for i, block in enumerate(model.transformer.h):
        def _mk(idx_=i):
            def hook(_mod, _inp, output):
                output.retain_grad()
                captured[idx_] = output
                return output
            return hook
        handles.append(block.register_forward_hook(_mk()))
    try:
        logits = model(idx)
        loss = logits[0, -1, t_id] - logits[0, -1, d_id]
        loss.backward()
    finally:
        for h in handles:
            h.remove()
    out = [
        f"trace upstream for {target_token!r} - {distractor_token!r}",
        f"  end logit diff = {float(loss):+.3f}",
        "",
        "layer   ||resid||    ||grad||    act·grad (last token)",
    ]
    for i in range(_layer_count()):
        if i not in captured or captured[i].grad is None:
            continue
        r = captured[i].detach().float()
        g = captured[i].grad.float()
        ag = float((r[:, -1] * g[:, -1]).sum().item())
        out.append(
            f"  L{i:<3}  {float(r.norm()):8.2f}   {float(g.norm()):8.4f}   {ag:+9.4f}"
        )
    model.zero_grad(set_to_none=True)
    return "\n".join(out)


# =============================================================================
# Training visualisation
# =============================================================================
@mcp.tool()
async def training_curve(report_path: str = "", width: int = 60, height: int = 18) -> str:
    """ASCII loss curve. Reads ``report_path`` (json or jsonl). If empty,
    auto-discovers MathGPT/reports/*.md or wandb logs and parses 'step / loss'.

    Set width/height to control the plot size.
    """
    paths = []
    if report_path:
        paths.append(Path(report_path))
    else:
        for p in (MATHGPT_ROOT / "reports").glob("*.md"):
            paths.append(p)
    if not paths:
        return "No reports found."
    pairs = []
    pat = re.compile(r"step\s*[:=]?\s*(\d+).*?loss\s*[:=]?\s*([0-9]+\.[0-9]+)", re.I)
    used = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in pat.finditer(text):
            pairs.append((int(m.group(1)), float(m.group(2))))
        used.append(str(p))
    if not pairs:
        return f"No 'step ... loss ...' pairs found in:\n  " + "\n  ".join(used)
    pairs.sort()
    xs = [s for s, _ in pairs]
    ys = [l for _, l in pairs]
    return _ascii_plot(xs, ys, width=width, height=height,
                       title=f"loss vs step ({len(pairs)} points from {len(used)} files)")


def _ascii_plot(xs, ys, width=60, height=18, title=""):
    if not xs:
        return "(empty)"
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    grid = [[" "] * width for _ in range(height)]
    for x, y in zip(xs, ys):
        cx = int((x - xmin) / (xmax - xmin) * (width - 1))
        cy = int((1 - (y - ymin) / (ymax - ymin)) * (height - 1))
        grid[cy][cx] = "*"
    lines = ["".join(row) for row in grid]
    out = [title] if title else []
    out.append(f"y={ymax:.3f} ┐ " + lines[0])
    for ln in lines[1:-1]:
        out.append("         │ " + ln)
    out.append(f"y={ymin:.3f} ┘ " + lines[-1])
    out.append("           " + f"x={xmin}".ljust(width // 2)
               + f"x={xmax}".rjust(width - width // 2))
    return "\n".join(out)


# =============================================================================
# Walkthrough demo
# =============================================================================
@mcp.tool()
async def run_tdb_walkthrough(
    prompt: str = DEFAULT_PROMPT,
    target_token: str = " 1",
    distractor_token: str = " 2",
) -> str:
    """End-to-end TDB-style debugging walk-through on one math prompt.

    Steps mirror what you'd do interactively in TDB:

      1. tokenise the prompt
      2. forward + per-layer summary
      3. attention entropy heat-map  (find sharp heads worth inspecting)
      4. attention distribution for the sharpest head, last query token
      5. logit-diff for target vs distractor
      6. trace upstream gradients
      7. ablation sweep over the 3 layers with strongest direct effect

    Returns the concatenated transcript so the caller can read it as a single
    debug session.
    """
    _require_model()
    torch = _torch()
    from nanochat import debug_hooks as dh
    sections: list[str] = []

    sections.append("== 1. Tokenisation ==")
    sections.append(await tokenize(prompt))

    sections.append("\n== 2. Per-layer forward summary ==")
    sections.append(await run_forward(prompt, capture_attn=True,
                                      target_token=target_token,
                                      distractor_token=distractor_token))

    sections.append("\n== 3. Attention entropy heat-map ==")
    sections.append(await attention_entropy_map(prompt))

    # find the (layer, head) with lowest entropy  -- "sharpest" head
    idx, ids = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with dh.capture(layers="all", attn_probs=True) as cap:
            _ = _state["model"](idx)
    best = (None, None, math.inf)
    for li, st in cap.layer_states.items():
        for hi, e in enumerate(st.attn_entropy_per_head):
            if e < best[2]:
                best = (li, hi, e)
    if best[0] is not None:
        sections.append(
            f"\n== 4. Sharpest head: layer={best[0]} head={best[1]} entropy={best[2]:.3f} =="
        )
        sections.append(await attention_distribution(best[0], best[1], -1, prompt))
    else:
        sections.append("\n== 4. No sharp head found ==")

    sections.append("\n== 5. Direction of interest ==")
    sections.append(await direction_of_interest(prompt, target_token, distractor_token))

    sections.append("\n== 6. Upstream gradient trace ==")
    sections.append(await trace_upstream(prompt, target_token, distractor_token))

    # Pick three layers with strongest |direct_effect| from step 2's capture.
    layer_scores = []
    for li, st in cap.layer_states.items():
        score = abs(st.delta_attn_norm) + abs(st.delta_mlp_norm)
        layer_scores.append((score, li))
    layer_scores.sort(reverse=True)
    top_layers = [li for _, li in layer_scores[:3]]
    sections.append("\n== 7. Ablation sweep on top-magnitude layers ==")
    for li in top_layers:
        sections.append(await ablate_node(
            prompt, attn_layers=str(li), mlp_layers="",
            target_token=target_token, distractor_token=distractor_token,
        ))
        sections.append(await ablate_node(
            prompt, attn_layers="", mlp_layers=str(li),
            target_token=target_token, distractor_token=distractor_token,
        ))
    return "\n".join(sections)


@mcp.tool()
async def status() -> str:
    """Return current debugger state (loaded model, device, etc.)."""
    if _state["model"] is None:
        return "no model loaded"
    cfg = _state["model"].config
    return json.dumps({
        "checkpoint": _state["checkpoint"],
        "device": str(_state["device"]),
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "vocab_size": cfg.vocab_size,
        "sequence_len": cfg.sequence_len,
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
