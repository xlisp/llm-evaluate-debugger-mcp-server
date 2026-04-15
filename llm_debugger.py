"""MCP server: TDB-style debugger driven by the ``tdb_hooks`` package.

Project-agnostic: at runtime you tell it which project / preset to attach to
(``mathgpt``, ``codechat``, ...) and the same set of tools work.

Run::
    python llm_debugger.py
or with Claude Code::
    "llm-debugger": {"command":"python","args":["/path/to/llm_debugger.py"]}
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# Local package (this script lives next to it)
sys.path.insert(0, str(Path(__file__).parent))
import tdb_hooks  # noqa: E402

# --- Project registry --------------------------------------------------------
# Each entry: how to find the source tree, which preset to attach with, and
# how to load a checkpoint into a model object.
PROJECTS: dict[str, dict[str, Any]] = {
    "mathgpt": {
        "root": Path("/data/data/com.termux/files/home/pypro/MathGPT"),
        "preset": "mathgpt",
        "loader": "mathgpt_loader",
    },
    "codechat": {
        "root": Path("/data/data/com.termux/files/home/pypro/CodeChat"),
        "preset": "codechat",
        "loader": "codechat_loader",
    },
}

DEFAULT_PROMPT_BY_PROJECT = {
    "mathgpt": (
        "Question: A bag has 3 red balls and 4 blue balls. "
        "If I draw two balls without replacement, what is the probability "
        "they are both red?\nAnswer:"
    ),
    "codechat": (
        "Write a Python function `is_prime(n)` that returns True if n is prime.\n"
        "def is_prime(n):\n"
    ),
}

mcp = FastMCP("llm-debugger")

_state: dict[str, Any] = {
    "project": None,
    "model": None,
    "tokenizer": None,
    "device": None,
    "handle": None,        # tdb_hooks.AttachHandle
    "checkpoint_id": None,
}


def _torch():
    import torch
    return torch


def _require():
    if _state["model"] is None:
        raise RuntimeError("Call load_model(project='mathgpt'|'codechat') first.")


# ============================================================================
# Per-project loaders
# ============================================================================
def mathgpt_loader(model_tag: Optional[str], step: Optional[int], device: str):
    root = PROJECTS["mathgpt"]["root"]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from math_gpt.compat import apply
        apply()
    except Exception:
        pass
    from nanochat.checkpoint_manager import load_model as _load
    import torch
    dev = torch.device(device)
    model, tokenizer, meta = _load("base", dev, "eval", model_tag=model_tag, step=step)
    return model, tokenizer, dev, meta


def codechat_loader(model_tag: Optional[str], step: Optional[int], device: str):
    root = PROJECTS["codechat"]["root"]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from codechat.gpt import GPT, GPTConfig, make_config, PRESETS
    from codechat.checkpoint import load_checkpoint  # type: ignore
    from codechat.tokenizer import get_tokenizer  # type: ignore
    dev = torch.device(device)
    cfg = make_config(model_tag or "d20")
    model = GPT(cfg).to(dev)
    if step is not None:
        load_checkpoint(model, step=step, device=dev)
    model.eval()
    tokenizer = get_tokenizer()
    meta = {"model_config": cfg.__dict__}
    return model, tokenizer, dev, meta


def random_loader(preset: str, n_layer: int, n_head: int, n_embd: int,
                  vocab_size: int, sequence_len: int, device: str):
    """Build a tiny random model for whichever project's GPT class."""
    import torch
    if preset in ("mathgpt", "nanochat"):
        root = PROJECTS["mathgpt"]["root"]
        sys.path.insert(0, str(root))
        try:
            from math_gpt.compat import apply
            apply()
        except Exception:
            pass
        from nanochat.gpt import GPT, GPTConfig
        cfg = GPTConfig(sequence_len=sequence_len, vocab_size=vocab_size,
                        n_layer=n_layer, n_head=n_head, n_kv_head=n_head,
                        n_embd=n_embd, window_pattern="L")
        with torch.device("meta"):
            model = GPT(cfg)
        model.to_empty(device=device)
        model.init_weights()
    elif preset == "codechat":
        root = PROJECTS["codechat"]["root"]
        sys.path.insert(0, str(root))
        from codechat.gpt import GPT, GPTConfig
        cfg = GPTConfig(vocab_size=vocab_size, depth=n_layer, n_embd=n_embd,
                        n_head=n_head, block_size=sequence_len, dropout=0.0,
                        grad_checkpoint=False)
        model = GPT(cfg).to(device)
    else:
        raise KeyError(f"random_loader: unknown preset {preset}")
    model.eval()
    return model, _DummyTokenizer(vocab_size), torch.device(device)


class _DummyTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
    def encode(self, text: str):
        return [hash(c) % self.vocab_size for c in text]
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)
    def get_vocab_size(self):
        return self.vocab_size


# ============================================================================
# MCP tools — model lifecycle
# ============================================================================
@mcp.tool()
async def list_projects() -> str:
    """List known target projects (preset + root path)."""
    rows = []
    for name, p in PROJECTS.items():
        rows.append(f"  {name:<10}  preset={p['preset']:<10}  root={p['root']}")
    rows.append("")
    rows.append("Custom projects: register at runtime via tdb_hooks.register_preset()")
    return "Known projects:\n" + "\n".join(rows)


@mcp.tool()
async def load_model(project: str, model_tag: Optional[str] = None,
                     step: Optional[int] = None, device: str = "cpu") -> str:
    """Load a real checkpoint for ``project`` ('mathgpt' or 'codechat') and
    attach tdb_hooks. After this call, all the inspection tools work.
    """
    if project not in PROJECTS:
        return f"unknown project '{project}'. known: {list(PROJECTS)}"
    loader = globals()[PROJECTS[project]["loader"]]
    try:
        model, tokenizer, dev, meta = loader(model_tag, step, device)
    except Exception as e:
        return f"loader error: {type(e).__name__}: {e}"
    _detach_existing()
    handle = tdb_hooks.attach(model, preset=PROJECTS[project]["preset"])
    _state.update(project=project, model=model, tokenizer=tokenizer, device=dev,
                  handle=handle, checkpoint_id=f"{project}/{model_tag}/step={step}")
    return (
        f"loaded {_state['checkpoint_id']} on {dev}; "
        f"hooks attached via preset='{PROJECTS[project]['preset']}'; "
        f"n_layer={handle.n_layer} n_head={handle.n_head} n_embd={handle.n_embd}"
    )


@mcp.tool()
async def load_random(project: str = "codechat", n_layer: int = 4, n_head: int = 4,
                      n_embd: int = 128, vocab_size: int = 1024,
                      sequence_len: int = 128, device: str = "cpu") -> str:
    """Build a randomly-initialised model for the chosen project (smoke test)."""
    if project not in PROJECTS:
        return f"unknown project '{project}'. known: {list(PROJECTS)}"
    try:
        model, tokenizer, dev = random_loader(
            PROJECTS[project]["preset"], n_layer, n_head, n_embd,
            vocab_size, sequence_len, device,
        )
    except Exception as e:
        return f"random_loader error: {type(e).__name__}: {e}"
    _detach_existing()
    handle = tdb_hooks.attach(model, preset=PROJECTS[project]["preset"])
    _state.update(project=project, model=model, tokenizer=tokenizer, device=dev,
                  handle=handle, checkpoint_id=f"{project}/random")
    return (
        f"random {project} model: n_layer={handle.n_layer} n_head={handle.n_head} "
        f"n_embd={handle.n_embd}; hooks attached"
    )


def _detach_existing():
    if _state.get("handle") is not None:
        try:
            _state["handle"].detach()
        except Exception:
            pass
    _state["handle"] = None


@mcp.tool()
async def detach_hooks() -> str:
    """Remove all tdb_hooks from the currently-loaded model."""
    if _state["handle"] is None:
        return "no hooks attached"
    _state["handle"].detach()
    _state["handle"] = None
    return "detached"


@mcp.tool()
async def status() -> str:
    """Show currently-loaded model info."""
    if _state["model"] is None:
        return "no model loaded"
    h = _state["handle"]
    return json.dumps({
        "project": _state["project"],
        "checkpoint": _state["checkpoint_id"],
        "device": str(_state["device"]),
        "n_layer": h.n_layer if h else None,
        "n_head": h.n_head if h else None,
        "n_embd": h.n_embd if h else None,
        "hooks_attached": h is not None,
    }, indent=2)


# ============================================================================
# MCP tools — TDB-style introspection
# ============================================================================
def _default_prompt() -> str:
    return DEFAULT_PROMPT_BY_PROJECT.get(_state["project"], "Hello world")


def _tokenize_to_tensor(prompt: str):
    torch = _torch()
    tok = _state["tokenizer"]
    ids = tok.encode(prompt) if hasattr(tok, "encode") else tok(prompt)
    if hasattr(ids, "ids"):
        ids = ids.ids
    if len(ids) < 2:
        ids = ids + [0]
    return torch.tensor([ids], dtype=torch.long, device=_state["device"]), ids


def _direction(target_id: int, distractor_id: int):
    torch = _torch()
    W = _state["handle"].lm_head.weight.detach().float()
    d = (W[target_id] - W[distractor_id])
    return d / (d.norm() + 1e-9)


@mcp.tool()
async def tokenize(prompt: str = "") -> str:
    """Tokenise a prompt; show (pos, id, piece)."""
    _require()
    prompt = prompt or _default_prompt()
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


@mcp.tool()
async def run_forward(prompt: str = "", capture_attn: bool = True,
                      target_token: Optional[str] = None,
                      distractor_token: Optional[str] = None) -> str:
    """Forward pass with TDB capture; returns per-layer summary."""
    _require()
    torch = _torch()
    prompt = prompt or _default_prompt()
    idx, ids = _tokenize_to_tensor(prompt)
    direction = None
    if target_token and distractor_token:
        tok = _state["tokenizer"]
        try:
            t_id = tok.encode(target_token)[0]
            d_id = tok.encode(distractor_token)[0]
            direction = _direction(t_id, d_id)
        except Exception as e:
            return f"target/distractor encode error: {e}"
    with torch.no_grad():
        with tdb_hooks.capture(layers="all", attn_probs=capture_attn,
                               direction=direction) as cap:
            _ = _state["model"](idx)
    rows = ["layer | |x|     |Δattn|  |Δmlp|  H̄(attn)  DE_attn   DE_mlp"]
    rows.append("-" * 72)
    h = _state["handle"]
    for i in range(h.n_layer):
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
    rows.append(f"prompt_tokens={len(ids)} capture_attn={capture_attn} "
                f"direction={'yes' if direction is not None else 'no'} project={_state['project']}")
    return "\n".join(rows)


@mcp.tool()
async def attention_distribution(layer: int, head: int, query_pos: int = -1,
                                 prompt: str = "", top_k: int = 8) -> str:
    """Top-k attention probabilities at one (layer, head, query_pos)."""
    _require()
    torch = _torch()
    prompt = prompt or _default_prompt()
    idx, ids = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with tdb_hooks.capture(layers=[layer], attn_probs=True) as cap:
            _ = _state["model"](idx)
    if layer not in cap.attn_probs:
        return f"no attention captured at layer {layer}"
    p = cap.attn_probs[layer][0]  # (H, T, T)
    if head >= p.size(0):
        return f"head {head} out of range (have {p.size(0)})"
    if query_pos < 0:
        query_pos = p.size(1) + query_pos
    row = p[head, query_pos]
    vals, top_idx = torch.topk(row, k=min(top_k, row.numel()))
    tok = _state["tokenizer"]
    out = [f"layer={layer} head={head} query_pos={query_pos} "
           f"(token={tok.decode([ids[query_pos]])!r})", ""]
    for v, ki in zip(vals.tolist(), top_idx.tolist()):
        try:
            piece = tok.decode([ids[ki]])
        except Exception:
            piece = "?"
        out.append(f"  key={ki:<4} p={v:6.3f} {'#' * int(v * 40)} {piece!r}")
    return "\n".join(out)


@mcp.tool()
async def attention_entropy_map(prompt: str = "") -> str:
    """ASCII heat-map of average attention entropy per (layer, head)."""
    _require()
    torch = _torch()
    prompt = prompt or _default_prompt()
    idx, _ = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with tdb_hooks.capture(layers="all", attn_probs=True) as cap:
            _ = _state["model"](idx)
    h = _state["handle"]
    L, H = h.n_layer, h.n_head
    glyphs = " .:-=+*#%@"
    all_vals = [v for st in cap.layer_states.values() for v in st.attn_entropy_per_head]
    if not all_vals:
        return "no attention probabilities captured"
    lo, hi = min(all_vals), max(all_vals) + 1e-9
    out = ["Attention entropy (lower = sharper)",
           "      " + "".join(f"h{j:<3}" for j in range(H))]
    for i in range(L):
        st = cap.layer_states.get(i)
        ent = st.attn_entropy_per_head if st else []
        cells = []
        for j in range(H):
            if j < len(ent):
                v = (ent[j] - lo) / (hi - lo)
                g = glyphs[min(int(v * (len(glyphs) - 1)), len(glyphs) - 1)]
                cells.append(f" {g}  ")
            else:
                cells.append(" ?  ")
        out.append(f"L{i:<3}  " + "".join(cells))
    out.append(f"\nrange: low={lo:.3f}  high={hi:.3f}")
    return "\n".join(out)


@mcp.tool()
async def top_logits(prompt: str = "", k: int = 10) -> str:
    """Top-k next-token candidates at the final position."""
    _require()
    torch = _torch()
    prompt = prompt or _default_prompt()
    idx, _ = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with tdb_hooks.capture(layers=[]) as cap:
            _ = _state["model"](idx)
    if cap.last_logits is None:
        return "no logits captured (model output shape unrecognised)"
    logits = cap.last_logits if cap.last_logits.dim() == 1 else cap.last_logits[0]
    vals, top_idx = torch.topk(logits, k=min(k, logits.numel()))
    tok = _state["tokenizer"]
    probs = torch.softmax(vals, dim=-1)
    out = [f"top-{k} continuations:"]
    for v, ki, pr in zip(vals.tolist(), top_idx.tolist(), probs.tolist()):
        try:
            piece = tok.decode([ki])
        except Exception:
            piece = "?"
        out.append(f"  id={ki:<6} logit={v:+7.3f} p≈{pr:5.3f}  {piece!r}")
    return "\n".join(out)


@mcp.tool()
async def direction_of_interest(prompt: str, target_token: str,
                                distractor_token: str) -> str:
    """Logit-diff for target vs distractor + per-layer direct effect."""
    _require()
    torch = _torch()
    tok = _state["tokenizer"]
    try:
        t_id = tok.encode(target_token)[0]
        d_id = tok.encode(distractor_token)[0]
    except Exception as e:
        return f"encode error: {e}"
    direction = _direction(t_id, d_id)
    idx, _ = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with tdb_hooks.capture(layers="all", direction=direction) as cap:
            _ = _state["model"](idx)
    last = cap.last_logits if cap.last_logits.dim() == 1 else cap.last_logits[0]
    diff = float(last[t_id] - last[d_id])
    out = [
        f"target={target_token!r}(id={t_id}) distractor={distractor_token!r}(id={d_id})",
        f"logit_diff = {diff:+.3f}   p(t)/p(d) ≈ {math.exp(diff):.3g}",
        "",
        "layer    DE(attn)    DE(mlp)",
    ]
    for i in range(_state["handle"].n_layer):
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
    """Zero-ablate attn/mlp at chosen layers; compare baseline vs ablated."""
    _require()
    torch = _torch()
    a = [int(x) for x in attn_layers.split(",") if x.strip()]
    m = [int(x) for x in mlp_layers.split(",") if x.strip()]
    idx, _ = _tokenize_to_tensor(prompt)
    tok = _state["tokenizer"]

    def _summary(logits):
        if target_token and distractor_token:
            t = tok.encode(target_token)[0]
            d = tok.encode(distractor_token)[0]
            return f"logit_diff(t-d) = {float(logits[t] - logits[d]):+.3f}"
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
        with tdb_hooks.capture(layers=[]) as cap:
            _ = _state["model"](idx)
        base = cap.last_logits if cap.last_logits.dim() == 1 else cap.last_logits[0]
        with tdb_hooks.ablate(attn_layers=a, mlp_layers=m):
            with tdb_hooks.capture(layers=[]) as cap2:
                _ = _state["model"](idx)
        ab = cap2.last_logits if cap2.last_logits.dim() == 1 else cap2.last_logits[0]
    return (
        f"ablate attn={a} mlp={m}\n"
        f"  baseline:  {_summary(base)}\n"
        f"  ablated :  {_summary(ab)}"
    )


@mcp.tool()
async def trace_upstream(prompt: str, target_token: str,
                         distractor_token: str) -> str:
    """Backward through the model; per-layer act·grad on post-block residual."""
    _require()
    torch = _torch()
    tok = _state["tokenizer"]
    try:
        t_id = tok.encode(target_token)[0]
        d_id = tok.encode(distractor_token)[0]
    except Exception as e:
        return f"encode error: {e}"
    idx, _ = _tokenize_to_tensor(prompt)
    model = _state["model"]
    h = _state["handle"]
    blocks = h.lm_head  # placeholder, real lookup below
    from tdb_hooks.attach import _get
    blk_list = _get(model, h.layout["blocks"])
    captured: dict[int, "torch.Tensor"] = {}
    handles = []
    for i, block in enumerate(blk_list):
        def _mk(idx_=i):
            def hook(_mod, _inp, output):
                t = output[0] if isinstance(output, tuple) else output
                t.retain_grad()
                captured[idx_] = t
                return output
            return hook
        handles.append(block.register_forward_hook(_mk()))
    try:
        out = model(idx)
        logits = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else out.logits)
        loss = logits[0, -1, t_id] - logits[0, -1, d_id]
        loss.backward()
    finally:
        for hh in handles:
            hh.remove()
    rows = [
        f"trace upstream: {target_token!r} - {distractor_token!r}",
        f"  end logit diff = {float(loss):+.3f}",
        "",
        "layer   ||resid||    ||grad||    act·grad (last token)",
    ]
    for i in range(h.n_layer):
        if i not in captured or captured[i].grad is None:
            continue
        r = captured[i].detach().float()
        g = captured[i].grad.float()
        ag = float((r[:, -1] * g[:, -1]).sum().item())
        rows.append(f"  L{i:<3}  {float(r.norm()):8.2f}   {float(g.norm()):8.4f}   {ag:+9.4f}")
    model.zero_grad(set_to_none=True)
    return "\n".join(rows)


# ============================================================================
# Training visualisation
# ============================================================================
@mcp.tool()
async def training_curve(report_path: str = "", project: str = "",
                         width: int = 60, height: int = 18) -> str:
    """ASCII loss curve. Auto-discovers reports/*.md in the project root if no path given."""
    paths: list[Path] = []
    if report_path:
        paths.append(Path(report_path))
    else:
        proj = project or _state["project"] or "mathgpt"
        if proj not in PROJECTS:
            return f"unknown project '{proj}'"
        for p in (PROJECTS[proj]["root"] / "reports").glob("*.md"):
            paths.append(p)
        for p in (PROJECTS[proj]["root"] / "runs").rglob("*.log"):
            paths.append(p)
    if not paths:
        return "no report files found"
    pat = re.compile(r"step\s*[:=]?\s*(\d+).*?loss\s*[:=]?\s*([0-9]+\.[0-9]+)", re.I)
    pairs = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in pat.finditer(text):
            pairs.append((int(m.group(1)), float(m.group(2))))
    if not pairs:
        return "no 'step ... loss ...' pairs found in:\n  " + "\n  ".join(map(str, paths))
    pairs.sort()
    return _ascii_plot([s for s, _ in pairs], [l for _, l in pairs],
                       width, height, f"loss vs step ({len(pairs)} pts)")


def _ascii_plot(xs, ys, width, height, title):
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
    out = [title]
    out.append(f"y={ymax:.3f} ┐ " + lines[0])
    for ln in lines[1:-1]:
        out.append("         │ " + ln)
    out.append(f"y={ymin:.3f} ┘ " + lines[-1])
    out.append("           " + f"x={xmin}".ljust(width // 2)
               + f"x={xmax}".rjust(width - width // 2))
    return "\n".join(out)


# ============================================================================
# Walkthrough demo
# ============================================================================
@mcp.tool()
async def run_tdb_walkthrough(prompt: str = "", target_token: str = "",
                              distractor_token: str = "") -> str:
    """Full 7-step TDB session on the current model. Defaults differ per project."""
    _require()
    torch = _torch()
    proj = _state["project"]
    prompt = prompt or _default_prompt()
    if not target_token or not distractor_token:
        # sensible defaults per project
        if proj == "mathgpt":
            target_token, distractor_token = " 1", " 2"
        else:
            target_token, distractor_token = " return", " pass"
    sections = []
    sections.append(f"== project={proj} ckpt={_state['checkpoint_id']} ==")
    sections.append("\n== 1. tokenisation ==")
    sections.append(await tokenize(prompt))
    sections.append("\n== 2. per-layer forward ==")
    sections.append(await run_forward(prompt, True, target_token, distractor_token))
    sections.append("\n== 3. attention entropy heat-map ==")
    sections.append(await attention_entropy_map(prompt))
    # find sharpest head
    idx, _ = _tokenize_to_tensor(prompt)
    with torch.no_grad():
        with tdb_hooks.capture(layers="all", attn_probs=True) as cap:
            _ = _state["model"](idx)
    best = (None, None, math.inf)
    for li, st in cap.layer_states.items():
        for hi, e in enumerate(st.attn_entropy_per_head):
            if e < best[2]:
                best = (li, hi, e)
    if best[0] is not None:
        sections.append(f"\n== 4. sharpest head L{best[0]} H{best[1]} (entropy={best[2]:.3f}) ==")
        sections.append(await attention_distribution(best[0], best[1], -1, prompt))
    sections.append("\n== 5. direction of interest ==")
    sections.append(await direction_of_interest(prompt, target_token, distractor_token))
    sections.append("\n== 6. upstream gradient trace ==")
    sections.append(await trace_upstream(prompt, target_token, distractor_token))
    layer_scores = sorted(
        ((abs(st.delta_attn_norm) + abs(st.delta_mlp_norm), li)
         for li, st in cap.layer_states.items()),
        reverse=True,
    )[:3]
    sections.append("\n== 7. ablation sweep on top-magnitude layers ==")
    for _, li in layer_scores:
        sections.append(await ablate_node(prompt, str(li), "", target_token, distractor_token))
        sections.append(await ablate_node(prompt, "", str(li), target_token, distractor_token))
    return "\n".join(sections)


# ============================================================================
# Filesystem helpers (let the agent browse the target project)
# ============================================================================
@mcp.tool()
async def read_file(file_path: str, max_chars: int = 80_000) -> str:
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return f"not a file: {file_path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


@mcp.tool()
async def list_directory(directory_path: str = ".") -> str:
    p = Path(directory_path)
    if not p.exists() or not p.is_dir():
        return f"not a directory: {directory_path}"
    return "\n".join(
        f"{'DIR ' if it.is_dir() else 'FILE'} {it.name:<40} "
        f"{(it.stat().st_size if it.is_file() else 0):>12,} bytes"
        for it in sorted(p.iterdir())
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
