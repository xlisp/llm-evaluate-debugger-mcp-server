# MathGPT × TDB Walk-through

A step-by-step layered debugging session that you can replay verbatim through the
`llm-debugger` MCP server. The flow mirrors OpenAI's *Transformer Debugger* but
runs against the MathGPT model (`/pypro/MathGPT`).

## Setup

```bash
pip install mcp torch
export NANOCHAT_BASE_DIR=/path/to/your/nanochat_base   # holds checkpoints/, tokenizer/
python /data/data/com.termux/files/home/pypro/llm-evaluate-debugger-mcp/llm_debugger.py
```

Or, register it with Claude Code:

```jsonc
// ~/.claude/mcp_servers.json
{
  "llm-debugger": {
    "command": "python",
    "args": ["/data/data/com.termux/files/home/pypro/llm-evaluate-debugger-mcp/llm_debugger.py"]
  }
}
```

If you have no checkpoint yet, call `load_random_model()` first to smoke-test
the tool surface.

## The prompt

We use a small probability question. The interesting thing is that the model
should answer **1/7** (or `1` as the leading numerator), not `2`:

> Question: A bag has 3 red balls and 4 blue balls.
> If I draw two balls without replacement, what is the probability they are both red?
> Answer:

* `target_token = " 1"`  -- correct numerator
* `distractor_token = " 2"` -- a plausible-but-wrong digit

## 7-step TDB flow

Just call `run_tdb_walkthrough(prompt=…, target_token=" 1", distractor_token=" 2")`
and you get all seven steps in one transcript. Or run them one at a time:

### 1. `tokenize(prompt)`
Verifies the BPE split. Look out for the tokens that wrap the numbers — the
direct-effect numbers are sensitive to how `1`, ` 1`, `1.`, `1/7` get split.

### 2. `run_forward(prompt, capture_attn=True, target_token=" 1", distractor_token=" 2")`
Single forward pass with the debug hooks enabled. The output table shows, per
layer:

| col          | meaning                                                        |
| ------------ | -------------------------------------------------------------- |
| `\|x\|`      | residual-stream norm before the block                          |
| `\|Δattn\|`  | **write magnitude** of attention (TDB *WRITE_NORM*)            |
| `\|Δmlp\|`   | write magnitude of MLP                                          |
| `H̄(attn)`   | mean attention entropy across heads (low = sharp)              |
| `DE_attn`    | **direct effect** of attention on the target-vs-distractor dir |
| `DE_mlp`     | direct effect of MLP                                            |

Use this as the orientation map: the layers with the largest `\|Δ\|` and the
largest `\|DE\|` are your candidates for deeper inspection.

### 3. `attention_entropy_map(prompt)`
ASCII heat-map of attention entropy across `(layer, head)`. Sharp heads
(low entropy) are the usual suspects for *induction* / *name-mover* style
behaviour — the kind of head TDB hunts for.

### 4. `attention_distribution(layer, head, query_pos=-1, prompt=...)`
Pick the sharpest cell from step 3 and inspect what it actually attends to.
You'll see something like:

```
layer=5 head=2 query_pos=-1 (token=' Answer')
  key=4    p=0.512 ###################  ' red'
  key=11   p=0.221 ########             ' balls'
  ...
```

If the top-attended token is semantically related to the answer, this head is
likely doing useful retrieval and is a candidate for an upstream-trace.

### 5. `direction_of_interest(prompt, " 1", " 2")`
Reports the final logit difference and the per-layer direct effect on the
direction `lm_head[" 1"] - lm_head[" 2"]`. Layers with strongly negative
`DE_attn` here are *actively pushing the wrong answer* — they are the
interesting layers to ablate next.

### 6. `trace_upstream(prompt, " 1", " 2")`
A real backward pass: gradients of the logit difference are propagated through
each block. Per layer we report `act·grad` on the post-block residual at the
final token — TDB's *estimated total effect*. Bigger absolute values mean a
layer's residual write matters more for the answer.

### 7. `ablate_node(prompt, attn_layers="…", mlp_layers="…", target_token=" 1", distractor_token=" 2")`
Zero-ablate the suspect components from steps 5 / 6 and verify the logit
difference moves the right way. The walk-through script automatically picks the
top-3 layers by total write magnitude and ablates each one's attention and MLP
in turn, so you can read the diff directly.

## Beyond one prompt -- training-time use

`training_curve()` reads `MathGPT/reports/*.md` for `step / loss` pairs and
draws an ASCII curve. Combine that with `run_tdb_walkthrough` at successive
checkpoints (`load_model(source='base', step=1000)`, `step=5000`, ...) to see
*when* during training a particular sharp head appears, or when a particular
direct-effect flips sign. That is the layered-debugging-over-time view the user
asked for.

## Hack points already added to MathGPT

The minimum invasive surgery is in two places:

* `nanochat/debug_hooks.py` — new file. Holds the `Capture` / `Ablate` context
  managers, the SDPA fallback that returns attention probabilities, and the
  recording helpers. All hot-path code is gated by `DEBUG['enabled']`.
* `nanochat/gpt.py` — three small edits:
  1. `import` the hooks module.
  2. In `CausalSelfAttention.forward`: when capture wants attention probs,
     route through `_dh.sdpa_with_probs` instead of FA3.
  3. In `Block.forward`: when active, call `record_block_io` and apply the
     ablation masks.
  4. In `GPT.forward`: when active, call `record_final` after the lm_head.

Training is **not** affected when `debug_hooks.DEBUG['enabled']` is False
(the default), so you can keep these patches in place during normal pretraining
and turn them on only from the MCP server.
