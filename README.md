# tdb-hooks · llm-evaluate-debugger-mcp

> 一个**通用的、可 pip 安装**的 TDB 风格 transformer 调试 hook 包，
> 加上一个基于它的 **MCP Server**。
> 任何 nanoGPT-like 模型（MathGPT / CodeChat / nanoGPT / HuggingFace GPT-2 / LLaMA …）
> 都可以两行代码挂上：
>
> ```python
> import tdb_hooks
> handle = tdb_hooks.attach(model, preset="codechat")
> ```
>
> **目标项目源码完全不需要改。** 全部走 PyTorch 原生的
> `register_forward_hook` + 临时 monkey-patch `F.scaled_dot_product_attention`。

参考 OpenAI 的 [Transformer Debugger (TDB)](https://github.com/openai/transformer-debugger)
术语：write magnitude、direct effect、estimated total effect、direction of
interest、ablate、trace upstream …… 全部实现。

---

## 仓库结构

```
llm-evaluate-debugger-mcp/
├── tdb_hooks/                 # ← 可 pip 安装的核心包
│   ├── __init__.py
│   ├── core.py                # Capture / Ablate context manager
│   ├── attach.py              # attach(model, ...) + 自动探测
│   ├── attention.py           # SDPA / flash_attn monkey-patch（抓注意力概率）
│   └── adapters.py            # 已知项目的 layout 预设
├── pyproject.toml             # pip install -e . 即可
├── llm_debugger.py            # MCP Server（基于 tdb_hooks，支持多项目切换，
│                              # 内置 read_file/list_directory 文件系统工具）
├── examples/
│   ├── mathgpt_debug.py
│   └── codechat_debug.py
├── tdb_walkthrough.md         # 一个具体 prompt 的 7 步 TDB 调试样例
└── README.md
```

---

## 安装

```bash
git clone https://github.com/xlisp/llm-evaluate-debugger-mcp-server && cd llm-evaluate-debugger-mcp-server
pip install -e .                 # 安装 tdb_hooks
pip install "mcp[cli]"
```

> 目标项目（MathGPT / CodeChat …）**不需要任何源码改动**。
> 之前往 `nanochat/gpt.py` 加的 patch 已经全部回滚，`debug_hooks.py` 也已删除。

---

## 30 秒上手

### 调试 MathGPT

```python
import sys, torch
sys.path.insert(0, "/data/data/com.termux/files/home/pypro/MathGPT")
from math_gpt.compat import apply; apply()

import tdb_hooks
from nanochat.checkpoint_manager import load_model

model, tokenizer, meta = load_model("base", torch.device("cpu"), "eval")
handle = tdb_hooks.attach(model, preset="mathgpt")     # ← 一行挂钩

idx = torch.tensor([tokenizer.encode("3+4=")], dtype=torch.long)
with torch.no_grad(), tdb_hooks.capture(attn_probs=True) as cap:
    _ = model(idx)

for i in range(handle.n_layer):
    st = cap.layer_states[i]
    print(f"L{i}  |Δattn|={st.delta_attn_norm:.2f}  |Δmlp|={st.delta_mlp_norm:.2f}")

with torch.no_grad(), tdb_hooks.ablate(attn_layers=[5]):
    out_ablated = model(idx)

handle.detach()
```

### 调试 CodeChat

只换一个 preset：

```python
sys.path.insert(0, "/data/data/com.termux/files/home/pypro/CodeChat")
from codechat.gpt import GPT, make_config

model = GPT(make_config("d20")).eval()
handle = tdb_hooks.attach(model, preset="codechat")    # ← 一行挂钩
```

完整示例见 `examples/mathgpt_debug.py` 和 `examples/codechat_debug.py`。

---

## 内置 preset

| preset      | blocks            | attn         | mlp   | lm_head    |
| ----------- | ----------------- | ------------ | ----- | ---------- |
| `mathgpt`   | `transformer.h`   | `attn`       | `mlp` | `lm_head`  |
| `nanochat`  | `transformer.h`   | `attn`       | `mlp` | `lm_head`  |
| `codechat`  | `blocks`          | `attn`       | `mlp` | `head`     |
| `nanogpt`   | `transformer.h`   | `attn`       | `mlp` | `lm_head`  |
| `hf_gpt2`   | `transformer.h`   | `attn`       | `mlp` | `lm_head`  |
| `llama`     | `model.layers`    | `self_attn`  | `mlp` | `lm_head`  |

不传 `preset` 时会按上面的顺序自动探测。如果都不匹配，
显式传路径就行：

```python
tdb_hooks.attach(model,
    blocks="transformer.h",
    attn="attn",
    mlp="mlp",
    lm_head="lm_head",
)
```

或者注册自己的 preset：

```python
tdb_hooks.register_preset("myproj", dict(
    blocks="model.decoder.layers",
    attn="self_attn",
    mlp="ffn",
    lm_head="output_projection",
))
tdb_hooks.attach(model, preset="myproj")
```

---

## tdb_hooks API

### `attach(model, preset=None, blocks=None, attn=None, mlp=None, lm_head=None) -> AttachHandle`
在 `model` 上挂载所有 forward hook。返回 `AttachHandle`，方便后续访问
`handle.n_layer / n_head / n_embd / lm_head`，以及 `handle.detach()`。

### `capture(layers="all", attn_probs=False, direction=None) -> Capture`
context manager。`with` 块内部所有 forward 都会写入 `cap`：

* `cap.layer_states[i].resid_pre_norm / delta_attn_norm / delta_mlp_norm`
* `cap.layer_states[i].attn_entropy_per_head`（开 `attn_probs=True` 后才有）
* `cap.layer_states[i].direct_effect_attn / direct_effect_mlp`（传 `direction` 后才有）
* `cap.attn_probs[i]`：形如 `(B, H, T, T)` 的注意力概率张量
* `cap.residual_post_block[i]`：每层 block 的 post-residual snapshot
* `cap.last_logits`：最终 logits 最后一个位置

### `ablate(attn_layers=(), mlp_layers=(), attn_head_mask=None)`
context manager。在 `with` 块内：

* 指定层的 `attn` 模块输出会被零化
* 指定层的 `mlp` 模块输出会被零化
* （`attn_head_mask` 留作扩展点）

### `is_active() / register_preset() / detach()`

详见各文件 docstring。

---

## MCP Server 用法

`llm_debugger.py` 把上面所有能力包成 MCP 工具，便于 Claude Code / Cursor 直接驱动。

注册：

```jsonc
// ~/.claude/mcp_servers.json
{
  "llm-debugger": {
    "command": "python",
    "args": ["/data/data/com.termux/files/home/pypro/llm-evaluate-debugger-mcp/llm_debugger.py"]
  }
}
```

### 提供的工具

#### 模型管理
* `list_projects()` — 列出已知项目
* `load_model(project, model_tag, step, device)` — 加载真实 checkpoint，自动 attach
* `load_random(project, ...)` — 随机权重小模型，跑通工具链
* `detach_hooks()` / `status()`

#### TDB 风格分层调试
* `tokenize(prompt)`
* `run_forward(prompt, capture_attn, target_token, distractor_token)` — 每层 |Δ| / 熵 / direct effect
* `attention_entropy_map(prompt)` — 注意力熵 ASCII 热力图
* `attention_distribution(layer, head, query_pos, prompt)` — 单头注意力 top-k
* `top_logits(prompt, k)`
* `direction_of_interest(prompt, target, distractor)`
* `trace_upstream(prompt, target, distractor)` — 真反向 act·grad
* `ablate_node(prompt, attn_layers, mlp_layers, target, distractor)`
* `run_tdb_walkthrough(prompt, target, distractor)` — 一键 7 步全流程

#### 训练可视化
* `training_curve(report_path, project, width, height)` — 自动从 `<project>/reports/*.md`
  抽 `step / loss` 对，画 ASCII 损失曲线

#### 文件系统
* `read_file(path, max_chars)` / `list_directory(path)`

### 同时调试两个项目

MCP Server 在内存中只保留一个模型，但**可以随时切换**：

```
> load_model(project="mathgpt", model_tag="d20")
> run_tdb_walkthrough()                  # 跑一遍 MathGPT
> load_model(project="codechat", model_tag="d20")
> run_tdb_walkthrough()                  # 自动切换到 CodeChat 流程
```

切换时旧的 hooks 会自动 `detach`，互不影响。

---

## 7 步分层调试流程（TDB 流程）

具体示例见 `tdb_walkthrough.md`。一句话总结：

1. **tokenize** — 看清 BPE 切法
2. **run_forward** — 找写入幅度 / direct effect 异常的层
3. **attention_entropy_map** — 找 sharp head（候选 induction / name-mover 头）
4. **attention_distribution** — 看 sharp head 实际在“看”哪个 token
5. **direction_of_interest** — 量化目标 vs 干扰，定位“把答案推错”的层
6. **trace_upstream** — 用真反向算每层 estimated total effect
7. **ablate_node** — 消融可疑层，对照 logit_diff 验证

---

## 为什么不用直接改源码

之前的方案是改 `MathGPT/nanochat/gpt.py`，加 4 处 patch + 一个新 `debug_hooks.py`。
现在改成挂 PyTorch 原生 `register_forward_hook`，好处：

* **零侵入** — 训练代码、checkpoint 兼容性、CI 全部不动
* **跨项目通用** — MathGPT、CodeChat、nanoGPT、HuggingFace、LLaMA 同一套 API
* **可装可卸** — `handle.detach()` 之后模型回到完全原生状态
* **不污染 git diff** — 调试不会留下提交残留

唯一“侵入式”的部分是抓注意力概率时临时 monkey-patch
`F.scaled_dot_product_attention` 和 `flash_attn.flash_attn_func` —— 这是必要恶，
因为这两个函数主动隐藏了 attention weights，没有别的口子。
patch 只在 `with capture(attn_probs=True)` 期间生效，结束后立即恢复。

---

## 工程上的边界

* 抓注意力概率比正常 forward 慢得多（手写 SDPA），长 prompt 慎用。
* `trace_upstream` 需要保留计算图，比 `run_forward` 显存高。
* 当前未集成 sparse autoencoder（TDB 里的 latent 维度）—— 留作扩展点。
* `_DummyTokenizer` 只在 `load_random()` 冒烟测试里用，真正调试请走项目自带 tokenizer。
