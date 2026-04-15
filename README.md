# llm-evaluate-debugger-mcp

一个基于 [OpenAI Transformer Debugger (TDB)](https://github.com/openai/transformer-debugger)
思路实现的 MCP Server，专门用来调试和可视化 [MathGPT](../MathGPT) 的训练过程
与推理行为。

可以让 Claude Code / Cursor 等支持 MCP 的客户端，直接通过工具调用：

- 加载 MathGPT 的 checkpoint
- 对一个具体 prompt 走一遍 **TDB 分层调试流程**
- 查看任意 `(layer, head)` 的注意力分布
- 计算 *direction of interest* 的 logit 差和每层 *direct effect*
- 对某些层做零消融（ablation）并对比输出
- 反向传播 *trace upstream*，给出每层的 *estimated total effect*
- 读训练报告，渲染 ASCII loss 曲线

## 文件总览

| 文件 | 作用 |
| ---- | ---- |
| `llm_debugger.py` | MCP Server 主程序，参考 `filesystem.py` 用 FastMCP 写成 |
| `filesystem.py`   | 原有的文件系统 MCP（保留）|
| `tdb_walkthrough.md` | 一个具体 prompt 的 7 步 TDB 调试样例 |
| `../MathGPT/nanochat/debug_hooks.py` | 给 MathGPT 加的 hook 模块（新文件）|
| `../MathGPT/nanochat/gpt.py` | 4 处最小侵入式埋点（新增 import / SDPA 分支 / Block 捕获分支 / 最终 hook）|

## 安装

```bash
pip install mcp torch
# MathGPT 训练数据/词表/checkpoint 的根目录
export NANOCHAT_BASE_DIR=/your/path/to/nanochat_base
```

直接跑：

```bash
python /data/data/com.termux/files/home/pypro/llm-evaluate-debugger-mcp/llm_debugger.py
```

注册到 Claude Code：

```jsonc
// ~/.claude/mcp_servers.json
{
  "llm-debugger": {
    "command": "python",
    "args": ["/data/data/com.termux/files/home/pypro/llm-evaluate-debugger-mcp/llm_debugger.py"]
  }
}
```

如果还没有训练好的 checkpoint，可以先调 `load_random_model()` 用随机权重的小模型
跑通整套工具链。

## 提供的 MCP 工具

### 模型管理
- `load_model(source, model_tag, step, device)` — 加载 base/sft/rl 阶段的 checkpoint
- `load_random_model(...)` — 构造一个随机初始化的小模型，用于冒烟测试
- `status()` — 查看当前已加载模型的配置

### TDB 风格分层调试
- `tokenize(prompt)` — 分词，看清 BPE 切法
- `run_forward(prompt, capture_attn, target_token, distractor_token)`
  — 一次前向，输出每层的：
  - 残差范数 `|x|`
  - 写入幅度 `|Δattn|`、`|Δmlp|`（TDB 中的 *write magnitude*）
  - 平均注意力熵 `H̄`
  - 在目标-干扰方向上的 *direct effect*
- `attention_entropy_map(prompt)` — `(layer, head)` 注意力熵 ASCII 热力图
- `attention_distribution(layer, head, query_pos, prompt)` — 单个头的注意力 top-k
- `top_logits(prompt, k)` — 最后位置的 top-k 候选
- `direction_of_interest(prompt, target, distractor)` — `lm_head[target]-lm_head[distractor]`
  方向上每层的直接效应
- `trace_upstream(prompt, target, distractor)` — 实跑反向，得到每层 `act·grad`
  即 TDB 的 *estimated total effect*
- `ablate_node(prompt, attn_layers, mlp_layers, target, distractor)` — 零消融
  指定层的 attn / mlp，比较 baseline 与 ablated 的 logit 差
- `run_tdb_walkthrough(prompt, target, distractor)` — 一键走完 7 步全流程

### 训练可视化
- `training_curve(report_path, width, height)` — 自动从 `MathGPT/reports/*.md`
  抽 `step / loss` 对，画 ASCII 损失曲线

### 文件系统（沿袭 filesystem.py 风格）
- `read_file(path, max_chars)` — 读源码 / 报告 / 日志
- `list_directory(path)` — 列目录

## 7 步分层调试流程（TDB 流程）

具体 prompt 见 `tdb_walkthrough.md`，使用一个概率题：

> 一袋有 3 个红球 4 个蓝球，不放回连抽两个都为红球的概率是多少？

`target=" 1"`（正确答案分子），`distractor=" 2"`（似是而非的错答）。

调用 `run_tdb_walkthrough(prompt, " 1", " 2")` 会一次性产出：

1. **Tokenisation** — 看清数字 / 空格的切法
2. **每层 forward 摘要** — 找写入幅度、direct effect 异常的层
3. **注意力熵热力图** — 找 sharp head（候选 induction / name-mover 头）
4. **最 sharp 头的注意力分布** — 看它实际在“看”哪个 token
5. **Direction of interest** — 量化目标 vs 干扰的 logit 差，定位“把答案推错”的层
6. **Upstream 梯度 trace** — 用真反向算每层的 estimated total effect
7. **消融对照** — 自动挑写入幅度 top-3 的层，分别 ablate attn / mlp，对比 logit 差变化

## 加在 MathGPT 上的 hack 点

为了把这些信息抓出来，在 MathGPT 里做了**最小侵入式**改动：

1. 新文件 `nanochat/debug_hooks.py`
   - `capture(...)` / `ablate(...)` 两个 contextmanager
   - `sdpa_with_probs(...)`：抓注意力概率时用的手写 SDPA fallback
   - 各种 `record_*` 记录函数
   - 全部行为由 `DEBUG['enabled']` 这个开关控制

2. `nanochat/gpt.py` 4 处 patch（全部用 `if _dh.is_active():` 守卫，
   训练默认路径完全不变）：
   - `import` debug_hooks 模块
   - `CausalSelfAttention.forward`：当需要注意力概率时，绕开 FA3 走 SDPA
   - `Block.forward`：捕获 attn_out / mlp_out / 残差，并应用消融 mask
   - `GPT.forward`：在 lm_head 之后记录最终 logits / 残差

训练时 `DEBUG['enabled']` 默认 False，hot path 不受影响；只有 MCP Server
打开 capture 上下文时才会真正记录 / 修改张量。

## 工程上的边界

- 注意力概率是用 SDPA fallback 抓的，开 `capture_attn=True` 比正常前向慢，
  长 prompt 慎用。
- `trace_upstream` 需要梯度，会把整张图保存下来，比 `run_forward` 显存高。
- `_DummyTokenizer` 只用于 `load_random_model()` 的冒烟测试，
  真正调试 MathGPT 时永远走 `RustBPETokenizer`。
- 当前未引入 sparse autoencoder（TDB 里的 latent 维度），保留作为后续扩展点。
