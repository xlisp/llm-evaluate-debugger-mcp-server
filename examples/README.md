```
❯ (.venv_train) root@aibox-:/mnt/openclaw/llm-evaluate-debugger-mcp-server# python
  examples/codechat_debug.py
  attached: n_layer=4 n_head=4
  L0  |_attn|=1.14  |_mlp|=2.33  _=1.916  top-attn-shape=(1, 4, 16, 16)
  L1  |_attn|=1.61  |_mlp|=2.43  _=1.916  top-attn-shape=(1, 4, 16, 16)
  L2  |_attn|=1.64  |_mlp|=2.66  _=1.916  top-attn-shape=(1, 4, 16, 16)
  L3  |_attn|=1.70  |_mlp|=2.46  _=1.917  top-attn-shape=(1, 4, 16, 16)
  L0 H0 last-token attn: [0.065, 0.056, 0.067, 0.06, 0.065, 0.062, 0.058, 0.066, 0.067, 0.055, 0.063, 0.063, 0.065, 0.062, 0.063,
   0.063]
  baseline argmax: 751   ablated argmax: 178
  (.venv_train) root@aibox-:/mnt/openclaw/llm-evaluate-debugger-mcp-server# 
```

⏺ MCP Server 就是 llm_debugger.py。你只跑了 examples/codechat_debug.py（那只是直接用 tdb_hooks 库跑的 demo），还没启
  MCP。使用步骤：

  1. 注册到 Claude Code 的 MCP 配置

  编辑 ~/.claude/mcp_servers.json（远程机器上）：

  {
    "llm-debugger": {
      "command": "/mnt/openclaw/CodeChat/.venv_train/bin/python",
      "args": ["/mnt/openclaw/llm-evaluate-debugger-mcp-server/llm_debugger.py"]
    }
  }

  注意 command 要用你跑通 demo 的那个 venv 的 python，否则 import 会失败。

  或者直接用 CLI 命令注册：

  claude mcp add llm-debugger /mnt/openclaw/CodeChat/.venv_train/bin/python
  /mnt/openclaw/llm-evaluate-debugger-mcp-server/llm_debugger.py

  2. 重启 Claude Code，然后在对话里直接让 Claude 调用

  重启后输入 /mcp 能看到 llm-debugger 已连接。然后像这样对话：

  帮我加载 codechat 的随机权重模型，然后跑一遍 TDB walkthrough，prompt 用 "def add(a, b):"，target 用 "return"

  Claude 会自动按顺序调用：
  - load_random(project="codechat") — 加载并 attach
  - tokenize(prompt="def add(a, b):") — 看 BPE
  - run_forward(...) — 每层 |Δattn|/|Δmlp|/熵
  - attention_entropy_map(...) — 熵热力图
  - run_tdb_walkthrough(...) — 一键 7 步
  - status() / detach_hooks()

  3. 手动冒烟测试（不通过 Claude，直接验证 server 能启）

  cd /mnt/openclaw/llm-evaluate-debugger-mcp-server
  python llm_debugger.py

  正常情况下会挂起等待 stdio（MCP 通过 stdin/stdout 说 JSON-RPC），没有报错就 OK，Ctrl-C 退出。有报错（比如 import 不到
  codechat）就说明 sys.path / 环境变量要调。

  4. 查看结果

  工具返回值 Claude 会直接显示在对话里——ASCII 热力图、top-k logits、每层 Δ 值都以文本形式打印。想看原始 tensor 还是走
  examples/codechat_debug.py 那种直接脚本更方便。

