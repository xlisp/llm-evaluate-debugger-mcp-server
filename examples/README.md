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

```

------ v1 example --------

```
➜  llm-evaluate-debugger-mcp-server git:(main) python examples/translate_en2zh_debug.py
vocab_size = 99  (specials + ascii + CJK chars from 30 pairs)
model: n_layer=4 n_head=4 n_embd=128  params=810,112

--- training 400 steps on 30 pairs ---
  step    1  loss=4.9812
  step   50  loss=0.8085

  step  100  loss=0.0021
  step  150  loss=0.0007
  step  200  loss=0.0005
  step  250  loss=0.0004
  step  300  loss=0.0003
  step  350  loss=0.0003
  step  400  loss=0.0002

--- sample translations ---
  'hello'              → '吗好'
  'thank you'          → ''
  'i love you'         → '谢'
  'goodbye'            → ''
  'hello'              → '吗好'

tdb_hooks attached: n_layer=4 n_head=4 n_embd=128 vocab=99

======================================================================
TDB WALKTHROUGH — translate: 'hello'
  target     = '你'  (expected first Chinese char)
  distractor = '早'
======================================================================

[1] tokenize
     0  id=0    tok='<en>'
     1  id=12   tok='h'
     2  id=9    tok='e'
     3  id=15   tok='l'
     4  id=15   tok='l'
     5  id=18   tok='o'
     6  id=1    tok='<zh>'

[2] run_forward  (|Δattn|, |Δmlp|, entropy per layer)
    L0  |resid|=  30.62  |Δattn|=  11.55  |Δmlp|=  18.65  H̄=0.983
    L1  |resid|=  41.94  |Δattn|=  12.66  |Δmlp|=  23.97  H̄=1.005
    L2  |resid|=  58.53  |Δattn|=  15.85  |Δmlp|=  21.61  H̄=0.859
    L3  |resid|=  73.43  |Δattn|=  25.83  |Δmlp|=  23.06  H̄=0.668

[3] attention_entropy_map  (low=sharp #,  high=diffuse .)
      H0 H1 H2 H3
L0    .  .  .  =
L1    .     :  -
L2    %  =  .  :
L3    #  #  @  +

[4] attention_distribution  (sharpest head L3 H2, entropy=0.573)
    last-token attends to:
      pos 3   tok='l'       p=0.652
      pos 1   tok='h'       p=0.195
      pos 2   tok='e'       p=0.090
      pos 6   tok='<zh>'    p=0.057
      pos 5   tok='o'       p=0.005

[5] direction_of_interest  (direct effect of each layer onto target−distractor)
    L0  attn→- 0.369   mlp→- 0.477
    L1  attn→- 0.179   mlp→- 1.570
    L2  attn→- 0.434   mlp→- 0.534
    L3  attn→+ 0.337   mlp→+ 2.377

[6] trace_upstream  (estimated total effect via real backward)
    L0  estimated_total_effect=-0.7194
    L1  estimated_total_effect=-1.3387
    L2  estimated_total_effect=-2.7707
    L3  estimated_total_effect=-0.0000

[7] ablate_node  (MLP L3 — top positive direct-effect)
    baseline logit_diff(target−distractor) = -0.2599
    ablated  logit_diff(target−distractor) = -2.2689
    Δ = -2.0091  (confirmed — ablating this layer hurts the target)

tdb_hooks detached — model is back to native state.
➜  llm-evaluate-debugger-mcp-server git:(main)
➜  llm-evaluate-debugger-mcp-server git:(main) pwd
/home/xlisp/PyPro/llm-evaluate-debugger-mcp-server
➜  llm-evaluate-debugger-mcp-server git:(main)
```
