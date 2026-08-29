# 化学棋 AlphaZero 自我对弈强化学习（纯自对弈，无手写 AI）

为 `化学棋Chemiss.html` 实现了一套 **AlphaZero 风格**的纯自我对弈强化学习，
不依赖游戏里任何手写评估/搜索函数。下面说明各文件的作用与用法。

## 文件

| 文件 | 作用 |
|---|---|
| `chemis_env.py` | 把 `ChemissGame` 的核心逻辑（棋盘状态、移动生成、键合判定、吃子、给电子、核变、衰变/射线、电性吸引、终局判定）移植成纯 Python 环境。`step(action)` 内含全部随机性（衰变触发概率、射线方向、β 电子 ±1），用标准库 `random` 实现。 |
| `model.py` | PyTorch 策略+价值双头网络（输入 `[17,8,8]`，输出策略 logits `[4710]` + 价值标量 `[-1,1]`）。3 个残差块（64 通道）。含导出扁平化 `model_weights.json` 的函数。 |
| `mcts.py` | PUCT 蒙特卡洛树搜索（`c_puct=1.0`）。叶子用网络价值评估。**随机性处理**：`expand()` 中若动作触发随机事件（衰变/射线），强制采样 3 次生成多个子节点，把随机性转成树上的概率分布。 |
| `train_selfplay.py` | 自我对弈主循环：当前网络+MCTS 下完整棋局，记录每一步 `(特征, π, z)` 进经验池，按 AlphaZero 损失更新网络，每 N 局用新网络替换自我对弈网络。 |
| `handwritten_ai.py` | 从 HTML `ChemissAI` 移植的**手写教师**（15 项评估 + alpha-beta negamax），用于 `--teacher_depth/--teacher_every` 的“ML vs 手写教师”对弈，产出高质量数据。 |
| `model_weights.json` | 训练完成后导出（若训练未跑满则仍会生成一份已训练的权重）。 |
| `化学棋Chemiss.html` | 在 `ChemissAI` 中新增 `loadMLWeights(json)`、`mlEvaluate(features)`、`_buildMLFeatures()`，并让 `evaluate()` 在 `useML===true` 时用 ML 推理替换原 15 项手写公式。 |

## 动作编码（与 HTML 无 UI 改动对齐）

- 普通移动：`(from_r*8+from_c)*64 + (to_r*8+to_c)` → `[0,4095]`
- 给电子：`4096 + (metal_sq*8 + dir_idx)` → `[4096,4607]`
- 核变：`4608 + nuclear_target_idx` → `[4608,4709]`
- 动作空间大小 `ACTION_SPACE_SIZE = 4710`

## 神经网络输入特征 `[17,8,8]`（从“执子方”视角）

通道 0~15（见 `chemis_env.build_features` / `_buildMLFeatures`）：
0/1 己/敌 Z 映射，2/3 己/敌总电子，4/5 己/敌电荷，6/7 己/敌键合，
8/9 己/敌眩晕，10/11 己/敌放射性，12/13 己/敌氢王，14 回合进度，15/16 己/敌推进度。

> 价值头输出的是**执子方**胜率 `[-1,1]`；`mlEvaluate` 返回 `this.color` 视角的
> 价值，`evaluate()` 把它线性放大到与搜索一致的分数（×50000）。

## 训练

```bash
# 完整规格（约 2000 局 / 200 模拟 / batch 256 / 每 50 局换网络）
python train_selfplay.py --num_games 2000 --num_simulations 200 \
    --batch_size 256 --games_per_update 50 --device cuda
```

想快速看效果可缩小规模：

```bash
python train_selfplay.py --num_games 8 --num_simulations 100 --batch_size 256 \
    --games_per_update 3 --device cuda
```

参数：
- `--num_games`：自我对弈总局数（默认 2000）
- `--num_simulations`：每步 MCTS 模拟次数（默认 200）
- `--batch_size`：从经验池采样批量（默认 256）
- `--games_per_update`：每 N 局用新网络替换自我对弈网络（默认 50）
- `--learning_rate` / `--c_reg` / `--buffer_size`
- `--out`：输出的权重 JSON 路径（默认 `model_weights.json`）
- `--checkpoint`：从已有 `*_model.pt` 继续训练

### 手写 AI 教师对弈（高质量数据，新增 `handwritten_ai.py`）

`handwritten_ai.py` 是把 HTML `ChemissAI` 移植到 Python 的教师：**15 项评估 + alpha-beta(negamax)**。
训练时每 `teacher_every` 局派 1 局「ML(MCTS) vs 手写教师」：
- ML 盘面记录其 MCTS 分布 π；
- 教师盘面记录 one-hot(教师所选动作)（**策略蒸馏**）；
- 都按最终胜负打 z 标签，让价值头学会“识别危险局面 / 别送王”。
使能：`--teacher_depth 2 --teacher_every 15`（depth=0 或 teacher_every=0 关闭）。

## 训练完成后的集成

`model_weights.json`（由 `train_selfplay.py` 生成）放在与 `化学棋Chemiss.html`
**同一目录**。网页版（GitHub Pages / 任意静态服务器）加载页面时会自动
`fetch('model_weights.json')` 并注入 `window.__CHEMISS_ML_JSON__`，
`ChemissAI.evaluate()` 检测到后懒加载并接管评估。

- 若服务环境不支持 fetch（如本地 `file://` 打开），可在控制台手动启用：
  ```js
  // 前提：已把权重读取为 json 对象
  game.ai.loadMLWeights(json); // 或
  const ai = new ChemissAI(game, 'white', 4); ai.loadMLWeights(json);
  ```
- 无权重时自动退回原 15 项手写评估，不影响原有玩法。

## HTML 端集成（AlphaZero 玩法）

- 加载权重后 `ChemissAI.useML === true`，`findBestMove()` **自动切换到 MCTS**：
  用神经网络产生的【策略先验 + 价值】做 PUCT（`c_puct=1.0`）搜索，模拟次数
  `ai._mctsSims`（默认 60，越大越强越慢），用到 `mlEvaluateFull()`（同时返回策略 logits 与价值）。
- **MCTS-Solver**：终端节点视为“已证明”结果，搜索会优先走“本方必胜”的子节点，
  并在整棵树上做 minimax 式证据传播（子节点全证明时反推父节点胜负），避免在已定性分支上浪费模拟。
- 游戏新增 `_cloneForSearch()` 用于克隆局面，MCTS 在克隆上展开，避免污染真实棋盘；
  克隆关闭渲染/动画/计时副作用。
- 纯 JS 手写的前向（conv/BN/ReLU/残差/fc/tanh）与 PyTorch 完全对齐（已在 Node 中对拍 ~1e-9）。

## 在 HTML 里动态演示：手写 AI vs Chemiss-Entrorpior（无后端）

设置栏有一个 **「手写AI vs Chemiss-Entrorpior 演示」** 按钮（⚡ 图标），点击即开始一盘
**手写 AI(黑) vs ML Chemiss-Entrorpior(白)** 的自动对弈，棋盘实时动画。

**无需任何后端/服务器**：权重以 `window.__CHEMISS_ML_JSON__`（base64 压缩，约 4.3MB）
的形式放在 `model_weights.js` 里，通过 `<script src="./model_weights.js">` 在 `file://`
下直接加载（不触发 CORS）。只要 `model_weights.js` 和 `化学棋Chemiss.html` 同目录：

- **直接双击打开 `化学棋Chemiss.html` 即可**；点设置栏 ⚡ 按钮开始演示。
- 控制台也可 `game.startAZDemo()`；再点一次按钮停止。
- 若无 `model_weights.js`（如网页版），会退回 `fetch('model_weights.json')` 或纯手写评估。

- 白方 = ML(MCTS)（`_mctsSims=40`，可在 `game.azDemo.sims` 调整）；黑方 = 手写 AI。
- 权重重导出：`python -c "from model import ChemisNet,save_model_weights_js as s; import torch; n=ChemisNet(); n.load_state_dict(torch.load('model_weights_model.pt')); s(n,'model_weights.js')"`

## 合法性规则修改：「送 H」非法

- `getLegalMoves()` 现在**始终**过滤掉会让己方氢王处于被将军状态的走法（不再因 AI 搜索而跳过）。
  也就是说：任何一方（含手写 AI / ML MCTS）都不能走出把自己的氢王送给对方吃的“送 H”步。
- 调试/编辑模式仍放宽（便于构造局面）。

## 随机性处理说明

`mcts.py` 在 `expand()` 中对每个动作做转移采样：用 `env._last_step_had_random`
判断该动作是否触发随机事件（衰变触发、射线方向、β 电子 ±1、离子键溢出射线、
裂变）。若随机，则强制采样 `STOCHASTIC_SAMPLES=3` 次并生成 ≤3 个兄弟子节点，
实现把随机性转化为树上的概率分布；确定性动作只生成 1 个子节点。

## 三方对战：MCTS-Solver vs AB剪枝 vs NNUE（battle.py）

把三种搜索/评估范式放同一时间盒（每步 `--budget_ms`）里做 round-robin 对弈，
找“同样算力下谁更强/更快”：

| 引擎 | 说明 |
|---|---|
| `MCTS-Solver` | 训练网络(策略+价值) + PUCT 树搜索，启用 MCTS-Solver（终端节点视为已证明、证据传播、必胜步优先），走 `mcts.py`。慢但理论上限高。 |
| `AB剪枝` | 手写 AI：15 项静态评估 + negamax alpha-beta，走 `handwritten_ai.py`。快、鲁棒。 |
| `NNUE` | 训练网络**价值头**做静态评估 + alpha-beta(negamax)，策略 logits 做走法排序；走 `nnue.py`。本质是“神经网络评估 + AB”，比 MCTS 轻。 |

```bash
python battle.py --games 3 --budget_ms 400 --max_moves 120 --seed 7
# 想更快看出差距：加大预算/上限（让对局别在大限处截断）
python battle.py --games 2 --budget_ms 700 --max_moves 200 --seed 11
```

- `--pairs "MCTS-Solver,AB剪枝;..."` 只跑指定对（默认全部两两对）。
- `--no_solver`：MCTS 关闭 Solver（对照实验）。
- 结论（预算 400ms、局长上限 120 时）多为和棋（三引擎都不差，且都规避“送 H”，
  对局常拖到大限截断）；要分出优劣须**抬高预算 + 局长上限**（见下）。

## HTML 复盘/分析：优先静态分析 + AB剪枝（避免卡顿）

复盘分析引擎 `PostGameAnalyzer` 原先创建的分析 AI 会触发 **每节点一次神经网络前向**
（`evaluate()` 走 `useML` 分支），深度搜索时非常卡。现已改为：

- 分析用 `_mkAI()` 创建的所有 ChemissAI **强制 `_disableML=true`**：`evaluate()` 回到
  手写 15 项静态评估，搜索用 **alpha-beta 剪枝**；复盘不再碰 ML/MCTS。
- 静态模式（默认，`_useStaticAnalysis=true`）深度 1 已足够快；深度搜索模式也用静态评估做叶子。
- 说明：演示按钮（⚡ 手写 AI vs Chemiss-Entrorpior）**不受影响**——它明确用 `_forceML` 实现，
  `_disableML` 不影响该演示路径。

## 单文件无后端：模型配置页 + 可调棋盘

### 模型配置页（`btnModelCfg`，⚙ 设置栏）

新增「模型配置」按钮，打开面板可为**对战双方**各选一种模型，并设 MCTS 模拟数，
点「开始自动对战」即由所选模型自动对弈：

| 模型 | 说明 |
|---|---|
| `手写AI` | 15 项静态评估 + alpha-beta 剪枝（`_aiModel='hand'`，`_disableML=true`） |
| `Chemiss-Entrorpior` | 神经网络(策略+价值) + PUCT 树搜索（`_aiModel='mcts'`，`_forceML=true`） |
| `NNUE` | 神经网络**价值头**做静态评估 + alpha-beta 剪枝（`_aiModel='nnue'`，`useML=true` 但 `findBestMove` 走 AB 而非 MCTS），比 MCTS 快 |

- 实现：`ChemissAI.applyModel(model)` 统一设置 `_aiModel/_forceML/_disableML/useML`；
  `findBestMove()` 只有在 `_aiModel==='mcts'`（或 `_forceML`）时才派发到 MCTS，否则走 alpha-beta。
- **仅 8×8 生效**：网络输入 `[17,8,8]`、动作空间 4710 都按 8×8 编码，故 `_isStdBoard()`
  为假（非 8×8 棋盘）时自动禁用 ML/NNUE，退回手写 AI。
- **人机/自动对弈也生效**：配置保存到 `game.aiModelConfig={white,black,sims}`，
  `_applyConfiguredModel(ai)` 在人机对战(`toggleAI`)、AI 自动对弈、难度/执色切换时统一应用；
  ⚡ 演示 (`azDemo`) 仍用 `whiteModel/blackModel`，优先于全局配置。
- **人机对战的说明**：人机模式下，AI 方按模型配置走子（AI 执黑用「黑方」模型、执白用「白方」）。
  侧栏会显示 **AI模型** 一行；事件日志也会写明 `AI 黑方 用 Chemiss-Entrorpior / NNUE / 手写AI`。
  模型配置页的说明区同步解释了三种模型及人机/自动对弈的差异。
- 已把「手写AI vs Chemiss-Entrorpior 演示」(`azDemo`) 升级为按颜色的 `whiteModel/blackModel`。

### 棋盘编辑：可调行列数（`btnBoardSize`）

棋盘编辑模式下新增 **行数/列数** 输入框 + 「应用尺寸」按钮（或控制台
`game.setBoardSize(rows, cols)`），把棋盘重建为任意 4~16 的正方形/矩形。

- 核心：`game.boardRows` / `game.boardCols`（默认 8）；`initBoard()` 按实际行列摆开局
  （白在底线 `R-1`、前排 `R-2`；黑在 `0`/`1`）；不足/超出 8 列自动居中或截断。
- **尺寸上限 32×32**：`setBoardSize(rows, cols)` 支持 4~32 的行列（更大会被夹到 32）。
- 所有走法/将军/成键/射线/序列化(PGN/FEN)/复盘渲染的硬编码 `8` 均改为 `rows/cols`。
- `serializeBoard()` 现在在 JSON 里记录 `rows/cols`，`deserializeBoard()` 据此重建，
  因此自定义尺寸的棋盘也能正确保存/复盘/导入导出。
- **导出 PGN 会写 `[BoardSize "RxC"]`**；导入时读回并据此重建棋盘（旧 PGN 无该标签时由 FEN 行列推断）。
- **ML 仅 8×8**：非 8×8 时手写 AI 自适应（评估中心/前进项已按实际尺寸缩放）。
- **渲染**：`renderBoard()` 用纯 CSS 计算格子尺寸
  `--cell-size = calc(var(--board-size) / var(--grid-n))`（`--grid-n = max(行,列)`），
  并让 `.board` 的宽/高紧贴网格，避免 `parseFloat('min(...)')` 取到 token 导致溢出。



