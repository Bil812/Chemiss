<div align="center">

# 化学棋 Chemiss

以化学原理为核心的本地双人对战策略棋类游戏

**简体中文** | [English](./README.en.md)

![Release](https://img.shields.io/badge/Release-v1.7.1-b0908a)
![License](https://img.shields.io/github/license/Bil812/Chemiss)
![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Windows%20%7C%20Android-6a9a6a)
![Language](https://img.shields.io/badge/Language-HTML%20%2F%20CSS%20%2F%20JavaScript-e8d44d)
![Online](https://img.shields.io/badge/Online-GitHub%20Pages-2f81f7)
![Last Commit](https://img.shields.io/github/last-commit/Bil812/Chemiss)

</div>

<img width="1280" height="640" alt="画板 1" src="https://github.com/user-attachments/assets/1bf84de8-6ff0-42a9-92fa-d0b1dc32ff27" />

## 项目简介

Chemiss（化学棋）是一款完全基于 HTML/CSS/JavaScript 实现的本地双人对战策略棋类游戏，无需服务器，浏览器打开即玩。棋子的移动、吃子、成键等机制模拟真实化学概念：离子键、共价键、金属键、电负性、放射性衰变、核裂变、同位素嬗变等。游戏目标为吃掉对方的氢王（H）棋子，或达成将死（对方的 H 被攻击且无法规避）。

<img width="1279" height="731" alt="屏幕截图 2026-06-07 013400" src="https://github.com/user-attachments/assets/f8c65cff-cfd3-438b-b52d-ccd5efcd9f03" />

## 特性

### 化学机制（v1.6.0）

- 完整实现周期表 1–7 周期、**103 种元素（H–Lr）** 及其常见同位素数据库
- 三种成键与吃子机制：
  - **离子键**：金属-非金属按例外表、电负性差、法扬斯规则判定，可直接吃子
  - **共价键**：相邻异色自动成键，每原子最多一根，按电负性差降序贪心分配
  - **金属键**：多原子簇共享电子海
- 电负性、极化力参与键型判定
- 电荷系统：电性吸引、金属给电子（阳离子/阴离子）
- Li 到达对方底线可核变为任意非 H 元素（自选同位素）
- α/β/γ 射线系统：半衰期衰变、吸收嬗变、电子扰动、眩晕、重核裂变

### AI 与模型（v1.7.0）

- **AlphaZero 风格自我对弈强化学习**：纯自我博弈训练出的策略+价值双头神经网络（无手写 AI 参与训练），权重由外部 `model_weights.js` 异步加载（首屏不被阻塞，`file://` 无后端可运行，文件与 HTML 同目录）
- **模型配置页**：为白/黑双方选择模型（手写AI / Chemiss-Entropior / NNUE），可调 MCTS 模拟数，启动自动对战
  - **Chemiss-Entropior**：神经网络(策略+价值) + PUCT 树搜索（带 MCTS-Solver 终端证明/证据传播）
  - **NNUE**：神经网络价值头做静态评估 + alpha-beta 剪枝（较快）
  - **手写AI**：15 项静态评估 + alpha-beta（快、鲁棒）
- **人机对战**：AI 方按模型配置走子，侧栏实时显示所用模型；仅标准 8×8 棋盘下 ML/NNUE 生效，非 8×8 自动退回手写AI
- **复盘分析优先静态+AB剪枝**：逐着法用快速静态评估+alpha-beta，避免深度搜索每节点一次神经网络前向造成卡顿

### 棋盘编辑与 PGN（v1.7.0）

- 棋盘编辑模式支持**调整行列数（4~32×32）**：`setBoardSize(rows, cols)` 重建棋盘并按尺寸自适应摆子（白底线 `R-1`/前排 `R-2`，黑 `0`/`1`；不足/超出 8 列自动居中/截断）
- 所有走法/将军/成键/射线/序列化/复盘渲染的硬编码 8 改为 `boardRows/boardCols`
- **导出 PGN 标注棋盘大小** `[BoardSize "RxC"]`；导入读回并据此重建，旧 PGN 由 FEN 行列自动推断

### 对弈功能

- 本地双人对战（纯前端，浏览器即玩）
- **胜利方式**：吃掉对方氢王（H），或达成**将死**（对方的 H 被攻击且无法规避，无需真的吃掉 H）
- 人机对战（AI 难度 1–6）与 AI 自动对弈
- 复盘分析：胜率曲线、着法评价与分类
- 棋谱记录与复制、**PGN 导入导出**（棋盘编辑模式下可保存/载入局面）
- **引导式新手教学**：首次进入自动展示，可在设置栏随时回放
- 计时器（Bullet / Blitz / Rapid / Classical 等预设）
- 联机对战（MQTT 房间码，无需服务器）

### 界面与工具

- 内置**元素查询工具**（化学棋工具 v1.4）：按序号/符号/中文名查询，常用/全部模式、行显示开关、键合关系、放射性、嬗变、悬停信息卡与点击跳转
- 可视化侧边栏：选中棋子信息卡（棋子 + 方向点 + 统计网格 + 状态徽章）、键合关系棋子图标、事件日志
- 规则弹窗内置初始棋子阵容与移动方向演示
- 主题设置与暗色模式、棋盘坐标显示、Debug 编辑模式
- 全弹窗统一淡出动画；移动端给电子指示器同心且虚线端点圆头
- 中文 / English 界面切换，移动端适配

## 快速开始

1. 克隆仓库，或直接下载 `Chemiss.html` 文件
2. 使用浏览器打开该 HTML 文件
3. 白方先行，点击己方棋子查看合法移动，再次点击目标格移动

> 在线体验：[化学棋 Chemiss](https://bil812.github.io/Chemiss/)（GitHub Pages）

## 游戏规则

完整规则见 [化学棋Chemiss规则.md](./化学棋Chemiss规则.md)，与游戏内置「规则」弹窗保持同步。

## 安装包下载

Windows 安装包（MSI）与 Android APK 见 [Releases](https://github.com/Bil812/Chemiss/releases)。

## 开发与构建

仓库已包含 Electron 与 Capacitor 构建所需文件（`package.json`、`main.js`、`preload.js`、`capacitor.config.json`、`build/`、`www/`、`android/`）。

环境要求：Node.js、JDK 21（仅 Android 构建需要）。

```bash
# 安装依赖
npm install

# Windows 安装包（MSI）
npm run build:msi

# Android APK
Copy-Item 化学棋Chemiss.html www\index.html -Force   # Windows
# cp 化学棋Chemiss.html www/index.html               # macOS / Linux
npx cap sync android
cd android
gradlew.bat assembleDebug   # Windows
# ./gradlew assembleDebug   # macOS / Linux
```

## 技术栈

- HTML / CSS / JavaScript：纯前端游戏引擎
- Electron：Windows 桌面版
- Capacitor：Android 打包
- electron-builder：MSI 安装包构建

## 许可证

本项目基于 [MIT](./LICENSE) 许可证开源。
