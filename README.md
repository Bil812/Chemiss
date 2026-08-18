<div align="center">

# 化学棋 Chemiss

以化学原理为核心的本地双人对战策略棋类游戏

![Release](https://img.shields.io/badge/Release-v1.6.0-b0908a)
![License](https://img.shields.io/github/license/Bil812/Chemiss)
![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Windows%20%7C%20Android-6a9a6a)
![Language](https://img.shields.io/badge/Language-HTML%20%2F%20CSS%20%2F%20JavaScript-e8d44d)
![Online](https://img.shields.io/badge/Online-GitHub%20Pages-2f81f7)
![Last Commit](https://img.shields.io/github/last-commit/Bil812/Chemiss)

</div>

<img width="1280" height="640" alt="画板 1" src="https://github.com/user-attachments/assets/1bf84de8-6ff0-42a9-92fa-d0b1dc32ff27" />

## 项目简介

Chemiss（化学棋）是一款完全基于 HTML/CSS/JavaScript 实现的本地双人对战策略棋类游戏，无需服务器，打开浏览器即可游玩。棋子的移动、吃子、成键等机制模拟了真实的化学概念：离子键、共价键、金属键、电负性、放射性衰变、核裂变、同位素嬗变等。游戏目标为吃掉对方的氢王（H）棋子。

<img width="1279" height="731" alt="屏幕截图 2026-06-07 013400" src="https://github.com/user-attachments/assets/f8c65cff-cfd3-438b-b52d-ccd5efcd9f03" />

## 特性

- 完整实现周期表 1–7 周期、103 种元素（H–Lr）及其常见同位素
- 离子键 / 共价键 / 金属键三种成键与吃子机制
- 电荷、电性吸引、金属给电子等战略元素
- Li 到达对方底线可核变为任意元素（含同位素选择）
- α/β/γ 射线与核裂变系统，射线可诱发嬗变或裂变
- 射线路径动画、棋子状态视觉反馈
- 内置元素查询工具（化学棋工具 v1.4）：按序号/符号/中文名查询，常用/全部模式、行显示开关、键合关系、放射性、嬗变、悬停信息卡
- 可视化侧边栏：选中棋子信息卡（棋子 + 方向点 + 统计网格 + 状态徽章）、键合关系棋子图标、事件日志
- 规则弹窗内置初始棋子阵容与移动方向演示
- 所有弹窗/浮层统一淡出动画；移动端给电子指示器同心且虚线端点圆头
- 暗色模式与 Debug 模式（自由移动棋子）
- 纯前端实现，可作为化学/棋类教学工具

## 快速开始

1. 克隆仓库，或直接下载 `Chemiss.html` 文件
2. 使用浏览器打开该 HTML 文件
3. 白方先行，点击己方棋子查看合法移动，再次点击目标格移动

> 在线体验：[化学棋 Chemiss](https://bil812.github.io/Chemiss/)（GitHub Pages）

## 游戏规则

完整规则见 [化学棋Chemiss规则.md](./化学棋Chemiss规则.md)，与游戏内置「规则」弹窗保持同步。

## 安装包下载

- Windows 安装包（MSI）与 Android APK：见 [Releases](https://github.com/Bil812/Chemiss/releases)

## 开发与构建

环境要求：Node.js、Electron、Capacitor CLI、JDK 21（Android 构建）。

```bash
# Windows 安装包（MSI）
npm run build:msi

# Android APK
Copy-Item 化学棋Chemiss.html www\index.html -Force
npx cap sync android
cd android
gradlew.bat assembleDebug   # 需 JDK 21
```

## 技术栈

- HTML / CSS / JavaScript：纯前端游戏引擎
- Electron：Windows 桌面版
- Capacitor：Android 打包
- electron-builder：MSI 安装包构建

## 许可证

本项目基于 [MIT](./LICENSE) 许可证开源。
