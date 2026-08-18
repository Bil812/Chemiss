<div align="center">

# Chemiss

A local two-player strategy chess game built on chemistry principles

[中文](./README.md) | **English**

![Release](https://img.shields.io/badge/Release-v1.6.0-b0908a)
![License](https://img.shields.io/github/license/Bil812/Chemiss)
![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Windows%20%7C%20Android-6a9a6a)
![Language](https://img.shields.io/badge/Language-HTML%20%2F%20CSS%20%2F%20JavaScript-e8d44d)
![Online](https://img.shields.io/badge/Online-GitHub%20Pages-2f81f7)
![Last Commit](https://img.shields.io/github/last-commit/Bil812/Chemiss)

</div>

<img width="1280" height="640" alt="Board 1" src="https://github.com/user-attachments/assets/1bf84de8-6ff0-42a9-92fa-d0b1dc32ff27" />

## Introduction

Chemiss is a local two-player strategy board game built entirely with HTML/CSS/JavaScript. It runs directly in the browser without a server. Movement, captures and bonding simulate real chemistry: ionic bonds, covalent bonds, metallic bonds, electronegativity, radioactive decay, nuclear fission and isotope transmutation. The goal is to capture the opponent's Hydrogen King (H).

<img width="1279" height="731" alt="Screenshot 2026-06-07 013400" src="https://github.com/user-attachments/assets/f8c65cff-cfd3-438b-b52d-ccd5efcd9f03" />

## Features

### Chemistry (v1.6.0)

- Full periodic table, periods 1-7, **103 elements (H-Lr)** with a common isotope database
- Three bonding and capture mechanics:
  - **Ionic bonds**: metal-nonmetal pairs judged by exception table, electronegativity difference and Fajans' rule; direct capture
  - **Covalent bonds**: adjacent opposite-color pieces auto-bond; one bond per atom, greedy assignment by descending electronegativity difference
  - **Metallic bonds**: multi-atom clusters share an electron sea
- Electronegativity and polarizing power affect bond type
- Charge system: electrostatic attraction, metal electron donation (cations/anions)
- Li reaching the opponent's back rank can transmute into any non-H element (isotope selectable)
- α/β/γ ray system: half-life decay, absorption transmutation, electron disturbance, stun, and heavy-nucleus fission

### Gameplay

- Local two-player mode (pure front-end, play in the browser)
- vs AI (difficulty 1-6) and AI auto-play
- Game review: win-probability chart, move evaluation and classification
- Move notation recording and copy
- Chess clock (Bullet / Blitz / Rapid / Classical presets)
- Online play via MQTT room codes (no server required)

### UI & Tools

- Built-in **element query tool** (Chemiss Tool v1.4): search by atomic number / symbol / Chinese name, common/full modes, row toggles, bond relations, radioactivity, transmutation, hover cards and click-to-jump
- Visual sidebar: selected-piece info card (piece + direction dots + stat grid + status badges), bond icons, event log
- Rules modal with initial lineup and movement demos
- Theme settings with dark mode, board coordinates, Debug edit mode
- Unified fade-out animations for all modals; mobile electron-donor indicator is concentric with round dash caps
- Chinese / English UI switching and mobile adaptation

## Quick Start

1. Clone the repository or download the `Chemiss.html` file
2. Open the HTML file in a browser
3. White moves first: click your piece to see legal moves, then click the destination

> Online demo: [Chemiss](https://bil812.github.io/Chemiss/) (GitHub Pages)

## Game Rules

Full rules are documented in [化学棋Chemiss规则.md](./化学棋Chemiss规则.md), kept in sync with the in-game rules modal.

## Downloads

Windows installer (MSI) and Android APK are available on the [Releases](https://github.com/Bil812/Chemiss/releases) page.

## Development & Build

The repository includes the Electron and Capacitor build files (`package.json`, `main.js`, `preload.js`, `capacitor.config.json`, `build/`, `www/`, `android/`).

Requirements: Node.js; JDK 21 is only needed for Android builds.

```bash
# Install dependencies
npm install

# Windows installer (MSI)
npm run build:msi

# Android APK
Copy-Item 化学棋Chemiss.html www\index.html -Force   # Windows
# cp 化学棋Chemiss.html www/index.html               # macOS / Linux
npx cap sync android
cd android
gradlew.bat assembleDebug   # Windows
# ./gradlew assembleDebug   # macOS / Linux
```

## Tech Stack

- HTML / CSS / JavaScript: pure front-end game engine
- Electron: Windows desktop app
- Capacitor: Android packaging
- electron-builder: MSI installer

## License

This project is open source under the [MIT](./LICENSE) license.
