# -*- coding: utf-8 -*-
"""
chemis_env.py
=============
纯 Python 移植的「化学棋 Chemiss」游戏环境。

仅移植核心**游戏逻辑**（棋盘状态、移动生成、键合判定、吃子、给电子、
核变、衰变、射线、电性吸引、终局判定），不包含任何 UI / 渲染代码。

随机性（衰变触发概率、射线方向、β 电子 ±1）统一用标准库 `random` 实现，
因此 `step(action)` 是随机的，适合作决策与环境交互（MCTS / 自我对弈）。

动作编码
--------
- 普通移动:  (from_r*8 + from_c)*64 + (to_r*8 + to_c)      -> [0, 4095]
- 给电子:    4096 + (metal_sq*8 + dir_idx)                 -> [4096, 4607]
- 核变:      4608 + nuclear_target_idx                     -> [4608, 4709]
动作空间大小: ACTION_SPACE_SIZE = 4710
"""

import random
import math
import copy

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BOARD_SIZE = 8
COLOR_WHITE = 'white'
COLOR_BLACK = 'black'
METAL = 'metal'
NONMETAL = 'nonmetal'

# 顺时针 8 方向: [行增量, 列增量]
ALL_DIRS = [
    [-1, 0], [-1, 1], [0, 1], [1, 1],
    [1, 0], [1, -1], [0, -1], [-1, -1],
]
# 偶/奇周期优先方向（决定棋子“能走哪些直线方向”，数量由电子数决定）
PRIORITY_EVEN = [
    [-1, 0], [0, 1], [0, -1], [1, 0],
    [-1, 1], [-1, -1], [1, 1], [1, -1],
]
PRIORITY_ODD = [
    [-1, 1], [-1, -1], [1, 1], [1, -1],
    [-1, 0], [0, 1], [0, -1], [1, 0],
]

# ---------------------------------------------------------------------------
# 元素定义（与 HTML PIECE_DEFS 完全一致）
# key: {symbol,Z,valence,en,mass,period,group,noble,isHK,radio,halfLife,decay,lanthanide}
# ---------------------------------------------------------------------------
PIECE_DEFS = {
    'H':  {'symbol': 'H',  'Z': 1, 'valence': 1, 'en': 2.20, 'mass': 1,   'period': 1, 'group': NONMETAL, 'isHK': True},
    'He': {'symbol': 'He', 'Z': 2, 'valence': 8, 'en': 0.0,  'mass': 4,   'period': 1, 'group': NONMETAL, 'noble': True},
    'Li': {'symbol': 'Li', 'Z': 3, 'valence': 1, 'en': 0.98, 'mass': 7,   'period': 2, 'group': METAL},
    'Be': {'symbol': 'Be', 'Z': 4, 'valence': 2, 'en': 1.57, 'mass': 9,   'period': 2, 'group': METAL},
    'B':  {'symbol': 'B',  'Z': 5, 'valence': 3, 'en': 2.04, 'mass': 11,  'period': 2, 'group': NONMETAL},
    'C':  {'symbol': 'C',  'Z': 6, 'valence': 4, 'en': 2.55, 'mass': 12,  'period': 2, 'group': NONMETAL},
    'N':  {'symbol': 'N',  'Z': 7, 'valence': 5, 'en': 3.04, 'mass': 14,  'period': 2, 'group': NONMETAL},
    'O':  {'symbol': 'O',  'Z': 8, 'valence': 6, 'en': 3.44, 'mass': 16,  'period': 2, 'group': NONMETAL},
    'F':  {'symbol': 'F',  'Z': 9, 'valence': 7, 'en': 3.98, 'mass': 19,  'period': 2, 'group': NONMETAL},
    'Ne': {'symbol': 'Ne', 'Z': 10, 'valence': 8, 'en': 0.0, 'mass': 20,  'period': 2, 'group': NONMETAL, 'noble': True},
    'Na': {'symbol': 'Na', 'Z': 11, 'valence': 1, 'en': 0.93, 'mass': 23, 'period': 3, 'group': METAL},
    'Mg': {'symbol': 'Mg', 'Z': 12, 'valence': 2, 'en': 1.31, 'mass': 24, 'period': 3, 'group': METAL},
    'Al': {'symbol': 'Al', 'Z': 13, 'valence': 3, 'en': 1.61, 'mass': 27, 'period': 3, 'group': METAL},
    'Si': {'symbol': 'Si', 'Z': 14, 'valence': 4, 'en': 1.90, 'mass': 28, 'period': 3, 'group': NONMETAL},
    'P':  {'symbol': 'P',  'Z': 15, 'valence': 5, 'en': 2.19, 'mass': 31, 'period': 3, 'group': NONMETAL},
    'S':  {'symbol': 'S',  'Z': 16, 'valence': 6, 'en': 2.58, 'mass': 32, 'period': 3, 'group': NONMETAL},
    'Cl': {'symbol': 'Cl', 'Z': 17, 'valence': 7, 'en': 3.16, 'mass': 35, 'period': 3, 'group': NONMETAL},
    'Ar': {'symbol': 'Ar', 'Z': 18, 'valence': 8, 'en': 0.0, 'mass': 40,  'period': 3, 'group': NONMETAL, 'noble': True},
    'K':  {'symbol': 'K',  'Z': 19, 'valence': 1, 'en': 0.82, 'mass': 39, 'period': 4, 'group': METAL, 'radio': True, 'halfLife': 8, 'decay': 'beta'},
    'Ca': {'symbol': 'Ca', 'Z': 20, 'valence': 2, 'en': 1.00, 'mass': 40, 'period': 4, 'group': METAL},
    'Sc': {'symbol': 'Sc', 'Z': 21, 'valence': 3, 'en': 1.36, 'mass': 45, 'period': 4, 'group': METAL},
    'Ti': {'symbol': 'Ti', 'Z': 22, 'valence': 4, 'en': 1.54, 'mass': 48, 'period': 4, 'group': METAL},
    'V':  {'symbol': 'V',  'Z': 23, 'valence': 5, 'en': 1.63, 'mass': 51, 'period': 4, 'group': METAL},
    'Cr': {'symbol': 'Cr', 'Z': 24, 'valence': 6, 'en': 1.66, 'mass': 52, 'period': 4, 'group': METAL},
    'Mn': {'symbol': 'Mn', 'Z': 25, 'valence': 7, 'en': 1.55, 'mass': 55, 'period': 4, 'group': METAL},
    'Fe': {'symbol': 'Fe', 'Z': 26, 'valence': 2, 'en': 1.83, 'mass': 56, 'period': 4, 'group': METAL},
    'Co': {'symbol': 'Co', 'Z': 27, 'valence': 3, 'en': 1.88, 'mass': 59, 'period': 4, 'group': METAL},
    'Ni': {'symbol': 'Ni', 'Z': 28, 'valence': 2, 'en': 1.91, 'mass': 58, 'period': 4, 'group': METAL},
    'Cu': {'symbol': 'Cu', 'Z': 29, 'valence': 1, 'en': 1.90, 'mass': 63, 'period': 4, 'group': METAL},
    'Zn': {'symbol': 'Zn', 'Z': 30, 'valence': 2, 'en': 1.65, 'mass': 64, 'period': 4, 'group': METAL},
    'Ga': {'symbol': 'Ga', 'Z': 31, 'valence': 3, 'en': 1.81, 'mass': 69, 'period': 4, 'group': METAL},
    'Ge': {'symbol': 'Ge', 'Z': 32, 'valence': 4, 'en': 2.01, 'mass': 73, 'period': 4, 'group': NONMETAL},
    'As': {'symbol': 'As', 'Z': 33, 'valence': 5, 'en': 2.18, 'mass': 75, 'period': 4, 'group': NONMETAL},
    'Se': {'symbol': 'Se', 'Z': 34, 'valence': 6, 'en': 2.55, 'mass': 79, 'period': 4, 'group': NONMETAL},
    'Br': {'symbol': 'Br', 'Z': 35, 'valence': 7, 'en': 2.96, 'mass': 80, 'period': 4, 'group': NONMETAL},
    'Kr': {'symbol': 'Kr', 'Z': 36, 'valence': 8, 'en': 3.00, 'mass': 84, 'period': 4, 'group': NONMETAL, 'noble': True},
    'Rb': {'symbol': 'Rb', 'Z': 37, 'valence': 1, 'en': 0.82, 'mass': 85, 'period': 5, 'group': METAL, 'radio': True, 'halfLife': 8, 'decay': 'beta'},
    'Sr': {'symbol': 'Sr', 'Z': 38, 'valence': 2, 'en': 0.95, 'mass': 88, 'period': 5, 'group': METAL},
    'Y':  {'symbol': 'Y',  'Z': 39, 'valence': 3, 'en': 1.22, 'mass': 89, 'period': 5, 'group': METAL},
    'Zr': {'symbol': 'Zr', 'Z': 40, 'valence': 4, 'en': 1.33, 'mass': 91, 'period': 5, 'group': METAL},
    'Nb': {'symbol': 'Nb', 'Z': 41, 'valence': 5, 'en': 1.6,  'mass': 93, 'period': 5, 'group': METAL},
    'Mo': {'symbol': 'Mo', 'Z': 42, 'valence': 6, 'en': 2.16, 'mass': 96, 'period': 5, 'group': METAL},
    'Tc': {'symbol': 'Tc', 'Z': 43, 'valence': 6, 'en': 1.9,  'mass': 98, 'period': 5, 'group': METAL, 'radio': True, 'halfLife': 7, 'decay': 'beta'},
    'Ru': {'symbol': 'Ru', 'Z': 44, 'valence': 3, 'en': 2.2,  'mass': 101, 'period': 5, 'group': METAL},
    'Rh': {'symbol': 'Rh', 'Z': 45, 'valence': 3, 'en': 2.28, 'mass': 103, 'period': 5, 'group': METAL},
    'Pd': {'symbol': 'Pd', 'Z': 46, 'valence': 2, 'en': 2.20, 'mass': 106, 'period': 5, 'group': METAL},
    'Ag': {'symbol': 'Ag', 'Z': 47, 'valence': 1, 'en': 1.93, 'mass': 107, 'period': 5, 'group': METAL},
    'Cd': {'symbol': 'Cd', 'Z': 48, 'valence': 2, 'en': 1.69, 'mass': 112, 'period': 5, 'group': METAL},
    'In': {'symbol': 'In', 'Z': 49, 'valence': 3, 'en': 1.78, 'mass': 115, 'period': 5, 'group': METAL},
    'Sn': {'symbol': 'Sn', 'Z': 50, 'valence': 4, 'en': 1.96, 'mass': 119, 'period': 5, 'group': METAL},
    'Sb': {'symbol': 'Sb', 'Z': 51, 'valence': 5, 'en': 2.05, 'mass': 122, 'period': 5, 'group': NONMETAL},
    'Te': {'symbol': 'Te', 'Z': 52, 'valence': 6, 'en': 2.1,  'mass': 128, 'period': 5, 'group': NONMETAL},
    'I':  {'symbol': 'I',  'Z': 53, 'valence': 7, 'en': 2.66, 'mass': 127, 'period': 5, 'group': NONMETAL},
    'Xe': {'symbol': 'Xe', 'Z': 54, 'valence': 8, 'en': 2.6,  'mass': 131, 'period': 5, 'group': NONMETAL, 'noble': True},
    'Cs': {'symbol': 'Cs', 'Z': 55, 'valence': 1, 'en': 0.79, 'mass': 133, 'period': 6, 'group': METAL},
    'Ba': {'symbol': 'Ba', 'Z': 56, 'valence': 2, 'en': 0.89, 'mass': 137, 'period': 6, 'group': METAL},
    'La': {'symbol': 'La', 'Z': 57, 'valence': 3, 'en': 1.10, 'mass': 139, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Ce': {'symbol': 'Ce', 'Z': 58, 'valence': 3, 'en': 1.12, 'mass': 140, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Pr': {'symbol': 'Pr', 'Z': 59, 'valence': 3, 'en': 1.13, 'mass': 141, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Nd': {'symbol': 'Nd', 'Z': 60, 'valence': 3, 'en': 1.14, 'mass': 144, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Pm': {'symbol': 'Pm', 'Z': 61, 'valence': 3, 'en': 1.13, 'mass': 145, 'period': 6, 'group': METAL, 'lanthanide': True, 'radio': True, 'halfLife': 5, 'decay': 'beta'},
    'Sm': {'symbol': 'Sm', 'Z': 62, 'valence': 3, 'en': 1.17, 'mass': 150, 'period': 6, 'group': METAL, 'lanthanide': True, 'radio': True, 'halfLife': 10, 'decay': 'alpha'},
    'Eu': {'symbol': 'Eu', 'Z': 63, 'valence': 2, 'en': 1.2,  'mass': 153, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Gd': {'symbol': 'Gd', 'Z': 64, 'valence': 3, 'en': 1.20, 'mass': 157, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Tb': {'symbol': 'Tb', 'Z': 65, 'valence': 3, 'en': 1.2,  'mass': 159, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Dy': {'symbol': 'Dy', 'Z': 66, 'valence': 3, 'en': 1.22, 'mass': 163, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Ho': {'symbol': 'Ho', 'Z': 67, 'valence': 3, 'en': 1.23, 'mass': 165, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Er': {'symbol': 'Er', 'Z': 68, 'valence': 3, 'en': 1.24, 'mass': 167, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Tm': {'symbol': 'Tm', 'Z': 69, 'valence': 3, 'en': 1.25, 'mass': 169, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Yb': {'symbol': 'Yb', 'Z': 70, 'valence': 2, 'en': 1.1,  'mass': 173, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Lu': {'symbol': 'Lu', 'Z': 71, 'valence': 3, 'en': 1.27, 'mass': 175, 'period': 6, 'group': METAL, 'lanthanide': True},
    'Hf': {'symbol': 'Hf', 'Z': 72, 'valence': 4, 'en': 1.3,  'mass': 178, 'period': 6, 'group': METAL},
    'Ta': {'symbol': 'Ta', 'Z': 73, 'valence': 5, 'en': 1.5,  'mass': 181, 'period': 6, 'group': METAL},
    'W':  {'symbol': 'W',  'Z': 74, 'valence': 6, 'en': 2.36, 'mass': 184, 'period': 6, 'group': METAL},
    'Re': {'symbol': 'Re', 'Z': 75, 'valence': 7, 'en': 1.9,  'mass': 187, 'period': 6, 'group': METAL, 'radio': True, 'halfLife': 10, 'decay': 'beta'},
    'Os': {'symbol': 'Os', 'Z': 76, 'valence': 4, 'en': 2.2,  'mass': 190, 'period': 6, 'group': METAL},
    'Ir': {'symbol': 'Ir', 'Z': 77, 'valence': 3, 'en': 2.20, 'mass': 192, 'period': 6, 'group': METAL},
    'Pt': {'symbol': 'Pt', 'Z': 78, 'valence': 2, 'en': 2.28, 'mass': 195, 'period': 6, 'group': METAL},
    'Au': {'symbol': 'Au', 'Z': 79, 'valence': 1, 'en': 2.54, 'mass': 197, 'period': 6, 'group': METAL},
    'Hg': {'symbol': 'Hg', 'Z': 80, 'valence': 2, 'en': 2.00, 'mass': 201, 'period': 6, 'group': METAL},
    'Tl': {'symbol': 'Tl', 'Z': 81, 'valence': 3, 'en': 1.62, 'mass': 204, 'period': 6, 'group': METAL},
    'Pb': {'symbol': 'Pb', 'Z': 82, 'valence': 4, 'en': 2.33, 'mass': 207, 'period': 6, 'group': METAL},
    'Bi': {'symbol': 'Bi', 'Z': 83, 'valence': 5, 'en': 2.02, 'mass': 209, 'period': 6, 'group': METAL},
    'Po': {'symbol': 'Po', 'Z': 84, 'valence': 6, 'en': 2.0,  'mass': 210, 'period': 6, 'group': METAL, 'radio': True, 'halfLife': 3, 'decay': 'alpha'},
    'At': {'symbol': 'At', 'Z': 85, 'valence': 7, 'en': 2.2,  'mass': 210, 'period': 6, 'group': NONMETAL, 'radio': True, 'halfLife': 3, 'decay': 'alpha'},
    'Rn': {'symbol': 'Rn', 'Z': 86, 'valence': 8, 'en': 2.2,  'mass': 222, 'period': 6, 'group': NONMETAL, 'noble': True, 'radio': True, 'halfLife': 2, 'decay': 'alpha'},
    'Fr': {'symbol': 'Fr', 'Z': 87, 'valence': 1, 'en': 0.7,  'mass': 223, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 2, 'decay': 'beta'},
    'Ra': {'symbol': 'Ra', 'Z': 88, 'valence': 2, 'en': 0.9,  'mass': 226, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 3, 'decay': 'alpha'},
    'Ac': {'symbol': 'Ac', 'Z': 89, 'valence': 3, 'en': 1.1,  'mass': 227, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 3, 'decay': 'beta'},
    'Th': {'symbol': 'Th', 'Z': 90, 'valence': 4, 'en': 1.3,  'mass': 232, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 4, 'decay': 'alpha'},
    'Pa': {'symbol': 'Pa', 'Z': 91, 'valence': 5, 'en': 1.5,  'mass': 231, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 3, 'decay': 'alpha'},
    'U':  {'symbol': 'U',  'Z': 92, 'valence': 6, 'en': 1.38, 'mass': 238, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 4, 'decay': 'alpha'},
    'Np': {'symbol': 'Np', 'Z': 93, 'valence': 5, 'en': 1.36, 'mass': 237, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 6, 'decay': 'alpha'},
    'Pu': {'symbol': 'Pu', 'Z': 94, 'valence': 4, 'en': 1.28, 'mass': 244, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 8, 'decay': 'alpha'},
    'Am': {'symbol': 'Am', 'Z': 95, 'valence': 3, 'en': 1.13, 'mass': 243, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 6, 'decay': 'alpha'},
    'Cm': {'symbol': 'Cm', 'Z': 96, 'valence': 3, 'en': 1.28, 'mass': 247, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 8, 'decay': 'alpha'},
    'Bk': {'symbol': 'Bk', 'Z': 97, 'valence': 3, 'en': 1.3,  'mass': 247, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 5, 'decay': 'alpha'},
    'Cf': {'symbol': 'Cf', 'Z': 98, 'valence': 3, 'en': 1.3,  'mass': 251, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 4, 'decay': 'alpha'},
    'Es': {'symbol': 'Es', 'Z': 99, 'valence': 3, 'en': 1.3,  'mass': 252, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 3, 'decay': 'alpha'},
    'Fm': {'symbol': 'Fm', 'Z': 100, 'valence': 3, 'en': 1.3, 'mass': 257, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 3, 'decay': 'alpha'},
    'Md': {'symbol': 'Md', 'Z': 101, 'valence': 3, 'en': 1.3, 'mass': 258, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 2, 'decay': 'alpha'},
    'No': {'symbol': 'No', 'Z': 102, 'valence': 2, 'en': 1.3, 'mass': 259, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 2, 'decay': 'alpha'},
    'Lr': {'symbol': 'Lr', 'Z': 103, 'valence': 3, 'en': 1.3, 'mass': 266, 'period': 7, 'group': METAL, 'radio': True, 'halfLife': 3, 'decay': 'alpha'},
}

Z_TO_KEY = {d['Z']: k for k, d in PIECE_DEFS.items()}

# 同位素数据库 key -> [ {mass, stable, radio, halfLife, decay, name} ]
ISOTOPES = {
    'H':  [{'mass': 1, 'stable': True}, {'mass': 2, 'stable': True, 'name': 'D'}, {'mass': 3, 'radio': True, 'halfLife': 3, 'decay': 'beta', 'name': 'T'}],
    'He': [{'mass': 4, 'stable': True}, {'mass': 3, 'stable': True}],
    'Li': [{'mass': 7, 'stable': True}, {'mass': 6, 'stable': True}],
    'Be': [{'mass': 9, 'stable': True}, {'mass': 10, 'radio': True, 'halfLife': 5, 'decay': 'beta'}],
    'B':  [{'mass': 11, 'stable': True}, {'mass': 10, 'stable': True}],
    'C':  [{'mass': 12, 'stable': True}, {'mass': 13, 'stable': True}, {'mass': 14, 'radio': True, 'halfLife': 6, 'decay': 'beta'}],
    'N':  [{'mass': 14, 'stable': True}, {'mass': 15, 'stable': True}],
    'O':  [{'mass': 16, 'stable': True}, {'mass': 17, 'stable': True}, {'mass': 18, 'stable': True}],
    'F':  [{'mass': 19, 'stable': True}, {'mass': 18, 'radio': True, 'halfLife': 2, 'decay': 'beta'}],
    'Ne': [{'mass': 20, 'stable': True}, {'mass': 21, 'stable': True}, {'mass': 22, 'stable': True}],
    'Na': [{'mass': 23, 'stable': True}, {'mass': 22, 'radio': True, 'halfLife': 2, 'decay': 'beta'}, {'mass': 24, 'radio': True, 'halfLife': 3, 'decay': 'beta'}],
    'Mg': [{'mass': 24, 'stable': True}, {'mass': 25, 'stable': True}, {'mass': 26, 'stable': True}],
    'Al': [{'mass': 27, 'stable': True}],
    'Si': [{'mass': 28, 'stable': True}, {'mass': 29, 'stable': True}, {'mass': 30, 'stable': True}],
    'P':  [{'mass': 31, 'stable': True}, {'mass': 32, 'radio': True, 'halfLife': 3, 'decay': 'beta'}, {'mass': 33, 'radio': True, 'halfLife': 4, 'decay': 'beta'}],
    'S':  [{'mass': 32, 'stable': True}, {'mass': 33, 'stable': True}, {'mass': 34, 'stable': True}, {'mass': 35, 'radio': True, 'halfLife': 4, 'decay': 'beta'}],
    'Cl': [{'mass': 35, 'stable': True}, {'mass': 37, 'stable': True}, {'mass': 36, 'radio': True, 'halfLife': 5, 'decay': 'beta'}],
    'Ar': [{'mass': 40, 'stable': True}, {'mass': 36, 'stable': True}, {'mass': 38, 'stable': True}],
    'K':  [{'mass': 39, 'stable': True}, {'mass': 40, 'radio': True, 'halfLife': 8, 'decay': 'beta'}, {'mass': 41, 'stable': True}],
    'Ca': [{'mass': 40, 'stable': True}, {'mass': 42, 'stable': True}, {'mass': 44, 'stable': True}, {'mass': 48, 'radio': True, 'halfLife': 6, 'decay': 'beta'}],
    'Sc': [{'mass': 45, 'stable': True}],
    'Ti': [{'mass': 48, 'stable': True}, {'mass': 46, 'stable': True}, {'mass': 47, 'stable': True}, {'mass': 49, 'stable': True}, {'mass': 50, 'stable': True}],
    'V':  [{'mass': 51, 'stable': True}, {'mass': 50, 'radio': True, 'halfLife': 5, 'decay': 'beta'}],
    'Cr': [{'mass': 52, 'stable': True}, {'mass': 53, 'stable': True}, {'mass': 54, 'stable': True}],
    'Mn': [{'mass': 55, 'stable': True}],
    'Fe': [{'mass': 56, 'stable': True}, {'mass': 54, 'stable': True}, {'mass': 57, 'stable': True}, {'mass': 58, 'stable': True}],
    'Co': [{'mass': 59, 'stable': True}, {'mass': 60, 'radio': True, 'halfLife': 5, 'decay': 'beta'}],
    'Ni': [{'mass': 58, 'stable': True}, {'mass': 60, 'stable': True}, {'mass': 62, 'stable': True}, {'mass': 64, 'stable': True}],
    'Cu': [{'mass': 63, 'stable': True}, {'mass': 65, 'stable': True}],
    'Zn': [{'mass': 64, 'stable': True}, {'mass': 66, 'stable': True}, {'mass': 68, 'stable': True}],
    'Ga': [{'mass': 69, 'stable': True}, {'mass': 71, 'stable': True}],
    'Ge': [{'mass': 74, 'stable': True}, {'mass': 72, 'stable': True}, {'mass': 73, 'stable': True}, {'mass': 76, 'stable': True}],
    'As': [{'mass': 75, 'stable': True}],
    'Se': [{'mass': 80, 'stable': True}, {'mass': 78, 'stable': True}, {'mass': 76, 'stable': True}, {'mass': 82, 'stable': True}],
    'Br': [{'mass': 79, 'stable': True}, {'mass': 81, 'stable': True}],
    'Kr': [{'mass': 84, 'stable': True}, {'mass': 86, 'stable': True}, {'mass': 82, 'stable': True}, {'mass': 83, 'stable': True}],
    'Rb': [{'mass': 85, 'stable': True}, {'mass': 87, 'radio': True, 'halfLife': 8, 'decay': 'beta'}],
    'Sr': [{'mass': 88, 'stable': True}, {'mass': 86, 'stable': True}, {'mass': 87, 'stable': True}, {'mass': 90, 'radio': True, 'halfLife': 4, 'decay': 'beta'}],
    'Y':  [{'mass': 89, 'stable': True}],
    'Zr': [{'mass': 90, 'stable': True}, {'mass': 91, 'stable': True}, {'mass': 92, 'stable': True}, {'mass': 94, 'stable': True}],
    'Nb': [{'mass': 93, 'stable': True}],
    'Mo': [{'mass': 98, 'stable': True}, {'mass': 96, 'stable': True}, {'mass': 95, 'stable': True}, {'mass': 92, 'stable': True}],
    'Tc': [{'mass': 98, 'radio': True, 'halfLife': 7, 'decay': 'beta'}, {'mass': 99, 'radio': True, 'halfLife': 6, 'decay': 'beta'}],
    'Ru': [{'mass': 102, 'stable': True}, {'mass': 104, 'stable': True}, {'mass': 101, 'stable': True}, {'mass': 99, 'stable': True}],
    'Rh': [{'mass': 103, 'stable': True}],
    'Pd': [{'mass': 106, 'stable': True}, {'mass': 108, 'stable': True}, {'mass': 105, 'stable': True}],
    'Ag': [{'mass': 107, 'stable': True}, {'mass': 109, 'stable': True}],
    'Cd': [{'mass': 114, 'stable': True}, {'mass': 112, 'stable': True}, {'mass': 111, 'stable': True}],
    'In': [{'mass': 115, 'stable': True}, {'mass': 113, 'stable': True}],
    'Sn': [{'mass': 120, 'stable': True}, {'mass': 118, 'stable': True}, {'mass': 116, 'stable': True}],
    'Sb': [{'mass': 121, 'stable': True}, {'mass': 123, 'stable': True}],
    'Te': [{'mass': 130, 'stable': True}, {'mass': 128, 'stable': True}, {'mass': 126, 'stable': True}],
    'I':  [{'mass': 127, 'stable': True}, {'mass': 129, 'radio': True, 'halfLife': 7, 'decay': 'beta'}, {'mass': 131, 'radio': True, 'halfLife': 2, 'decay': 'beta'}],
    'Xe': [{'mass': 132, 'stable': True}, {'mass': 129, 'stable': True}, {'mass': 131, 'stable': True}, {'mass': 134, 'stable': True}],
    'Cs': [{'mass': 133, 'stable': True}, {'mass': 137, 'radio': True, 'halfLife': 5, 'decay': 'beta'}],
    'Ba': [{'mass': 138, 'stable': True}, {'mass': 137, 'stable': True}, {'mass': 136, 'stable': True}],
    'La': [{'mass': 139, 'stable': True}],
    'Ce': [{'mass': 140, 'stable': True}, {'mass': 142, 'radio': True, 'halfLife': 6, 'decay': 'beta'}],
    'Pr': [{'mass': 141, 'stable': True}],
    'Nd': [{'mass': 142, 'stable': True}, {'mass': 144, 'radio': True, 'halfLife': 7, 'decay': 'alpha'}, {'mass': 143, 'stable': True}],
    'Pm': [{'mass': 145, 'radio': True, 'halfLife': 5, 'decay': 'beta'}, {'mass': 147, 'radio': True, 'halfLife': 4, 'decay': 'beta'}],
    'Sm': [{'mass': 152, 'stable': True}, {'mass': 147, 'radio': True, 'halfLife': 6, 'decay': 'alpha'}, {'mass': 154, 'stable': True}],
    'Eu': [{'mass': 153, 'stable': True}],
    'Gd': [{'mass': 158, 'stable': True}, {'mass': 160, 'stable': True}, {'mass': 156, 'stable': True}],
    'Tb': [{'mass': 159, 'stable': True}],
    'Dy': [{'mass': 164, 'stable': True}, {'mass': 162, 'stable': True}, {'mass': 163, 'stable': True}],
    'Ho': [{'mass': 165, 'stable': True}],
    'Er': [{'mass': 166, 'stable': True}, {'mass': 167, 'stable': True}, {'mass': 168, 'stable': True}],
    'Tm': [{'mass': 169, 'stable': True}],
    'Yb': [{'mass': 174, 'stable': True}, {'mass': 172, 'stable': True}, {'mass': 173, 'stable': True}],
    'Lu': [{'mass': 175, 'stable': True}, {'mass': 176, 'radio': True, 'halfLife': 7, 'decay': 'beta'}],
    'Hf': [{'mass': 180, 'stable': True}, {'mass': 178, 'stable': True}, {'mass': 177, 'stable': True}],
    'Ta': [{'mass': 181, 'stable': True}],
    'W':  [{'mass': 184, 'stable': True}, {'mass': 186, 'stable': True}, {'mass': 182, 'stable': True}],
    'Re': [{'mass': 187, 'radio': True, 'halfLife': 8, 'decay': 'beta'}, {'mass': 185, 'stable': True}],
    'Os': [{'mass': 192, 'stable': True}, {'mass': 190, 'stable': True}, {'mass': 189, 'stable': True}],
    'Ir': [{'mass': 193, 'stable': True}, {'mass': 191, 'stable': True}],
    'Pt': [{'mass': 195, 'stable': True}, {'mass': 194, 'stable': True}, {'mass': 196, 'stable': True}],
    'Au': [{'mass': 197, 'stable': True}],
    'Hg': [{'mass': 202, 'stable': True}, {'mass': 200, 'stable': True}, {'mass': 199, 'stable': True}],
    'Tl': [{'mass': 205, 'stable': True}, {'mass': 203, 'stable': True}],
    'Pb': [{'mass': 208, 'stable': True}, {'mass': 206, 'stable': True}, {'mass': 207, 'stable': True}, {'mass': 210, 'radio': True, 'halfLife': 3, 'decay': 'beta'}],
    'Bi': [{'mass': 209, 'radio': True, 'halfLife': 8, 'decay': 'alpha'}],
    'Po': [{'mass': 210, 'radio': True, 'halfLife': 3, 'decay': 'alpha'}, {'mass': 218, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}],
    'At': [{'mass': 210, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}, {'mass': 211, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}],
    'Rn': [{'mass': 222, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}, {'mass': 220, 'radio': True, 'halfLife': 1, 'decay': 'alpha'}],
    'Fr': [{'mass': 223, 'radio': True, 'halfLife': 2, 'decay': 'beta'}],
    'Ra': [{'mass': 226, 'radio': True, 'halfLife': 3, 'decay': 'alpha'}, {'mass': 228, 'radio': True, 'halfLife': 2, 'decay': 'beta'}],
    'Ac': [{'mass': 227, 'radio': True, 'halfLife': 3, 'decay': 'beta'}, {'mass': 228, 'radio': True, 'halfLife': 2, 'decay': 'beta'}],
    'Th': [{'mass': 232, 'radio': True, 'halfLife': 8, 'decay': 'alpha'}, {'mass': 230, 'radio': True, 'halfLife': 6, 'decay': 'alpha'}, {'mass': 228, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}],
    'Pa': [{'mass': 231, 'radio': True, 'halfLife': 5, 'decay': 'alpha'}],
    'U':  [{'mass': 238, 'radio': True, 'halfLife': 8, 'decay': 'alpha'}, {'mass': 235, 'radio': True, 'halfLife': 7, 'decay': 'alpha'}, {'mass': 234, 'radio': True, 'halfLife': 6, 'decay': 'alpha'}],
    'Np': [{'mass': 237, 'radio': True, 'halfLife': 6, 'decay': 'alpha'}, {'mass': 239, 'radio': True, 'halfLife': 4, 'decay': 'beta'}],
    'Pu': [{'mass': 244, 'radio': True, 'halfLife': 8, 'decay': 'alpha'}, {'mass': 239, 'radio': True, 'halfLife': 7, 'decay': 'alpha'}, {'mass': 238, 'radio': True, 'halfLife': 6, 'decay': 'alpha'}],
    'Am': [{'mass': 243, 'radio': True, 'halfLife': 6, 'decay': 'alpha'}, {'mass': 241, 'radio': True, 'halfLife': 7, 'decay': 'alpha'}],
    'Cm': [{'mass': 247, 'radio': True, 'halfLife': 8, 'decay': 'alpha'}, {'mass': 245, 'radio': True, 'halfLife': 5, 'decay': 'alpha'}],
    'Bk': [{'mass': 247, 'radio': True, 'halfLife': 5, 'decay': 'alpha'}, {'mass': 249, 'radio': True, 'halfLife': 3, 'decay': 'beta'}],
    'Cf': [{'mass': 251, 'radio': True, 'halfLife': 4, 'decay': 'alpha'}, {'mass': 252, 'radio': True, 'halfLife': 3, 'decay': 'alpha'}],
    'Es': [{'mass': 252, 'radio': True, 'halfLife': 3, 'decay': 'alpha'}, {'mass': 254, 'radio': True, 'halfLife': 2, 'decay': 'beta'}],
    'Fm': [{'mass': 257, 'radio': True, 'halfLife': 3, 'decay': 'alpha'}, {'mass': 255, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}],
    'Md': [{'mass': 258, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}, {'mass': 260, 'radio': True, 'halfLife': 1, 'decay': 'alpha'}],
    'No': [{'mass': 259, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}, {'mass': 255, 'radio': True, 'halfLife': 1, 'decay': 'alpha'}],
    'Lr': [{'mass': 266, 'radio': True, 'halfLife': 3, 'decay': 'alpha'}, {'mass': 262, 'radio': True, 'halfLife': 2, 'decay': 'alpha'}],
}

DEFAULT_ISOTOPE = {}
for k, lst in ISOTOPES.items():
    stable = next((i for i in lst if i.get('stable')), lst[0])
    DEFAULT_ISOTOPE[k] = stable['mass']


# ---------------------------------------------------------------------------
# 动作空间
# ---------------------------------------------------------------------------
NUCLEAR_TARGETS = [k for k in PIECE_DEFS if not PIECE_DEFS[k].get('isHK')]
NUCLEAR_TARGET_IDX = {k: i for i, k in enumerate(NUCLEAR_TARGETS)}
NORMAL_MOVE_SIZE = 64 * 64                 # 4096
GIVE_ELECTRON_OFFSET = NORMAL_MOVE_SIZE    # 4096
GIVE_ELECTRON_SIZE = 64 * 8                # 512
NUCLEAR_OFFSET = GIVE_ELECTRON_OFFSET + GIVE_ELECTRON_SIZE   # 4608
NUCLEAR_SIZE = len(NUCLEAR_TARGETS)        # 102
ACTION_SPACE_SIZE = NUCLEAR_OFFSET + NUCLEAR_SIZE               # 4710


# ---------------------------------------------------------------------------
# 棋子类
# ---------------------------------------------------------------------------
class Piece:
    # 使用 __slots__ 显著降低每个棋子对象的内存（对 MCTS 树里大量 env 克隆尤为关键）
    __slots__ = (
        'defKey', 'symbol', 'Z', 'valence', 'baseValence', 'electroneg', 'period', 'group',
        'isHydrogenKing', 'noble', 'lanthanide', 'color', 'id', 'row', 'col',
        'sharedElectrons', 'bonusElectrons', 'bonusElectronRounds', 'chargeRounds',
        'stunned', 'stunRounds', 'electronsGiven', 'decayCounter', 'bondedWith',
        '_bonusBatches', '_spawnCol', 'mass', 'radioactive', 'halfLifeRounds',
        'decayMode', 'isotopeName', '_overflowRays',
    )

    def __init__(self, def_key, color, piece_id, isotope_spec=None):
        d = PIECE_DEFS[def_key]
        self.defKey = def_key
        self.symbol = d['symbol']
        self.Z = d['Z']
        self.valence = d['valence']
        self.baseValence = d['valence']
        self.electroneg = d['en']
        self.period = d['period']
        self.group = d['group']
        self.isHydrogenKing = d.get('isHK', False)
        self.noble = d.get('noble', False)
        self.lanthanide = d.get('lanthanide', False)
        self.color = color
        self.id = piece_id
        self.row = -1
        self.col = -1
        self.sharedElectrons = 0
        self.bonusElectrons = 0
        self.bonusElectronRounds = 0
        self.chargeRounds = 0
        self.stunned = False
        self.stunRounds = 0
        self.electronsGiven = 0
        self.decayCounter = 0
        self.bondedWith = None
        self._bonusBatches = []
        self._spawnCol = -1
        self._overflowRays = 0
        if isotope_spec is not None:
            self.mass = isotope_spec.get('mass')
            self.radioactive = isotope_spec.get('radio', False)
            self.halfLifeRounds = isotope_spec.get('halfLife', 0)
            self.decayMode = isotope_spec.get('decay', None)
            self.isotopeName = isotope_spec.get('name', None)
        else:
            self.mass = d['mass'] or DEFAULT_ISOTOPE.get(def_key, d['mass']) or 1
            self.radioactive = d.get('radio', False)
            self.halfLifeRounds = d.get('halfLife', 0)
            self.decayMode = d.get('decay', None)
            self.isotopeName = None

    @property
    def totalElectrons(self):
        return self.valence + self.sharedElectrons + self.bonusElectrons

    @property
    def charge(self):
        return self.electronsGiven - self.bonusElectrons

    @charge.setter
    def charge(self, v):
        self.electronsGiven = v + (self.bonusElectrons or 0)

    @property
    def effectiveMoveDirs(self):
        return min(self.totalElectrons, 8)

    @property
    def effectiveMoveSteps(self):
        return self.totalElectrons

    @property
    def isEvenPeriod(self):
        return self.period % 2 == 0

    def getPriorityDirections(self):
        base = PRIORITY_EVEN if self.isEvenPeriod else PRIORITY_ODD
        color_factor = 1 if self.color == COLOR_WHITE else -1
        dirs = [[dr * color_factor, dc] for dr, dc in base]
        col_mirror = 1 if (self._spawnCol >= 0 and self._spawnCol < 4) else -1
        dirs = [[dr, dc * col_mirror] for dr, dc in dirs]
        return dirs[:self.effectiveMoveDirs]

    def __repr__(self):
        return f"<Piece {self.symbol} {self.color} @({self.row},{self.col}) Z={self.Z} val={self.valence}>"


# ---------------------------------------------------------------------------
# 游戏环境
# ---------------------------------------------------------------------------
class ChemisEnv:
    def __init__(self):
        self.board = []
        self.pieces = []
        self.whitePieces = []
        self.blackPieces = []
        self.bonds = []
        self.currentTurn = COLOR_WHITE
        self.roundNumber = 1
        self.gameOver = False
        self.winner = None
        self.winReason = None
        self._whiteNoStepCount = 0
        self._blackNoStepCount = 0
        self._pendingLi = None   # 移动到底线但尚未升变的 Li
        self._rng = random.Random()
        self._ai_searching = False  # 为 True 时跳过将军过滤（加速 AI 搜索/自我对弈，与原游戏 _aiSearching 一致）
        self._last_step_had_random = False
        self.initBoard()

    # ------------------------------------------------------------------
    # 初始化 / 棋盘工具
    # ------------------------------------------------------------------
    def initBoard(self):
        self.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.pieces = []
        self.whitePieces = []
        self.blackPieces = []
        self.bonds = []
        self.roundNumber = 1
        self.gameOver = False
        self.winner = None
        self._whiteNoStepCount = 0
        self._blackNoStepCount = 0
        self._pendingLi = None
        piece_id = 0

        def place(p, r, c):
            p.row = r
            p.col = c
            p._spawnCol = c
            self.board[r][c] = p
            self.pieces.append(p)
            (self.whitePieces if p.color == COLOR_WHITE else self.blackPieces).append(p)

        back = ['O', 'P', 'Si', 'F', 'H', 'C', 'N', 'S']
        front = ['Na', 'Na', 'Li', 'Li', 'Li', 'Li', 'Na', 'Na']
        for i, k in enumerate(back):
            place(Piece(k, COLOR_WHITE, piece_id), 7, i); piece_id += 1
        for i, k in enumerate(front):
            place(Piece(k, COLOR_WHITE, piece_id), 6, i); piece_id += 1
        for i, k in enumerate(back):
            place(Piece(k, COLOR_BLACK, piece_id), 0, i); piece_id += 1
        for i, k in enumerate(front):
            place(Piece(k, COLOR_BLACK, piece_id), 1, i); piece_id += 1
        self.rebuildAllBonds()

    def clone(self):
        """深拷贝一份环境，用于 MCTS 子节点展开。"""
        c = ChemisEnv.__new__(ChemisEnv)
        c.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        c.pieces = []
        c.whitePieces = []
        c.blackPieces = []
        c.bonds = [dict(b) for b in self.bonds]
        c.currentTurn = self.currentTurn
        c.roundNumber = self.roundNumber
        c.gameOver = self.gameOver
        c.winner = self.winner
        c.winReason = self.winReason
        c._whiteNoStepCount = self._whiteNoStepCount
        c._blackNoStepCount = self._blackNoStepCount
        c._pendingLi = None
        c._rng = self._rng
        c._ai_searching = self._ai_searching
        c._last_step_had_random = self._last_step_had_random
        # 深拷贝棋子
        for p in self.pieces:
            np_ = copy.deepcopy(p)
            c.pieces.append(np_)
            (c.whitePieces if np_.color == COLOR_WHITE else c.blackPieces).append(np_)
            if np_.row >= 0 and np_.col >= 0:
                c.board[np_.row][np_.col] = np_
        if self._pendingLi is not None:
            c._pendingLi = c.getPieceById(self._pendingLi.id)
        return c

    def getPieceById(self, pid):
        for p in self.pieces:
            if p.id == pid:
                return p
        return None

    def getAdjacentPieces(self, row, col):
        res = []
        for dr, dc in ALL_DIRS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc]:
                res.append(self.board[nr][nc])
        return res

    def getNextId(self):
        return max([0] + [p.id for p in self.pieces]) + 1

    def placePiece(self, p):
        self.board[p.row][p.col] = p
        self.pieces.append(p)
        (self.whitePieces if p.color == COLOR_WHITE else self.blackPieces).append(p)

    def removePiece(self, p):
        if p is None:
            return
        if p.row >= 0 and p.col >= 0:
            self.board[p.row][p.col] = None
        if p in self.pieces:
            self.pieces.remove(p)
        arr = self.whitePieces if p.color == COLOR_WHITE else self.blackPieces
        if p in arr:
            arr.remove(p)
        self.removeAllBondsFor(p)
        self.rebuildAllBonds()

    # ------------------------------------------------------------------
    # 键合判定
    # ------------------------------------------------------------------
    def _covalentRank(self, a, b, refCol=None, refRow=None):
        # a,b: dict with enDiff, col, row
        if b['enDiff'] != a['enDiff']:
            return b['enDiff'] - a['enDiff']
        colA = max(refCol, a['col']) if refCol is not None else a['col']
        colB = max(refCol, b['col']) if refCol is not None else b['col']
        if colB != colA:
            return colB - colA
        rowA = max(refRow, a['row']) if refRow is not None else a['row']
        rowB = max(refRow, b['row']) if refRow is not None else b['row']
        if rowB != rowA:
            return rowB - rowA
        aPos = a['row'] * 8 + a['col']
        bPos = b['row'] * 8 + b['col']
        return aPos - bPos

    def judgeBondType(self, metalPiece, nonmetalPiece):
        exceptions = {
            'Li-H': 'ionic', 'Na-H': 'ionic', 'K-H': 'ionic', 'Rb-H': 'ionic',
            'Cs-H': 'ionic', 'Fr-H': 'ionic', 'Ca-H': 'ionic', 'Sr-H': 'ionic',
            'Ba-H': 'ionic', 'Ra-H': 'ionic', 'Be-H': 'covalent', 'Mg-H': 'covalent',
            'Na-P': 'covalent', 'Al-Cl': 'covalent', 'Fe-Cl': 'covalent',
            'Cu-Cl': 'covalent', 'Ag-I': 'covalent',
        }
        key = metalPiece.symbol + '-' + nonmetalPiece.symbol
        if key in exceptions:
            return exceptions[key]
        enDiff = abs(metalPiece.electroneg - nonmetalPiece.electroneg)
        metalValence = metalPiece.valence
        metalPeriod = metalPiece.period
        if enDiff > 2.0:
            return 'ionic'
        if enDiff < 0.8:
            return 'covalent'
        polarizingPower = metalValence / metalPeriod
        anionSoftness = nonmetalPiece.period
        if polarizingPower > 0.6 and anionSoftness >= 3:
            return 'covalent'
        if polarizingPower > 0.6 and anionSoftness <= 2:
            return 'ionic'
        if polarizingPower <= 0.5 and enDiff > 1.0:
            return 'ionic'
        return 'ionic' if enDiff > 1.7 else 'covalent'

    def _bondTargetsAt(self, selectedPiece, atRow, atCol, excludeId=None):
        if selectedPiece is None or selectedPiece.noble:
            return []
        if atRow < 0 or atCol < 0:
            return []
        results = []
        for p in self.pieces:
            if p.row < 0 or p.col < 0:
                continue
            if excludeId is not None and p.id == excludeId:
                continue
            if p.color == selectedPiece.color or p.noble or p is selectedPiece:
                continue
            dr = abs(p.row - atRow)
            dc = abs(p.col - atCol)
            if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
                continue
            if (selectedPiece.group == METAL and p.group == NONMETAL) or \
               (selectedPiece.group == NONMETAL and p.group == METAL):
                metal = selectedPiece if selectedPiece.group == METAL else p
                nonmetal = selectedPiece if selectedPiece.group == NONMETAL else p
                if self.judgeBondType(metal, nonmetal) == 'ionic':
                    continue
                octetSel = 2 if selectedPiece.isHydrogenKing else 8
                octetP = 2 if p.isHydrogenKing else 8
                needSel = octetSel - selectedPiece.totalElectrons
                needP = octetP - p.totalElectrons
                sharedE = min(max(0, needSel), max(0, needP))
                results.append(p.id)
                continue
            if selectedPiece.group == METAL and p.group == METAL:
                already = selectedPiece.bondedWith and p.id in selectedPiece.bondedWith
                if not already:
                    results.append(p.id)
                continue
            if selectedPiece.group == NONMETAL and p.group == NONMETAL:
                octetSel = 2 if selectedPiece.isHydrogenKing else 8
                octetP = 2 if p.isHydrogenKing else 8
                needSel = octetSel - selectedPiece.totalElectrons
                needP = octetP - p.totalElectrons
                sharedE = min(max(0, needSel), max(0, needP))
                results.append(p.id)
        if selectedPiece.group == NONMETAL and len(results) >= 1:
            return self._pickCovalentTarget(selectedPiece, results, atRow, atCol)
        return results

    def _pickCovalentTarget(self, selectedPiece, candidateIds, atRow, atCol):
        candidates = []
        for cid in candidateIds:
            p = self.getPieceById(cid)
            candidates.append({'id': cid, 'enDiff': abs(selectedPiece.electroneg - p.electroneg),
                               'col': p.col, 'row': p.row})
        candidates.sort(key=lambda cand: self._covRankKey(cand, atCol, atRow))
        best = candidates[0]
        target = self.getPieceById(best['id'])
        if target is None:
            return []
        if target.bondedWith and len(target.bondedWith) > 0 and selectedPiece.id not in target.bondedWith:
            cpId = target.bondedWith[0]
            cp = self.getPieceById(cpId)
            if cp is None:
                return [best['id']]
            cpRank = {'enDiff': abs(cp.electroneg - target.electroneg), 'col': cp.col, 'row': cp.row, 'id': cp.id}
            selRank = {'enDiff': best['enDiff'], 'col': atCol, 'row': atRow, 'id': selectedPiece.id}
            if self._covalentRank(selRank, cpRank, target.col, target.row) >= 0:
                return []
        return [best['id']]

    def _covRankKey(self, cand, refCol, refRow):
        # sort key mirrors _covalentRank order (returns a tuple we negate appropriately)
        return (-cand['enDiff'],
                -max(refCol, cand['col']),
                -max(refRow, cand['row']),
                cand['row'] * 8 + cand['col'])

    def rebuildAllBonds(self):
        for p in self.pieces:
            p.sharedElectrons = 0
            p.bondedWith = None
        self.bonds = []
        candidates = []
        for p in self.pieces:
            if p.noble or p.row < 0 or p.col < 0:
                continue
            neighbors = self.getAdjacentPieces(p.row, p.col)
            for n in neighbors:
                if n.color == p.color:
                    continue
                if n.noble:
                    continue
                if p.id > n.id:
                    continue
                enDiff = abs(p.electroneg - n.electroneg)
                bothNonMetal = p.group == NONMETAL and n.group == NONMETAL
                bothMetal = p.group == METAL and n.group == METAL
                isMetalNonmetal = (not bothNonMetal) and (not bothMetal)
                if isMetalNonmetal:
                    metal = p if p.group == METAL else n
                    nonmetal = p if p.group == NONMETAL else n
                    if self.judgeBondType(metal, nonmetal) == 'ionic':
                        continue
                    needP = 8 - p.totalElectrons
                    needN = 8 - n.totalElectrons
                    sharedE = min(max(0, needP), max(0, needN))
                    candidates.append({'p1': p.id, 'p2': n.id, 'type': 'covalent', 'shared': sharedE, 'enDiff': enDiff})
                    continue
                if bothMetal:
                    candidates.append({'p1': p.id, 'p2': n.id, 'type': 'metallic', 'shared': 1, 'enDiff': enDiff})
                elif bothNonMetal:
                    octetP = 2 if p.isHydrogenKing else 8
                    octetN = 2 if n.isHydrogenKing else 8
                    needP = octetP - p.totalElectrons
                    needN = octetN - n.totalElectrons
                    sharedE = min(max(0, needP), max(0, needN))
                    candidates.append({'p1': p.id, 'p2': n.id, 'type': 'covalent', 'shared': sharedE, 'enDiff': enDiff})

        metallicCandidates = [c for c in candidates if c['type'] == 'metallic']
        covalentCandidates = [c for c in candidates if c['type'] != 'metallic']

        # 共价键贪心排序
        covalentCandidates.sort(key=lambda c: self._bondSortKey(c))

        covBonded = set()

        def applyBond(c):
            p1 = self.getPieceById(c['p1'])
            p2 = self.getPieceById(c['p2'])
            if p1 is None or p2 is None:
                return
            p1.sharedElectrons = max(p1.sharedElectrons, c['shared'])
            p2.sharedElectrons = max(p2.sharedElectrons, c['shared'])
            if p1.bondedWith is None:
                p1.bondedWith = []
            if p2.bondedWith is None:
                p2.bondedWith = []
            if p2.id not in p1.bondedWith:
                p1.bondedWith.append(p2.id)
            if p1.id not in p2.bondedWith:
                p2.bondedWith.append(p1.id)
            self.bonds.append({'p1': c['p1'], 'p2': c['p2'], 'type': c['type'], 'shared': c['shared']})

        for c in covalentCandidates:
            if c['p1'] in covBonded or c['p2'] in covBonded:
                continue
            covBonded.add(c['p1'])
            covBonded.add(c['p2'])
            applyBond(c)

        covalentShared = {}
        for p in self.pieces:
            if p.group == METAL:
                covalentShared[p.id] = p.sharedElectrons

        for c in metallicCandidates:
            applyBond(c)

        self.expandMetallicClusters(covalentShared)

    def _bondSortKey(self, c):
        pa = self.getPieceById(c['p1'])
        pb = self.getPieceById(c['p2'])
        pa2 = self.getPieceById(c['p2'])
        pb2 = self.getPieceById(c['p1'])
        raCol = max(pa.col if pa else 0, pa2.col if pa2 else 0)
        rbCol = max(pb.col if pb else 0, pb2.col if pb2 else 0)
        raRow = max(pa.row if pa else 0, pa2.row if pa2 else 0)
        rbRow = max(pb.row if pb else 0, pb2.row if pb2 else 0)
        raMinPos = min((pa.row if pa else 0) * 8 + (pa.col if pa else 0),
                       (pa2.row if pa2 else 0) * 8 + (pa2.col if pa2 else 0))
        rbMinPos = min((pb.row if pb else 0) * 8 + (pb.col if pb else 0),
                       (pb2.row if pb2 else 0) * 8 + (pb2.col if pb2 else 0))
        return (-c['enDiff'], -raCol, -rbCol, -raRow, -rbRow, raMinPos, rbMinPos)

    def expandMetallicClusters(self, covalentShared=None):
        metallicBonds = [b for b in self.bonds if b['type'] == 'metallic']
        if len(metallicBonds) == 0:
            return
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for b in metallicBonds:
            union(b['p1'], b['p2'])
        clusters = {}
        for b in metallicBonds:
            for pid in (b['p1'], b['p2']):
                root = find(pid)
                clusters.setdefault(root, []).append(pid)
        for ids in clusters.values():
            members = [self.getPieceById(i) for i in ids]
            members = [m for m in members if m and m.group == METAL and m.row >= 0]
            if len(members) == 0:
                continue
            N = len(members)
            totalE = 0
            for m in members:
                totalE += m.valence
            sharedPerMetal = min(8, math.ceil(totalE / N) + 1) if totalE >= N else 0
            for m in members:
                covShare = (covalentShared.get(m.id, 0) if covalentShared else 0)
                m.sharedElectrons = min(8, covShare + max(0, sharedPerMetal - m.valence))

    def removeAllBondsFor(self, piece):
        self.bonds = [b for b in self.bonds if b['p1'] != piece.id and b['p2'] != piece.id]
        if piece.bondedWith:
            for pid in list(piece.bondedWith):
                other = self.getPieceById(pid)
                if other and other.bondedWith:
                    other.bondedWith = [x for x in other.bondedWith if x != piece.id]
                    if len(other.bondedWith) == 0:
                        other.bondedWith = None
        piece.bondedWith = None
        piece.sharedElectrons = 0

    # ------------------------------------------------------------------
    # 吃子判定
    # ------------------------------------------------------------------
    def canCapture(self, attacker, defender):
        if attacker is None or defender is None:
            return False
        if attacker.color == defender.color:
            return False
        if attacker.noble or defender.noble:
            return False
        isMetalNonmetal = (attacker.group == METAL and defender.group == NONMETAL) or \
                          (attacker.group == NONMETAL and defender.group == METAL)
        if isMetalNonmetal:
            metal = attacker if attacker.group == METAL else defender
            nonmetal = attacker if attacker.group == NONMETAL else defender
            if self.judgeBondType(metal, nonmetal) == 'ionic':
                return True
            if not defender.bondedWith or len(defender.bondedWith) == 0:
                return False
            if attacker.bondedWith and defender.id in attacker.bondedWith:
                return False
            return attacker.id not in defender.bondedWith
        if not defender.bondedWith or len(defender.bondedWith) == 0:
            return False
        if attacker.bondedWith and defender.id in attacker.bondedWith:
            return False
        return attacker.id not in defender.bondedWith

    # ------------------------------------------------------------------
    # 走法生成
    # ------------------------------------------------------------------
    def _rawLegalMoves(self, piece):
        moves = []
        if piece is None or piece.row < 0 or piece.col < 0 or piece.stunned:
            return moves
        dirs = piece.getPriorityDirections()
        maxSteps = piece.effectiveMoveSteps
        for dr, dc in dirs:
            for step in range(1, maxSteps + 1):
                nr = piece.row + dr * step
                nc = piece.col + dc * step
                if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                    break
                target = self.board[nr][nc]
                if target is not None:
                    if target.color != piece.color and self.canCapture(piece, target):
                        moves.append({'row': nr, 'col': nc, 'isAttack': True, 'targetPiece': target})
                    break
                else:
                    moves.append({'row': nr, 'col': nc, 'isAttack': False, 'targetPiece': None})
        return moves

    def getLegalMoves(self, piece, applyCheckFilter=True):
        moves = self._rawLegalMoves(piece)
        if not applyCheckFilter or self.gameOver or self._ai_searching:
            return moves
        color = piece.color
        safe = []
        for m in moves:
            if self._moveLeavesOwnKingSafe(piece, m, color):
                safe.append(m)
        return safe

    def _moveLeavesOwnKingSafe(self, piece, m, color):
        # 临时应用走法，再检查己方 H 是否被将军
        board = self.board
        fromRow, fromCol = piece.row, piece.col
        toRow, toCol = m['row'], m['col']
        capturedInPath = m.get('targetPiece')
        # 保存
        board[fromRow][fromCol] = None
        board[toRow][toCol] = piece
        oldRow, oldCol = piece.row, piece.col
        piece.row, piece.col = toRow, toCol
        # 移出捕获的棋子
        removed_target = None
        if capturedInPath is not None:
            removed_target = capturedInPath
            # 临时移除
            self._removePieceFake(capturedInPath)
        # 重算键合（吃子会改变键合状态）
        self.rebuildAllBonds()
        safe = not self.isInCheck(color)
        # 还原
        if removed_target is not None:
            self._restorePieceFake(removed_target)
        self.rebuildAllBonds()
        piece.row, piece.col = oldRow, oldCol
        board[toRow][toCol] = capturedInPath
        board[fromRow][fromCol] = piece
        return safe

    def _removePieceFake(self, p):
        if p.row >= 0:
            self.board[p.row][p.col] = None
        if p in self.pieces:
            self.pieces.remove(p)
        arr = self.whitePieces if p.color == COLOR_WHITE else self.blackPieces
        if p in arr:
            arr.remove(p)

    def _restorePieceFake(self, p):
        self.pieces.append(p)
        (self.whitePieces if p.color == COLOR_WHITE else self.blackPieces).append(p)
        if p.row >= 0:
            self.board[p.row][p.col] = p

    # ------------------------------------------------------------------
    # 吃子
    # ------------------------------------------------------------------
    def handleCapture(self, attacker, defender):
        isIonic = False
        if (attacker.group == METAL and defender.group == NONMETAL) or \
           (attacker.group == NONMETAL and defender.group == METAL):
            metal = attacker if attacker.group == METAL else defender
            nonmetal = attacker if attacker.group == NONMETAL else defender
            isIonic = self.judgeBondType(metal, nonmetal) == 'ionic'
        if defender.isHydrogenKing:
            self.gameOver = True
            self.winner = attacker.color
            self.winReason = 'hk_captured'
            self.removePiece(defender)
            return False
        if isIonic:
            defenderElectrons = defender.valence + defender.bonusElectrons
            maxOctet = 2 if attacker.isHydrogenKing else 8
            maxOctet = max(0, maxOctet - attacker.valence - attacker.sharedElectrons - attacker.bonusElectrons)
            actualGain = min(defenderElectrons, maxOctet)
            if actualGain > 0:
                self._addBonusElectrons(attacker, actualGain, 3)
            overflow = defenderElectrons - actualGain
            if overflow > 0:
                attacker._overflowRays = min(overflow, 8)
        self.removePiece(defender)
        return isIonic

    # ------------------------------------------------------------------
    # 移动
    # ------------------------------------------------------------------
    def movePiece(self, piece, toRow, toCol, isAttack, target):
        fromRow, fromCol = piece.row, piece.col
        self.board[fromRow][fromCol] = None
        wasIonic = False
        if isAttack and target is not None:
            piece.sharedElectrons = 0
            piece.bondedWith = None
            wasIonic = self.handleCapture(piece, target)
        elif self.board[toRow][toCol] is not None:
            self.board[fromRow][fromCol] = piece
            return False
        piece.row = toRow
        piece.col = toCol
        self.board[toRow][toCol] = piece
        self.updateAllBonds()
        if wasIonic and getattr(piece, '_overflowRays', None):
            numRays = piece._overflowRays
            del piece._overflowRays
            self._last_step_had_random = True
            # 发射溢出 β 射线（每一条打到第一个撞见的棋子，随机电子±1/裂变）
            centerIdx = self._rng.randrange(8)
            for fi in range(numRays):
                offset = round((fi - (numRays - 1) / 2) * 2)
                dirIdx = (centerIdx + offset) % 8
                fdir = ALL_DIRS[dirIdx]
                self._emitOverflowRay(piece, fdir)
        return True

    def _emitOverflowRay(self, source, fdir):
        hit = False
        for s in range(1, 5):
            nr = source.row + fdir[0] * s
            nc = source.col + fdir[1] * s
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                break
            tgt = self.board[nr][nc]
            if tgt is None or tgt not in self.pieces:
                continue
            hit = True
            if self.tryFissionOnHit(tgt, 'beta'):
                break
            change = 1 if self._rng.random() < 0.5 else -1
            if tgt.isHydrogenKing:
                maxBonus = max(0, 2 - tgt.valence - tgt.sharedElectrons - tgt.bonusElectrons)
                clamped = max(0, min(tgt.bonusElectrons + change, maxBonus) - tgt.bonusElectrons)
                self._addBonusElectrons(tgt, clamped, 3)
            else:
                self._addBonusElectrons(tgt, change, 3)
            break
        return hit

    def updateAllBonds(self):
        self.rebuildAllBonds()

    # ------------------------------------------------------------------
    # 将军检测
    # ------------------------------------------------------------------
    def _canCaptureHFrom(self, piece, row, col, oppH):
        if oppH is None or oppH.row < 0:
            return False
        if piece is None or piece.row < 0 or piece.col < 0:
            return False
        if piece.color == oppH.color:
            return False
        if piece.noble or oppH.noble:
            return False
        dr = oppH.row - row
        dc = oppH.col - col
        steps = max(abs(dr), abs(dc))
        if steps < 1:
            return False
        if abs(dr) != abs(dc) and dr != 0 and dc != 0:
            return False
        sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
        sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
        cr, cc = row + sdr, col + sdc
        while cr != oppH.row or cc != oppH.col:
            if cr < 0 or cr >= 8 or cc < 0 or cc >= 8:
                return False
            if self.board[cr][cc] is not None:
                return False
            cr += sdr
            cc += sdc
        if steps > piece.effectiveMoveSteps:
            return False
        dirs = piece.getPriorityDirections()
        if not any(ddr == sdr and ddc == sdc for ddr, ddc in dirs):
            return False
        return self.canCapture(piece, oppH)

    def _canCapturePiece(self, attacker, target, targetRow, targetCol):
        if attacker is None or target is None:
            return False
        if attacker.row < 0 or attacker.col < 0 or targetRow < 0 or targetCol < 0:
            return False
        dr = targetRow - attacker.row
        dc = targetCol - attacker.col
        steps = max(abs(dr), abs(dc))
        if steps < 1:
            return False
        if abs(dr) != abs(dc) and dr != 0 and dc != 0:
            return False
        sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
        sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
        dirs = attacker.getPriorityDirections()
        if not any(ddr == sdr and ddc == sdc for ddr, ddc in dirs):
            return False
        if steps > attacker.effectiveMoveSteps:
            return False
        cr, cc = attacker.row + sdr, attacker.col + sdc
        while cr != targetRow or cc != targetCol:
            if cr < 0 or cr >= 8 or cc < 0 or cc >= 8:
                return False
            if self.board[cr][cc] is not None:
                return False
            cr += sdr
            cc += sdc
        return self.canCapture(attacker, target)

    def _canBeCapturedByOpponent(self, piece, row, col):
        """判断对方是否有棋子能吃到 (row,col) 处的 piece（用于判断将军是否会被反吃）。"""
        oppColor = COLOR_BLACK if piece.color == COLOR_WHITE else COLOR_WHITE
        oppPieces = self.blackPieces if oppColor == COLOR_WHITE else self.whitePieces
        for opp in oppPieces:
            if opp.row < 0 or opp.col < 0 or opp.stunned or opp.totalElectrons <= 0:
                continue
            if self._canCapturePiece(opp, piece, row, col):
                return True
        return False

    def isInCheck(self, color):
        myPieces = self.whitePieces if color == COLOR_WHITE else self.blackPieces
        oppPieces = self.blackPieces if color == COLOR_WHITE else self.whitePieces
        myH = next((p for p in myPieces if p.isHydrogenKing and p.row >= 0), None)
        if myH is None:
            return False
        for opp in oppPieces:
            if opp.row < 0 or opp.col < 0 or opp.stunned or opp.totalElectrons <= 0:
                continue
            if self._canCaptureHFrom(opp, opp.row, opp.col, myH):
                return True
        return False

    def hasAvailableSteps(self, color):
        pieces = self.whitePieces if color == COLOR_WHITE else self.blackPieces
        for p in pieces:
            if p.row < 0 or p.col < 0:
                continue
            if p.stunned or p.totalElectrons <= 0:
                continue
            if len(self.getLegalMoves(p)) > 0:
                return True
            if p.group == METAL and p.totalElectrons > 0:
                neighbors = self.getAdjacentPieces(p.row, p.col)
                if any(n.row >= 0 and n.col >= 0 and n.color == p.color and n.group == NONMETAL and not n.noble
                       and (n.totalElectrons < (2 if n.isHydrogenKing else 8)) for n in neighbors):
                    return True
        return False

    def isCheckmate(self, color):
        if self.gameOver:
            return False
        if not self.isInCheck(color):
            return False
        pieces = self.whitePieces if color == COLOR_WHITE else self.blackPieces
        for p in pieces:
            if p.row < 0 or p.col < 0 or p.stunned or p.totalElectrons <= 0:
                continue
            if len(self.getLegalMoves(p)) > 0:
                return False
        return True

    # ------------------------------------------------------------------
    # 给电子
    # ------------------------------------------------------------------
    def canGiveElectron(self, metal):
        if metal is None or metal.row < 0 or metal.col < 0 or metal.group != METAL \
           or metal.color != self.currentTurn or metal.totalElectrons <= 0:
            return None
        return [n for n in self.getAdjacentPieces(metal.row, metal.col)
                if n.row >= 0 and n.col >= 0 and n.color == metal.color and n.group == NONMETAL and not n.noble
                and (n.totalElectrons < (2 if n.isHydrogenKing else 8))]

    def giveElectron(self, metal, nonmetal):
        if nonmetal.isHydrogenKing and nonmetal.totalElectrons >= 2:
            return
        if not nonmetal.isHydrogenKing and nonmetal.totalElectrons >= 8:
            return
        clusterSet = {metal.id}
        queue = [metal.id]
        while queue:
            cur = self.getPieceById(queue.pop(0))
            if cur is None or not cur.bondedWith:
                continue
            for pid in cur.bondedWith:
                if pid in clusterSet:
                    continue
                p = self.getPieceById(pid)
                if p is None or p.group != METAL or p.row < 0:
                    continue
                if not any(b['type'] == 'metallic' and
                           ((b['p1'] == cur.id and b['p2'] == pid) or (b['p1'] == pid and b['p2'] == cur.id))
                           for b in self.bonds):
                    continue
                clusterSet.add(pid)
                queue.append(pid)
        cluster = [self.getPieceById(i) for i in clusterSet]
        cluster = [p for p in cluster if p and p.group == METAL and p.row >= 0]
        if len(cluster) > 1:
            eligible = [p for p in cluster if p.electronsGiven < p.baseValence]
            if not eligible:
                return
            eligible.sort(key=lambda p: (p.electronsGiven, p.id))
            donor = eligible[0]
            donor.valence = max(0, donor.valence - 1)
            donor.electronsGiven += 1
            donor.chargeRounds = max(donor.chargeRounds or 0, 3)
        else:
            if metal.electronsGiven >= metal.baseValence:
                return
            metal.valence = max(0, metal.valence - 1)
            metal.electronsGiven += 1
            metal.chargeRounds = 3
        nonmetal.chargeRounds = 3
        self._addBonusElectrons(nonmetal, 1, 3)
        self.rebuildAllBonds()

    # ------------------------------------------------------------------
    # 核变 (Li 升变)
    # ------------------------------------------------------------------
    def canNuclearTransform(self, piece):
        if piece is None or piece.symbol != 'Li' or piece.color != self.currentTurn:
            return False
        return piece.row == 0 if piece.color == COLOR_WHITE else piece.row == 7

    def nuclearTransform(self, piece, newKey, isotopeSpec=None):
        if newKey not in PIECE_DEFS or (newKey == 'H'):
            return
        np_ = Piece(newKey, piece.color, piece.id, isotopeSpec or None)
        np_.row = piece.row
        np_.col = piece.col
        np_._spawnCol = piece.col
        self.board[piece.row][piece.col] = np_
        idx = self.pieces.index(piece)
        self.pieces[idx] = np_
        arr = self.whitePieces if piece.color == COLOR_WHITE else self.blackPieces
        arr[arr.index(piece)] = np_
        self.removeAllBondsFor(piece)
        self.rebuildAllBonds()
        return np_

    # ------------------------------------------------------------------
    # 电子批次 / 计时器
    # ------------------------------------------------------------------
    def _addBonusElectrons(self, piece, amount, rounds):
        if amount == 0:
            return
        if amount > 0:
            piece.bonusElectrons += amount
            if not hasattr(piece, '_bonusBatches') or piece._bonusBatches is None:
                piece._bonusBatches = []
            piece._bonusBatches.append({'amount': amount, 'roundsLeft': rounds})
            piece.bonusElectronRounds = max(piece.bonusElectronRounds or 0, rounds)
        else:
            toRemove = abs(amount)
            if not hasattr(piece, '_bonusBatches') or piece._bonusBatches is None:
                piece._bonusBatches = []
            i = 0
            while i < len(piece._bonusBatches) and toRemove > 0:
                batch = piece._bonusBatches[i]
                removeFromBatch = min(batch['amount'], toRemove)
                batch['amount'] -= removeFromBatch
                toRemove -= removeFromBatch
                if batch['amount'] <= 0:
                    piece._bonusBatches.pop(i)
                else:
                    i += 1
            piece.bonusElectrons = max(0, piece.bonusElectrons - abs(amount))
            piece.bonusElectronRounds = max([b['roundsLeft'] for b in piece._bonusBatches]) if piece._bonusBatches else 0

    def decrementTimers(self):
        for p in self.pieces:
            if p._bonusBatches and len(p._bonusBatches) > 0:
                for i in range(len(p._bonusBatches) - 1, -1, -1):
                    p._bonusBatches[i]['roundsLeft'] -= 1
                    if p._bonusBatches[i]['roundsLeft'] <= 0:
                        p.bonusElectrons = max(0, p.bonusElectrons - p._bonusBatches[i]['amount'])
                        p._bonusBatches.pop(i)
                p.bonusElectronRounds = max([b['roundsLeft'] for b in p._bonusBatches]) if p._bonusBatches else 0
            if p.chargeRounds > 0:
                p.chargeRounds -= 1
                if p.chargeRounds == 0:
                    if p.electronsGiven > 0:
                        p.valence += p.electronsGiven
                        p.electronsGiven = 0
            if p.stunRounds > 0:
                p.stunRounds -= 1
                if p.stunRounds == 0:
                    p.stunned = False

    # ------------------------------------------------------------------
    # 电性吸引
    # ------------------------------------------------------------------
    def resolveElectrostaticAttraction(self):
        charged = [p for p in self.pieces if p.charge != 0 and p.row >= 0 and p.col >= 0]
        if len(charged) < 2:
            return
        pairs = []
        for i in range(len(charged)):
            for j in range(i + 1, len(charged)):
                if charged[i].charge * charged[j].charge < 0:
                    pairs.append([charged[i], charged[j]])
        if not pairs:
            return

        def pairRank(pair):
            a, b = pair
            crossColor = 1 if a.color != b.color else 0
            dr = a.row - b.row
            dc = a.col - b.col
            r2 = dr * dr + dc * dc or 1
            coulomb = abs(a.charge * b.charge) / r2
            enDiff = abs(a.electroneg - b.electroneg)
            rightmostCol = max(a.col, b.col)
            rightmostRow = max(a.row, b.row)
            return [crossColor, coulomb, enDiff, rightmostCol, rightmostRow]

        pairs.sort(key=lambda pair: [-x for x in pairRank(pair)])
        p1, p2 = pairs[0]
        s1 = abs(p1.charge)
        s2 = abs(p2.charge)
        if self.currentTurn == COLOR_WHITE and p1.color == COLOR_BLACK:
            s1 = max(0, s1 - 1)
        if self.currentTurn == COLOR_BLACK and p1.color == COLOR_WHITE:
            s1 = max(0, s1 - 1)
        self.moveToward(p1, p2, s1)
        self.moveToward(p2, p1, s2)
        self.updateAllBonds()

    def moveToward(self, mover, target, steps):
        if steps <= 0:
            return
        dr = 0 if target.row - mover.row == 0 else (1 if target.row - mover.row > 0 else -1)
        dc = 0 if target.col - mover.col == 0 else (1 if target.col - mover.col > 0 else -1)
        cr, cc = mover.row, mover.col
        rem = steps
        lastEmptyR, lastEmptyC = cr, cc
        while rem > 0:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                break
            if nr == target.row and nc == target.col:
                break
            rem -= 1
            if self.board[nr][nc]:
                cr, cc = nr, nc
                continue
            cr, cc = nr, nc
            lastEmptyR, lastEmptyC = cr, cc
        if lastEmptyR == mover.row and lastEmptyC == mover.col:
            return
        self.board[mover.row][mover.col] = None
        mover.row, mover.col = lastEmptyR, lastEmptyC
        self.board[lastEmptyR][lastEmptyC] = mover

    # ------------------------------------------------------------------
    # 核衰变 / 射线
    # ------------------------------------------------------------------
    def findKeyByZ(self, z):
        return Z_TO_KEY.get(z)

    def findClosestIsotope(self, elementKey, targetMass):
        isos = ISOTOPES.get(elementKey, [])
        if not isos:
            return None
        best = isos[0]
        bestDiff = abs(best['mass'] - targetMass)
        for iso in isos:
            d = abs(iso['mass'] - targetMass)
            if d < bestDiff:
                best = iso
                bestDiff = d
        return best

    def tryFissionOnHit(self, target, rayName):
        if target.Z >= 90 and self._rng.random() < 0.4:
            self.fissionEffect(target)
            return True
        return False

    def transformPiece(self, piece, newKey):
        if newKey not in PIECE_DEFS:
            return None
        if piece.isHydrogenKing:
            self.gameOver = True
            self.winner = COLOR_BLACK if piece.color == COLOR_WHITE else COLOR_WHITE
            self.winReason = 'hk_transmute'
            self.removePiece(piece)
            return None
        np_ = Piece(newKey, piece.color, piece.id, None)
        np_.row = piece.row
        np_.col = piece.col
        np_.chargeRounds = piece.chargeRounds
        np_.stunned = piece.stunned
        np_.stunRounds = piece.stunRounds
        np_.bonusElectrons = piece.bonusElectrons
        np_.bonusElectronRounds = piece.bonusElectronRounds
        np_._bonusBatches = copy.deepcopy(piece._bonusBatches)
        np_.electronsGiven = piece.electronsGiven
        self.board[piece.row][piece.col] = np_
        idx = self.pieces.index(piece)
        self.pieces[idx] = np_
        arr = self.whitePieces if piece.color == COLOR_WHITE else self.blackPieces
        arr[arr.index(piece)] = np_
        self.removeAllBondsFor(piece)
        self.rebuildAllBonds()
        return np_

    def resolveRadioactiveDecay(self):
        snapshot = list(self.pieces)
        for p in snapshot:
            if p not in self.pieces:
                continue
            if p.radioactive and p.halfLifeRounds > 0:
                p.decayCounter += 1
                if p.decayCounter >= p.halfLifeRounds and self._rng.random() < 0.55:
                    self.triggerDecay(p)
                    p.decayCounter = 0

    def triggerDecay(self, piece):
        if piece is None or piece not in self.pieces:
            return
        self._last_step_had_random = True
        mode = piece.decayMode or 'alpha'
        dirIdx = self._rng.randrange(8)
        d = ALL_DIRS[dirIdx]
        rng_ = 2 if mode == 'alpha' else (4 if mode == 'beta' else 8)
        emitGamma = (mode in ('alpha', 'beta')) and self._rng.random() < 0.5
        gammaDir = ALL_DIRS[self._rng.randrange(8)] if emitGamma else None
        if mode == 'alpha':
            newZ = piece.Z - 2
            newKey = self.findKeyByZ(newZ)
            if newKey:
                np_ = self.transformPiece(piece, newKey)
                if np_ is None:
                    return
                self.alphaRayEffect(np_, d)
                if emitGamma and gammaDir:
                    self.gammaRayEffect(np_, gammaDir)
            else:
                piece.stunned = True
                piece.stunRounds = 1
        elif mode == 'beta':
            newZ = piece.Z + 1
            newKey = self.findKeyByZ(newZ)
            if newKey:
                np_ = self.transformPiece(piece, newKey)
                if np_ is None:
                    return
                self.betaRayEffect(np_, d)
                if emitGamma and gammaDir:
                    self.gammaRayEffect(np_, gammaDir)
            else:
                piece.stunned = True
                piece.stunRounds = 1
        else:
            self.gammaRayEffect(piece, d)

    def _rayFirstHit(self, source, d, frange):
        for s in range(1, frange + 1):
            nr = source.row + d[0] * s
            nc = source.col + d[1] * s
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                return None
            t = self.board[nr][nc]
            if t is not None and t not in self.pieces:
                continue
            if t is not None and t.id != source.id:
                return t
        return None

    def alphaRayEffect(self, source, d):
        for s in range(1, 3):
            nr = source.row + d[0] * s
            nc = source.col + d[1] * s
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                break
            t = self.board[nr][nc]
            if t is None or t not in self.pieces or t.id == source.id:
                continue
            if self.tryFissionOnHit(t, 'alpha'):
                return
            oldMass = t.mass
            newZ = t.Z + 2
            newKey = self.findKeyByZ(newZ)
            if newKey and newKey in PIECE_DEFS:
                targetMass = oldMass + 4
                iso = self.findClosestIsotope(newKey, targetMass)
                if iso:
                    if iso.get('stable'):
                        spec = {'mass': iso['mass'], 'radio': False, 'halfLife': 0, 'decay': None, 'name': iso.get('name')}
                    else:
                        spec = {'mass': iso['mass'], 'radio': True,
                                'halfLife': iso.get('halfLife', 3), 'decay': iso.get('decay', 'alpha'),
                                'name': iso.get('name')}
                    np_ = self.transformPiece(t, newKey)
                    if np_ is None:
                        return
                    np_.mass = spec['mass']
                    np_.radioactive = spec['radio']
                    np_.halfLifeRounds = spec['halfLife']
                    np_.decayMode = spec['decay']
                    np_.isotopeName = spec['name']
                    np_.stunned = True
                    np_.stunRounds = 1
                else:
                    t.stunned = True
                    t.stunRounds = 1
            else:
                t.valence = max(1, t.valence - 1)
                t.stunned = True
                t.stunRounds = 1
            return

    def betaRayEffect(self, source, d):
        for s in range(1, 5):
            nr = source.row + d[0] * s
            nc = source.col + d[1] * s
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                break
            t = self.board[nr][nc]
            if t is None or t not in self.pieces or t.id == source.id:
                continue
            if self.tryFissionOnHit(t, 'beta'):
                return
            change = 1 if self._rng.random() < 0.5 else -1
            if t.isHydrogenKing:
                maxBonus = max(0, 2 - t.valence - t.sharedElectrons - t.bonusElectrons)
                clamped = max(0, min(t.bonusElectrons + change, maxBonus) - t.bonusElectrons)
                self._addBonusElectrons(t, clamped, 3)
            else:
                self._addBonusElectrons(t, change, 3)
            return

    def gammaRayEffect(self, source, d):
        for s in range(1, 9):
            nr = source.row + d[0] * s
            nc = source.col + d[1] * s
            if nr < 0 or nr >= 8 or nc < 0 or nc >= 8:
                break
            t = self.board[nr][nc]
            if t is None or t not in self.pieces or t.id == source.id:
                continue
            if self.tryFissionOnHit(t, 'gamma'):
                return
            t.stunned = True
            t.stunRounds = 1
            return

    def getAdjacentEmptyCells(self, r, c):
        res = []
        for d, e in ALL_DIRS:
            nr, nc = r + d, c + e
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] is None:
                res.append({'row': nr, 'col': nc})
        return res

    def fissionEffect(self, piece):
        self._last_step_had_random = True
        if piece.Z < 90:
            self.gammaRayEffect(piece, ALL_DIRS[self._rng.randrange(8)])
            return
        fissionPool = [k for k, d in PIECE_DEFS.items()
                       if 30 <= d['Z'] <= 60 and not d.get('noble') and not d.get('isHK')
                       and any(i.get('stable') for i in ISOTOPES.get(k, []))]
        if not fissionPool:
            return
        k1 = fissionPool[self._rng.randrange(len(fissionPool))]
        k2 = fissionPool[self._rng.randrange(len(fissionPool))]
        retries = 0
        while k2 == k1 and len(fissionPool) > 1 and retries < 50:
            k2 = fissionPool[self._rng.randrange(len(fissionPool))]
            retries += 1

        def makeIsotope(k):
            isos = ISOTOPES.get(k, [])
            stable = next((i for i in isos if i.get('stable')), isos[0]) if isos else {'mass': 0}
            return {'mass': stable['mass'], 'radio': bool(stable.get('radio')),
                    'halfLife': stable.get('halfLife', 0), 'decay': stable.get('decay'), 'name': stable.get('name')}

        cells = self.getAdjacentEmptyCells(piece.row, piece.col)
        if cells:
            iso1 = makeIsotope(k1)
            np1 = Piece(k1, piece.color, self.getNextId(), iso1)
            np1.row = cells[0]['row']
            np1.col = cells[0]['col']
            self.placePiece(np1)
            if len(cells) > 1:
                iso2 = makeIsotope(k2)
                np2 = Piece(k2, piece.color, self.getNextId(), iso2)
                np2.row = cells[1]['row']
                np2.col = cells[1]['col']
                self.placePiece(np2)
        if piece.isHydrogenKing:
            self.gameOver = True
            self.winner = COLOR_BLACK if piece.color == COLOR_WHITE else COLOR_WHITE
            self.winReason = 'hk_fission'
        self.removePiece(piece)
        self.updateAllBonds()

    # ------------------------------------------------------------------
    # 回合结束（终局判定 / 随机处理）
    # ------------------------------------------------------------------
    def endTurn(self):
        self.updateAllBonds()
        if self.currentTurn == COLOR_WHITE:
            self._whiteNoStepCount = 0
        else:
            self._blackNoStepCount = 0
        self.currentTurn = COLOR_BLACK if self.currentTurn == COLOR_WHITE else COLOR_WHITE
        if self.currentTurn == COLOR_WHITE:
            self.roundNumber += 1
            self.decrementTimers()
        self._pendingLi = None
        self.resolveElectrostaticAttraction()
        self.resolveRadioactiveDecay()
        self.updateAllBonds()
        if not self.gameOver and self.isCheckmate(self.currentTurn):
            self.gameOver = True
            self.winner = COLOR_BLACK if self.currentTurn == COLOR_WHITE else COLOR_WHITE
            self.winReason = 'checkmate'
            return
        skipIter = 0
        while not self.gameOver and not self.hasAvailableSteps(self.currentTurn) and skipIter < 8:
            skipIter += 1
            if self.isCheckmate(self.currentTurn):
                self.gameOver = True
                self.winner = COLOR_BLACK if self.currentTurn == COLOR_WHITE else COLOR_WHITE
                self.winReason = 'checkmate'
                break
            if self.currentTurn == COLOR_WHITE:
                self._whiteNoStepCount += 1
            else:
                self._blackNoStepCount += 1
            cnt = self._whiteNoStepCount if self.currentTurn == COLOR_WHITE else self._blackNoStepCount
            if cnt >= 4:
                self.gameOver = True
                self.winner = 'draw'
                self.winReason = 'draw_by_skip'
                break
            self.currentTurn = COLOR_BLACK if self.currentTurn == COLOR_WHITE else COLOR_WHITE
            if self.currentTurn == COLOR_WHITE:
                self.roundNumber += 1
                self.decrementTimers()
            self.updateAllBonds()

    # ------------------------------------------------------------------
    # 动作 / 一步交互
    # ------------------------------------------------------------------
    def _dirIndexOf(self, dr, dc):
        for i, (ddr, ddc) in enumerate(ALL_DIRS):
            if ddr == dr and ddc == dc:
                return i
        return -1

    def getAvailableActions(self, for_color=None):
        color = for_color or self.currentTurn
        actions = []
        if self.gameOver:
            return actions
        # 待升变的 Li：只能核变
        if self._pendingLi is not None:
            for key in NUCLEAR_TARGETS:
                actions.append(NUCLEAR_OFFSET + NUCLEAR_TARGET_IDX[key])
            return actions
        pieces = self.whitePieces if color == COLOR_WHITE else self.blackPieces
        for p in pieces:
            if p.row < 0 or p.col < 0 or p.stunned or p.totalElectrons <= 0:
                continue
            legal = self.getLegalMoves(p)
            for m in legal:
                actions.append((p.row * 8 + p.col) * 64 + (m['row'] * 8 + m['col']))
            if p.group == METAL and p.electronsGiven < p.baseValence:
                targets = self.canGiveElectron(p)
                if targets:
                    for t in targets:
                        di = self._dirIndexOf(t.row - p.row, t.col - p.col)
                        if di >= 0:
                            actions.append(GIVE_ELECTRON_OFFSET + (p.row * 8 + p.col) * 8 + di)
        return actions

    def _result(self):
        features = build_features(self)
        done = self.gameOver
        return features, done

    def step(self, action):
        """执行动作。返回 (features, done)。"""
        if self.gameOver:
            return self._result()
        self._last_step_had_random = False
        color = self.currentTurn

        # 待升变 Li：只能核变
        if self._pendingLi is not None:
            if NUCLEAR_OFFSET <= action < ACTION_SPACE_SIZE:
                key = NUCLEAR_TARGETS[action - NUCLEAR_OFFSET]
                li = self._pendingLi
                self._pendingLi = None
                self.nuclearTransform(li, key, None)
                if not self.gameOver:
                    self.endTurn()
            return self._result()

        if 0 <= action < NORMAL_MOVE_SIZE:
            from_sq = action // 64
            to_sq = action % 64
            fr, fc = from_sq // 8, from_sq % 8
            tr, tc = to_sq // 8, to_sq % 8
            piece = self.board[fr][fc] if (0 <= fr < 8 and 0 <= fc < 8) else None
            if piece is None or piece.color != color:
                return self._result()
            legal = self.getLegalMoves(piece)
            move = next((m for m in legal if m['row'] == tr and m['col'] == tc), None)
            if move is None:
                return self._result()
            target = move.get('targetPiece')
            isAttack = move.get('isAttack', False)
            self.movePiece(piece, tr, tc, isAttack, target)
            # Li 走到底线 → 触发升变，暂停回合
            if not self.gameOver and piece.symbol == 'Li' and self.canNuclearTransform(piece):
                self._pendingLi = piece
                return self._result()
            if not self.gameOver:
                self.endTurn()
        elif GIVE_ELECTRON_OFFSET <= action < NUCLEAR_OFFSET:
            rel = action - GIVE_ELECTRON_OFFSET
            metal_sq, dir_idx = rel // 8, rel % 8
            mr, mc = metal_sq // 8, metal_sq % 8
            metal = self.board[mr][mc] if (0 <= mr < 8 and 0 <= mc < 8) else None
            if metal is None or metal.color != color or metal.group != METAL:
                return self._result()
            d = ALL_DIRS[dir_idx]
            nr, nc = mr + d[0], mc + d[1]
            if not (0 <= nr < 8 and 0 <= nc < 8):
                return self._result()
            nonmetal = self.board[nr][nc]
            if nonmetal is None or nonmetal.color != color or nonmetal.group != NONMETAL:
                return self._result()
            self.giveElectron(metal, nonmetal)
            if not self.gameOver:
                self.endTurn()
        elif NUCLEAR_OFFSET <= action < ACTION_SPACE_SIZE:
            key = NUCLEAR_TARGETS[action - NUCLEAR_OFFSET]
            pieces = self.whitePieces if color == COLOR_WHITE else self.blackPieces
            li = next((p for p in pieces if p.symbol == 'Li' and p.row >= 0 and self.canNuclearTransform(p)), None)
            if li is None:
                return self._result()
            self.nuclearTransform(li, key, None)
            if not self.gameOver:
                self.endTurn()
        return self._result()

    def lastOutcomeForPerspective(self, persp):
        """返回从 persp 方视角的胜率标签: 1 胜 / -1 负 / 0 和。"""
        if self.winner == 'draw' or self.winner is None:
            return 0.0
        return 1.0 if self.winner == persp else -1.0


# ---------------------------------------------------------------------------
# 特征构建: [17, 8, 8]，从 persp 方视角
# ---------------------------------------------------------------------------
def build_features(env, persp=None):
    import numpy as np
    if persp is None:
        persp = env.currentTurn
    opp = COLOR_BLACK if persp == COLOR_WHITE else COLOR_WHITE
    feat = np.zeros((17, 8, 8), dtype=np.float32)

    def fill(p, is_my):
        if p.row < 0 or p.col < 0:
            return
        r, c = p.row, p.col
        z = p.Z / 103.0
        total = p.totalElectrons / 12.0
        chg = p.charge / 8.0
        bonded = 1.0 if (p.bondedWith and len(p.bondedWith) > 0) else 0.0
        stun = 1.0 if p.stunned else 0.0
        radio = 1.0 if p.radioactive else 0.0
        hk = 1.0 if p.isHydrogenKing else 0.0
        adv = (7 - p.row) / 7.0 if p.color == COLOR_WHITE else p.row / 7.0
        base = 0 if is_my else 1
        feat[base + 0, r, c] = z          # 0/1: 己方/敌方 Z
        feat[base + 2, r, c] = total      # 2/3: 总电子
        feat[base + 4, r, c] = chg        # 4/5: 电荷
        feat[base + 6, r, c] = bonded     # 6/7: 键合
        feat[base + 8, r, c] = stun       # 8/9: 眩晕
        feat[base + 10, r, c] = radio     # 10/11: 放射性
        feat[base + 12, r, c] = hk        # 12/13: 氢王
        if is_my:
            feat[15, r, c] = adv
        else:
            feat[16, r, c] = adv

    for p in env.whitePieces:
        fill(p, persp == COLOR_WHITE)
    for p in env.blackPieces:
        fill(p, persp == COLOR_BLACK)
    feat[14, :, :] = env.roundNumber / 40.0
    return feat
