# -*- coding: utf-8 -*-
"""
nnue.py
=======
NNUE(inspired) 引擎 —— 用训练好的化学棋神经网络（策略+价值双头）做两件事：
1. **价值头**作为静态评估函数（替代手写 15 项 evaluate），供 alpha-beta(negamax) 使用；
2. **策略头 logits**作为走法排序的启发（优先搜索网络认为好的着法，大幅提升剪枝效率）。

这样它比 MCTS 快得多（每节点一次前向，不做整树多次模拟），却比纯手写评估更智能；
本质就是“神经网络评估 + alpha-beta 剪枝”的 NNUE 式混合搜索。

采用迭代加深 + 时间盒（deadline），保证在给定毫秒内给出最佳着法。
"""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from chemis_env import ChemisEnv, build_features, ACTION_SPACE_SIZE, COLOR_WHITE, COLOR_BLACK
from chemis_env import GIVE_ELECTRON_OFFSET, NUCLEAR_OFFSET, NUCLEAR_TARGETS, NUCLEAR_TARGET_IDX
from model import ChemisNet


# 与 HTML 端 evaluate() 对齐：net 价值 [-1,1] × 50000
VALUE_SCALE = 50000.0
WIN_SCORE = 100000000.0


class NnueAI:
    def __init__(self, net, device='cpu', max_depth=20, time_budget_ms=1000.0):
        self.net = net
        self.device = device
        self.max_depth = max_depth
        self.time_budget_ms = time_budget_ms
        self._deadline = None
        self._transposition = {}
        # 历史启发
        self._history = {}
        self._root_v = 0.0

    # ------------------------------------------------------------------
    # 静态评估：net 价值头（执子方视角），终局 ±1e8
    # ------------------------------------------------------------------
    def evaluate(self, env):
        if env.gameOver:
            return WIN_SCORE if env.winner == env.currentTurn else -WIN_SCORE
        feat = build_features(env)
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(x)
        return float(v.squeeze().item()) * VALUE_SCALE

    def _eval_with_logits(self, env):
        """返回 (value, policy_logits)。供根节点选着法时同时拿策略排序。"""
        feat = build_features(env)
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(x)
        return float(v.squeeze().item()) * VALUE_SCALE, logits.squeeze(0).cpu()

    # ------------------------------------------------------------------
    # 走法生成 + 排序（吃子/已胜优先，再按历史启发）
    # ------------------------------------------------------------------
    def _actions(self, env):
        env._ai_searching = True
        return env.getAvailableActions()

    def _order_score(self, env, a):
        if a >= NUCLEAR_OFFSET:
            return 50
        if a >= GIVE_ELECTRON_OFFSET:
            return 100
        fr = a // 64
        to = a % 64
        target = env.board[to // 8][to % 8]
        if target is not None:
            return 1000 if target.isHydrogenKing else 200
        return 0

    def _policy_order(self, env, actions, logits):
        """用策略 logits 作为主序，吃子/威胁加分。返回排序后 actions。"""
        ordered = []
        for a in actions:
            s = float(logits[a]) + self._order_score(env, a) * 0.01
            ordered.append((s, a))
        ordered.sort(key=lambda t: t[0], reverse=True)
        return [a for _, a in ordered]

    # ------------------------------------------------------------------
    # 转置表键
    # ------------------------------------------------------------------
    def _board_key(self, env):
        return (env.currentTurn, env.roundNumber,
                tuple(sorted((p.id, p.row, p.col, p.totalElectrons) for p in env.pieces if p.row >= 0)))

    # ------------------------------------------------------------------
    # negamax + alpha-beta（时间盒内迭代加深）
    # ------------------------------------------------------------------
    def choose_action(self, env, time_budget_ms=None, net=None):
        if time_budget_ms is not None:
            self.time_budget_ms = time_budget_ms
        self._deadline = time.time() * 1000 + self.time_budget_ms
        self._transposition = {}
        self._history = {}

        color = env.currentTurn
        actions = self._actions(env)
        if not actions:
            return None

        v_root, logits = self._eval_with_logits(env)
        self._root_v = v_root
        actions = self._policy_order(env, actions, logits)

        best = actions[0]
        completed = 0
        for depth in range(1, self.max_depth + 1):
            if time.time() * 1000 > self._deadline:
                break
            best, alpha_out = self._search_root(env, actions, depth)
            completed = depth
            if alpha_out >= WIN_SCORE * 0.5:
                break   # 已证明必胜，提前停
        return best

    def _search_root(self, env, actions, depth):
        best = actions[0]
        best_val = -1e18
        alpha = -1e18
        for a in actions:
            child = env.clone()
            child._ai_searching = True
            child.step(a)
            val = -self._negamax(child, depth - 1, -1e18, -alpha, child.currentTurn)
            if val > best_val:
                best_val = val
                best = a
            if val > alpha:
                alpha = val
            if time.time() * 1000 > self._deadline:
                break
        return best, alpha

    def _negamax(self, env, depth, alpha, beta, color):
        if time.time() * 1000 > self._deadline:
            return 0
        if env.gameOver:
            return WIN_SCORE if env.winner == color else -WIN_SCORE
        if depth <= 0:
            v = self.evaluate(env)
            # 归一到与胜分可比，但归一化会削弱区分，直接用原值即可（alpha-beta 只比大小）
            return v

        key = self._board_key(env)
        tt = self._transposition.get(key)
        if tt is not None and tt[0] >= depth:
            return tt[1]

        acts = self._actions(env)
        if not acts:
            return self.evaluate(env)

        # 排序：历史启发 + 吃子优先
        def oh(a):
            s = self._order_score(env, a)
            return s * 10 + (self._history.get((a, env.currentTurn), 0))
        acts = sorted(acts, key=oh, reverse=True)

        best = -1e18
        for a in acts:
            child = env.clone()
            child._ai_searching = True
            child.step(a)
            val = -self._negamax(child, depth - 1, -beta, -alpha, child.currentTurn)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                # 记录历史启发
                self._history[(a, env.currentTurn)] = self._history.get((a, env.currentTurn), 0) + depth * depth
                break
        self._transposition[key] = (depth, best)
        return best


def load_nnue_net(weights_path='model_weights_model.pt', device='cpu'):
    net = ChemisNet().to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    return net
