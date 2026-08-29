# -*- coding: utf-8 -*-
"""
mcts.py
=======
带有 PUCT 指导的蒙特卡洛树搜索（蒙特卡洛树搜索），AlphaZero / 随机环境适配版。
可选启用 **MCTS-Solver**（与 HTML 版逻辑一致）：终端节点视为“已证明”结果，
节点带 solved 标记（+1 本方必胜 / -1 本方必败 / 0 未知），选择时优先走本方必胜之子，
并在整棵树上做 minimax 式证据传播（子节点全证明时反推父节点胜负），
已证明结果在反向传播时覆盖样本平均 Q（避免再浪费模拟）。

关键点
------
1. PUCT 公式:  U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
2. **随机性处理**: 化学棋的走子会触发衰变/射线等随机事件。在 `expand()` 中对每个
   动作做采样：若该动作的转移是随机的（`env._last_step_had_random`），强制采样
   多次（STOCHASTIC_SAMPLES=3）生成多个子节点，把随机性转化为树上的概率分布；
   确定性动作只生成一个子节点。开启 solver 时关闭随机兄弟节点（minimax 证据传播
   要求每个动作只对应一个孩子，才能做 AND/OR 语义）。
3. 叶子节点直接用网络价值评估（不做完整随机对局 rollout），这是 AlphaZero 的做法。
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from chemis_env import ChemisEnv, build_features, ACTION_SPACE_SIZE

STOCHASTIC_SAMPLES = 3


class Node:
    __slots__ = ('env', 'prior', 'parent', 'action', 'children', 'N', 'W', 'Q',
                 'expanded', 'terminal', 'stochastic', 'solved')

    def __init__(self, env, prior, parent, action):
        self.env = env
        self.prior = prior
        self.parent = parent
        self.action = action
        self.children = []
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.expanded = False
        self.terminal = bool(env is not None and env.gameOver)
        self.stochastic = False
        self.solved = 0   # +1 本方必胜 / -1 本方必败 / 0 未知（MCTS-Solver）


class MCTS:
    def __init__(self, net, device='cpu', num_simulations=200, c_puct=1.0,
                 stochastic_samples=STOCHASTIC_SAMPLES, use_solver=False):
        self.net = net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.stochastic_samples = stochastic_samples
        self.use_solver = use_solver

    # ------------------------------------------------------------------
    # 网络前向
    # ------------------------------------------------------------------
    def _evaluate(self, env):
        feat = build_features(env)
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(x)
        logits = logits.squeeze(0)
        v = v.squeeze(0).squeeze(0).item()
        return logits.cpu(), v

    def _priors_for_legal(self, logits, legal_actions):
        if not legal_actions:
            return {}
        idx = torch.tensor(legal_actions, dtype=torch.long)
        sel = logits[idx]
        sel = sel - sel.max()
        exp = torch.exp(sel)
        probs = exp / exp.sum()
        return {a: float(probs[i]) for i, a in enumerate(legal_actions)}

    # ------------------------------------------------------------------
    # 转移（采样一次随机结果）
    # ------------------------------------------------------------------
    def _transition(self, env, action):
        child = env.clone()
        child._ai_searching = True
        child.step(action)
        return child

    # ------------------------------------------------------------------
    # 树搜索
    # ------------------------------------------------------------------
    def search(self, root_env, net=None, add_dirichlet=True, temperature=None, deadline_ms=None):
        """对 root_env 执子方做 MCTS，返回 (策略 π dict[action->prob], 价值 v)。

        temperature: 若不传则由调用方决定；传 0 直接返回最青睐动作的 one-hot π，
                     传 1 返回访问次数占比。
        deadline_ms: 时间盒（epoch ms）。达到即停止加模拟，保证与其它引擎公平。
        """
        import time as _time
        root = Node(root_env.clone(), prior=1.0, parent=None, action=None)
        root.env._ai_searching = True
        logits, root_v = self._evaluate(root.env)
        self._expand(root, logits)
        if add_dirichlet and len(root.children) > 0:
            # AlphaZero 式根节点探索噪声
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(root.children)).astype(np.float32)
            for c, n in zip(root.children, noise):
                c.prior = 0.75 * c.prior + 0.25 * n
        deadline = _time.time() * 1000 + deadline_ms if deadline_ms is not None else None
        for _ in range(self.num_simulations):
            if deadline is not None and _time.time() * 1000 > deadline:
                break
            self._simulate(root)
        # 统计访问次数 → 策略
        pi = {}
        total = sum(c.N for c in root.children)
        for c in root.children:
            pi[c.action] = pi.get(c.action, 0.0) + c.N / max(1, total)
        if temperature == 0:
            if root.children:
                best_c = max(root.children, key=lambda c: c.N)
                pi = {c.action: (1.0 if c is best_c else 0.0) for c in root.children}
        return pi, root_v

    def _simulate(self, node):
        while node.expanded and not node.terminal:
            child = self._select(node)
            if child.env is None:
                child.env = self._transition(node.env, child.action)
                child.terminal = child.env.gameOver
                child.stochastic = child.env._last_step_had_random
                if child.stochastic and not self.use_solver:
                    # 若为随机转移，生成额外兄弟子节点，把随机性变成分布
                    self._add_stochastic_siblings(node, child.action)
            node = child
        if node.terminal:
            value = node.env.lastOutcomeForPerspective(node.env.currentTurn)
            node.solved = int(value)   # 终局为确定性结果，视为“已证明”
        else:
            logits, value = self._evaluate(node.env)
            self._expand(node, logits)
        # 反向传播（逐层取反，价值始终以“当前执子方”视角计）
        self._backprop(node, value)
        # MCTS-Solver：沿路径向上做证据（solved）传播
        if self.use_solver:
            up = node.parent
            while up is not None:
                self._maybe_solve(up)
                up = up.parent

    def _backprop(self, node, value):
        while node is not None:
            node.N += 1
            if node.solved != 0:
                value = node.solved   # 已证明结果覆盖样本平均
            node.W += value
            node.Q = node.W / node.N
            if node.parent is not None:
                same = node.env.currentTurn == node.parent.env.currentTurn
                if not same:
                    value = -value
            node = node.parent

    def _select(self, node):
        # MCTS-Solver：存在“本方必胜”子节点时直接选，不再浪费模拟
        if self.use_solver:
            for c in node.children:
                if self._child_win_for_me(node, c) == 1:
                    return c
        best = None
        best_ucb = -1e18
        for c in node.children:
            if c.N == 0:
                u = c.prior * math.sqrt(node.N + 1)
            else:
                u = c.Q + self.c_puct * c.prior * math.sqrt(node.N) / (1 + c.N)
            if u > best_ucb:
                best_ucb = u
                best = c
        return best

    def _expand(self, node, logits):
        legal = node.env.getAvailableActions()
        node.children = []
        node.expanded = True
        if not legal:
            return
        priors = self._priors_for_legal(logits, legal)
        for a in legal:
            node.children.append(Node(env=None, prior=priors[a], parent=node, action=a))

    def _add_stochastic_siblings(self, node, action):
        # 统计该动作已有的子节点数
        existing = [c for c in node.children if c.action == action]
        count = len(existing)
        if count >= self.stochastic_samples:
            return
        target = min(self.stochastic_samples, count + 1)
        base_prior = existing[0].prior * count if count > 0 else 0.0
        # 重新平摊每个兄弟子节点（含已有）的 prior
        parent_prior = (existing[0].prior * count) if count > 0 else 0.0
        for c in existing:
            c.prior = parent_prior / target
        for _ in range(count, target):
            child = Node(env=None, prior=parent_prior / target, parent=node, action=action)
            node.children.append(child)

    # ------------------------------------------------------------------
    # MCTS-Solver 辅助：从 node 视角看 child 的已验证结果
    # ------------------------------------------------------------------
    def _child_win_for_me(self, node, child):
        if child.solved == 0 or node.env is None or child.env is None:
            return 0
        same = node.env.currentTurn == child.env.currentTurn
        return child.solved if same else -child.solved

    def _maybe_solve(self, node):
        """当某节点所有子节点都已证明时，用 minimax 逻辑推出该节点的胜负。"""
        if node.solved != 0 or not node.children:
            return
        all_solved = True
        any_win = False
        for c in node.children:
            win_me = self._child_win_for_me(node, c)
            if win_me == 0:
                all_solved = False
                break
            if win_me == 1:
                any_win = True
        if all_solved:
            node.solved = 1 if any_win else -1


def mcts_action_probs(mcts, root_env, net):
    """便捷入口。返回 (π dict, value, root_clone)。"""
    pi, value = mcts.search(root_env, net)
    return pi, value


def select_action_from_pi(pi, temperature=1.0, rng=None):
    """按 π 采样一个动作用于自我对弈（temperature=1 用多项式采样）。"""
    actions = list(pi.keys())
    if not actions:
        return None
    probs = np.array([pi[a] for a in actions], dtype=np.float64)
    probs = probs / probs.sum()
    if temperature == 0:
        idx = int(np.argmax(probs))
    else:
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / probs.sum()
        idx = int(np.random.choice(len(actions), p=probs))
    return actions[idx]
