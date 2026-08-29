# -*- coding: utf-8 -*-
"""
battle.py
=========
三方大乱斗：MCTS-Solver（训练网络 + MCTS + Solver） vs AB剪枝（手写 AI negamax）
vs NNUE（训练网络价值头做静态评估 + alpha-beta 剪枝）。

公平性：每个引擎在每步都拿**同样的时间盒**（--budget_ms），在规定毫秒内给出最优着法，
从而反映“同样算力下谁更强/更快”。

用法：
    python battle.py --games 2 --budget_ms 300 --max_moves 100 --seed 1
"""

import argparse
import time
import math
import random
import numpy as np
import torch

from chemis_env import ChemisEnv, build_features, COLOR_WHITE
from model import ChemisNet
from mcts import MCTS
from handwritten_ai import HandwrittenAI
from nnue import NnueAI, load_nnue_net


# 引擎名字
MCTS_NAME = 'MCTS-Solver'
AB_NAME = 'AB剪枝'
NNUE_NAME = 'NNUE'


class MCTSPlayer:
    """MCTS(-Solver) 引擎封装：共用网络，use_solver=True，同时间盒。"""
    def __init__(self, net, device, use_solver=True, num_simulations=600):
        self.net = net
        self.device = device
        self.mcts = MCTS(net, device=device, num_simulations=num_simulations, use_solver=use_solver)

    def name(self):
        return MCTS_NAME

    def choose_action(self, env, budget_ms):
        env._ai_searching = True
        pi, _v = self.mcts.search(env, self.net, add_dirichlet=False, temperature=0,
                                  deadline_ms=budget_ms)
        if not pi:
            return None
        # temperature=0 已是 one-hot argmax；取非零那个
        nz = [(a, p) for a, p in pi.items() if p > 0]
        if nz:
            return max(nz, key=lambda t: t[1])[0]
        return max(pi.items(), key=lambda t: t[1])[0]


class ABPlayer:
    """手写 AI：negamax + alpha-beta，时间盒内迭代加深到尽可能深。"""
    def __init__(self, max_depth=6):
        self.max_depth = max_depth

    def name(self):
        return AB_NAME

    def choose_action(self, env, budget_ms):
        env._ai_searching = True
        best = None
        deadline = time.time() * 1000 + budget_ms
        # 迭代加深：从浅到深，在时间盒内尽量加深
        for d in range(1, self.max_depth + 1):
            remaining = deadline - time.time() * 1000
            if remaining <= 0:
                break
            ai = HandwrittenAI(depth=d)
            a = ai.choose_action(env, time_budget_ms=remaining)
            if a is not None:
                best = a
        return best


class NNUEPlayer:
    """NNUE：训练网络价值头静态评估 + alpha-beta，时间盒内迭代加深。"""
    def __init__(self, net, device, max_depth=8):
        self.net = net
        self.device = device
        self.nnue = NnueAI(net, device=device, max_depth=max_depth)

    def name(self):
        return NNUE_NAME

    def choose_action(self, env, budget_ms):
        env._ai_searching = True
        return self.nnue.choose_action(env, time_budget_ms=budget_ms)


# ---------------------------------------------------------------------------
# 一局对战
# ---------------------------------------------------------------------------
def play_game(white, black, budget_ms, max_moves=120, verbose=False):
    env = ChemisEnv()
    env._ai_searching = True
    moves = 0
    t_white = 0.0
    t_black = 0.0
    last_log = time.time()
    while not env.gameOver and moves < max_moves:
        player = white if env.currentTurn == COLOR_WHITE else black
        t0 = time.time()
        action = player.choose_action(env, budget_ms)
        dt = time.time() - t0
        if env.currentTurn == COLOR_WHITE:
            t_white += dt
        else:
            t_black += dt
        if action is None:
            break
        actions = env.getAvailableActions()
        if action not in actions:
            # 引擎给了非法动作，容错：取第一个合法动作
            action = actions[0] if actions else None
            if action is None:
                break
        env.step(action)
        moves += 1
        if verbose and (time.time() - last_log > 5 or moves <= 2):
            last_log = time.time()
            print(f"    ... move {moves}/{max_moves} turn={env.currentTurn} winner={env.winner}",
                  flush=True)
    return {'winner': env.winner, 'moves': moves, 't_white': t_white, 't_black': t_black}


def engines(net, device, use_solver=True):
    return {
        MCTS_NAME: MCTSPlayer(net, device, use_solver=use_solver),
        AB_NAME: ABPlayer(),
        NNUE_NAME: NNUEPlayer(net, device),
    }


def run_tournament(net, device, games_per_pair=2, budget_ms=300, max_moves=120,
                   seed=0, pairs=None, use_solver=True, verbose=False):
    engine = engines(net, device, use_solver=use_solver)
    names = list(engine.keys())
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    results = {}
    for wname, bname in pairs:
        results[(wname, bname)] = {'w': 0, 'l': 0, 'd': 0, 'moves': [], 't_white': [], 't_black': []}
        games = games_per_pair
        for g in range(games):
            # 交替主色，综合两个引擎谁先手都覆盖
            if g % 2 == 0:
                white, black = engine[wname], engine[bname]
            else:
                white, black = engine[bname], engine[wname]
            print(f"[{wname} vs {bname}] game {g+1}/{games} ({white.name()}白/{black.name()}黑)",
                  flush=True)
            r = play_game(white, black, budget_ms, max_moves, verbose=verbose)
            outcome = r['winner']
            res = results[(wname, bname)]
            res['moves'].append(r['moves'])
            res['t_white'].append(r['t_white'])
            res['t_black'].append(r['t_black'])
            if outcome == 'draw' or outcome is None:
                res['d'] += 1
            else:
                # outcome 是胜者颜色；wname 白时 if outcome==white → w胜; 若黑方引擎就是 bname
                # 用实际对局中白/黑的引擎归属决定输赢
                if outcome == 'white' and g % 2 == 0:
                    res['w'] += 1
                elif outcome == 'black' and g % 2 == 1:
                    res['w'] += 1
                elif outcome == 'white' and g % 2 == 1:
                    res['l'] += 1
                elif outcome == 'black' and g % 2 == 0:
                    res['l'] += 1
                else:
                    res['d'] += 1
    return results


def print_results(results, total_games):
    print('\n' + '=' * 78)
    print(f'对战结果（每对 {total_games} 局）')
    print('=' * 78)
    header = f'{"对局":<22}{"胜":>4}{"负":>4}{"和":>4}{"平均局长":>10}{"白均走时":>10}{"黑均走时":>10}'
    print(header)
    for (wname, bname), res in results.items():
        mvs = res['moves']
        avg_m = sum(mvs) / len(mvs) if mvs else 0
        tw = sum(res['t_white']) / len(res['t_white']) if res['t_white'] else 0
        tb = sum(res['t_black']) / len(res['t_black']) if res['t_black'] else 0
        print(f'{wname+" vs "+bname:<22}{res["w"]:>4}{res["l"]:>4}{res["d"]:>4}{avg_m:>10.1f}{tw:>10.2f}{tb:>10.2f}')
    print('=' * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=2, help='每对（主色交替）局数')
    ap.add_argument('--budget_ms', type=int, default=300, help='每步时间盒(ms)')
    ap.add_argument('--max_moves', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--weights', default='model_weights_model.pt')
    ap.add_argument('--no_solver', action='store_true', help='MCTS 关闭 solver(对照)')
    ap.add_argument('--pairs', default=None, help='逗号分隔引擎对，如 MCTS-Solver,AB剪枝')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device if (args.device != 'cpu' or True) else 'cpu'
    net = load_nnue_net(args.weights, device=device)
    net.eval()

    pairs = None
    if args.pairs:
        pairs = [tuple(p.split(',')) for p in args.pairs.split(';') if ',' in p]

    t0 = time.time()
    results = run_tournament(net, device, games_per_pair=args.games, budget_ms=args.budget_ms,
                             max_moves=args.max_moves, seed=args.seed, pairs=pairs,
                             use_solver=not args.no_solver, verbose=args.verbose)
    print_results(results, args.games)
    print(f'总耗时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
