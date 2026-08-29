# -*- coding: utf-8 -*-
"""
train_selfplay.py
=================
化学棋 AlphaZero 风格纯自我对弈训练 —— **异步混合架构：CPU 推理 + GPU 训练**。

架构
----
- 多个 **CPU worker 进程**各自独立跑 MCTS 推理（`model.to('cpu')`，每 worker 一份只读权重副本）。
  它们不断把 每局 的 (特征向量, MCTS 概率 π, 最终胜负 z) 通过 `multiprocessing.Queue` 写回。
- 一个 **GPU 主进程**从队列取数据（小数据：三元组，不含棋盘对象），在 `model.to('cuda')` 上做反向传播。
- 二者由 Queue 解耦：CPU 只负责源源不断产出数据，GPU 只负责训练。
- 每 `games_per_update` 局把新权重广播给所有 CPU worker（更新推理模型）。

内存优化（让 6 worker 跑满）
--------------------------
- `Piece` 使用 `__slots__`（见 chemis_env.py），大幅降低 MCTS 树里大量 env 克隆的对象内存。
- worker 只返回 `(特征向量, π, z)` 三元组，**绝不**回传棋盘对象。
- 强制 `spawn`（`force=True`），避免 Linux `fork` 复制整个父进程内存。

API
---
模块把核心训练抽成两个可复用函数，**不隐藏自动主循环**，由你按顺序调用：
    collect_data(buffer, sample_queue, ...)   # 从 worker 队列收集新对局 -> 写入经验池
    train_network(net, optimizer, buffer, ...) # GPU 上训练一个 batch
另有 worker 生命周期函数与一个可运行示例 `main()`。

用法（跑一整夜）：
    python train_selfplay.py --num_games 2000 --num_simulations 200 --batch_size 256 \
        --games_per_update 50 --workers 6 --device cuda --worker_device cpu
"""

import argparse
import os
import sys
import time
import random
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemis_env import ChemisEnv, build_features, ACTION_SPACE_SIZE
from model import ChemisNet, save_model_weights
from mcts import MCTS, select_action_from_pi
from handwritten_ai import HandwrittenAI

# 强制 spawn，避免 Linux fork 复制整个父进程内存（Windows 本来就只能是 spawn）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
_CTX = mp.get_context('spawn')


# ---------------------------------------------------------------------------
# 经验池（主进程使用）
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size=300_000):
        self.max_size = max_size
        self.features = []
        self.legal = []
        self.pi = []
        self.z = []

    def add(self, features, legal_actions, pi_vec, z):
        self.features.append(features)
        self.legal.append(np.asarray(legal_actions, dtype=np.int64))
        self.pi.append(np.asarray(pi_vec, dtype=np.float32))
        self.z.append(float(z))
        if len(self.features) > self.max_size:
            self.features.pop(0); self.legal.pop(0); self.pi.pop(0); self.z.pop(0)

    def __len__(self):
        return len(self.features)

    def sample(self, batch_size):
        n = len(self.features)
        idxs = random.sample(range(n), min(batch_size, n))
        feats = np.stack([self.features[i] for i in idxs])
        z = np.array([self.z[i] for i in idxs], dtype=np.float32)
        return feats, idxs, z


def pi_to_vec(pi_dict, legal_actions):
    return np.array([pi_dict.get(a, 0.0) for a in legal_actions], dtype=np.float32)


# ---------------------------------------------------------------------------
# AlphaZero 损失 + GPU 训练
# ---------------------------------------------------------------------------
def alpha_zero_loss(logits, v, pi_vec, legal_actions, z, c_reg=1e-4, params=None):
    B = logits.size(0)
    dev = logits.device
    value_loss = ((z - v) ** 2).mean()
    policy_loss = 0.0
    for i in range(B):
        la = legal_actions[i].to(dev)
        pi = pi_vec[i].to(dev)
        if la.numel() == 0:
            continue
        lp = F.log_softmax(logits[i][la], dim=0)
        policy_loss = policy_loss - (pi * lp).sum()
    policy_loss = policy_loss / B
    reg = 0.0
    if params is not None:
        for p in params:
            reg = reg + (p ** 2).sum()
    return value_loss + policy_loss + c_reg * reg, float(value_loss.item()), float(policy_loss.item()), float(reg.item())


def train_network(net, optimizer, buffer, batch_size, device, c_reg=1e-4):
    """在 device(GPU) 上训练一个 batch，返回 (loss, value_loss, policy_loss)。"""
    feats, idxs, z = buffer.sample(batch_size)
    x = torch.from_numpy(feats).to(device)
    zt = torch.from_numpy(z).to(device)
    legal = [torch.as_tensor(buffer.legal[i], dtype=torch.long) for i in idxs]
    pi = [torch.as_tensor(buffer.pi[i], dtype=torch.float32) for i in idxs]
    net.train()
    optimizer.zero_grad()
    logits, v = net(x)
    loss, vl, pl, _ = alpha_zero_loss(logits, v.squeeze(1), pi, legal, zt, c_reg, list(net.parameters()))
    loss.backward()
    optimizer.step()
    return float(loss.item()), vl, pl


# ---------------------------------------------------------------------------
# 自我对弈一局（worker 进程内运行；只返回小三元组）
# ---------------------------------------------------------------------------
def self_play_game(mcts, net, device, temperature_moves=30):
    env = ChemisEnv()
    env._ai_searching = True
    game_data = []
    move = 0
    while not env.gameOver:
        pi, _ = mcts.search(env, net, add_dirichlet=True)
        legal = list(pi.keys())
        if not legal:
            break
        temp = 1.0 if move < temperature_moves else 0.3
        game_data.append((build_features(env), legal, pi_to_vec(pi, legal), env.currentTurn))
        env.step(select_action_from_pi(pi, temperature=temp))
        move += 1
        if move > 120:
            break
    winner = env.winner
    samples = []
    for feat, legal, pivec, turn in game_data:
        z = 0.0 if (winner == 'draw' or winner is None) else (1.0 if winner == turn else -1.0)
        samples.append((feat, legal, pivec, z))   # 只返回特征/π/z，不含棋盘对象
    return samples, winner, move


# ---------------------------------------------------------------------------
# ML(MCTS) vs 手写 AI 教师：ML 执 ml_color，教师执另一侧，把对局数据进训练
# ML 盘面记录其 MCTS 分布 π；教师盘面记录 one-hot(教师所选动作)（即策略蒸馏）。
# ---------------------------------------------------------------------------
def self_play_vs_teacher(mcts, net, device, teacher_depth, ml_color, temperature_moves=30):
    env = ChemisEnv()
    env._ai_searching = True
    teacher = HandwrittenAI(depth=teacher_depth)
    game_data = []
    move = 0
    while not env.gameOver:
        if env.currentTurn == ml_color:
            pi, _ = mcts.search(env, net, add_dirichlet=True)
            legal = list(pi.keys())
            if not legal:
                break
            temp = 1.0 if move < temperature_moves else 0.3
            game_data.append((build_features(env), legal, pi_to_vec(pi, legal), env.currentTurn))
            action = select_action_from_pi(pi, temperature=temp)
        else:
            legal = env.getAvailableActions()
            if not legal:
                break
            action = teacher.choose_action(env)
            if action is None or action not in legal:
                break
            pv = np.zeros(len(legal), dtype=np.float32)
            pv[legal.index(action)] = 1.0
            game_data.append((build_features(env), legal, pv, env.currentTurn))
        env.step(action)
        move += 1
        if move > 120:
            break
    winner = env.winner
    samples = []
    for feat, legal, pivec, turn in game_data:
        z = 0.0 if (winner == 'draw' or winner is None) else (1.0 if winner == turn else -1.0)
        samples.append((feat, legal, pivec, z))
    return samples, winner, move


# ---------------------------------------------------------------------------
# CPU worker 进程
# ---------------------------------------------------------------------------
def worker_run(rank, initial_weights, num_simulations, temperature_moves,
               worker_device, sample_queue, control_queue, seed, teacher_depth=0, teacher_every=0):
    random.seed(seed + rank * 7919)
    np.random.seed(seed + rank * 7919)
    torch.manual_seed(seed + rank * 7919)
    try:
        net = ChemisNet().to(worker_device)
        if initial_weights is not None:
            net.load_state_dict({k: v.to(worker_device) for k, v in initial_weights.items()})
        mcts = MCTS(net, device=worker_device, num_simulations=num_simulations)
        local_count = 0
        while True:
            got_stop = False
            while True:
                try:
                    msg = control_queue.get_nowait()
                except Exception:
                    break
                if msg['type'] == 'stop':
                    got_stop = True
                elif msg['type'] == 'update':
                    net.load_state_dict({k: v.to(worker_device) for k, v in msg['weights'].items()})
            if got_stop:
                return
            # 交替：每 teacher_every 局派 1 局“ML(MCTS) vs 手写教师”，其余正常自对弈
            if teacher_every > 0 and teacher_depth > 0 and (local_count % teacher_every == teacher_every - 1):
                ml_color = 'white' if (local_count // teacher_every) % 2 == 0 else 'black'
                samples, winner, moves = self_play_vs_teacher(mcts, net, worker_device, teacher_depth, ml_color, temperature_moves)
            else:
                samples, winner, moves = self_play_game(mcts, net, worker_device, temperature_moves)
            sample_queue.put({'type': 'samples', 'samples': samples, 'winner': winner, 'moves': moves})
            local_count += 1
    except Exception as e:
        import traceback
        sample_queue.put({'type': 'error', 'msg': f'worker{rank}: {e}\n{traceback.format_exc()}'})


# ---------------------------------------------------------------------------
# worker 生命周期（供用户调用）
# ---------------------------------------------------------------------------
def create_worker_pool(initial_weights, workers, num_simulations, worker_device='cpu',
                       temperature_moves=30, seed=0, teacher_depth=0, teacher_every=0):
    """启动 CPU 推理 worker 进程池，返回 (procs, sample_queue, control_queue)。"""
    sample_queue = _CTX.Queue()
    control_queue = _CTX.Queue()
    procs = []
    for rank in range(workers):
        p = _CTX.Process(target=worker_run,
                         args=(rank, initial_weights, num_simulations, temperature_moves,
                               worker_device, sample_queue, control_queue, seed,
                               teacher_depth, teacher_every))
        p.start()
        procs.append(p)
    return procs, sample_queue, control_queue


def stop_workers(procs, control_queue):
    for p in procs:
        try:
            control_queue.put({'type': 'stop'})
        except Exception:
            pass
    for p in procs:
        p.join(timeout=20)
        if p.is_alive():
            p.terminate()


def broadcast_weights(control_queue, procs, net, src_device):
    sd = {k: v.detach().to('cpu') for k, v in net.state_dict().items()}
    for p in procs:
        try:
            control_queue.put({'type': 'update', 'weights': sd})
        except Exception:
            pass


def collect_data(buffer, sample_queue, target_games=1, timeout=10.0):
    """从 worker 队列收集新对局写入经验池。阻塞至凑够 target_games 或超时。

    返回 (collected_games, samples_added, winners_list)。"""
    collected, samples_added = 0, 0
    winners = []
    deadline = time.time() + timeout
    while collected < target_games and time.time() < deadline:
        try:
            msg = sample_queue.get(timeout=max(0.05, deadline - time.time()))
        except Exception:
            break
        if isinstance(msg, dict) and msg.get('type') == 'error':
            print('[WARN]', msg['msg'])
            continue
        if not (isinstance(msg, dict) and msg.get('type') == 'samples'):
            continue
        for feat, legal, pivec, z in msg['samples']:
            buffer.add(feat, legal, pivec, z)
        samples_added += len(msg['samples'])
        winners.append(msg['winner'])
        collected += 1
    return collected, samples_added, winners


# ---------------------------------------------------------------------------
# 可运行的主循环（演示如何按顺序调用上面的函数）
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Chemiss AlphaZero 异步混合(CPU推理+GPU训练)自我对弈')
    ap.add_argument('--num_games', type=int, default=2000)
    ap.add_argument('--num_simulations', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--games_per_update', type=int, default=50)
    ap.add_argument('--train_every_game', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--c_reg', type=float, default=1e-4)
    ap.add_argument('--buffer_size', type=int, default=300000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='GPU 训练设备')
    ap.add_argument('--worker_device', default='cpu', help='CPU 推理设备')
    ap.add_argument('--workers', type=int, default=0, help='并行 worker 数；0=auto(=核数-1, 上限6)')
    ap.add_argument('--temperature_moves', type=int, default=30)
    ap.add_argument('--teacher_depth', type=int, default=0, help='>0 时用“手写AI教师”对弈（与该深度 alpha-beta），产生高素质数据')
    ap.add_argument('--teacher_every', type=int, default=0, help='每 N 局派 1 局“ML vs 手写教师”；0=关闭')
    ap.add_argument('--save_every', type=int, default=25)
    ap.add_argument('--out', default='model_weights.json')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    workers = args.workers if args.workers > 0 else min(os.cpu_count() - 1, 6)
    print(f'GPU trainer = {device} | CPU workers = {workers} | games = {args.num_games} | sims = {args.num_simulations}')

    train_net = ChemisNet().to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        train_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    optimizer = torch.optim.Adam(train_net.parameters(), lr=args.learning_rate)
    buffer = ReplayBuffer(max_size=args.buffer_size)

    initial_weights = {k: v.detach().to('cpu') for k, v in train_net.state_dict().items()}
    procs, sample_q, control_q = create_worker_pool(initial_weights, workers, args.num_simulations,
                                                    args.worker_device, args.temperature_moves, args.seed,
                                                    args.teacher_depth, args.teacher_every)

    total_samples = 0
    wins = {'white': 0, 'black': 0, 'draw': 0}
    games_done = 0
    loss = 0.0; vl = 0.0; pl = 0.0
    t0 = time.time()

    def emit(line):
        """打印到 stdout 并写入 train.log（flush），保证后台也能实时看到进度。"""
        print(line, flush=True)
        try:
            with open('train.log', 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass

    while games_done < args.num_games:
        col, ns, wlist = collect_data(buffer, sample_q, target_games=1, timeout=10.0)
        if col > 0:
            games_done += col
            total_samples += ns
            for w in wlist:
                if w in wins:
                    wins[w] += 1
            if games_done % args.train_every_game == 0 and len(buffer) >= 32:
                loss, vl, pl = train_network(train_net, optimizer, buffer, args.batch_size, device, args.c_reg)
            if games_done % args.games_per_update == 0:
                broadcast_weights(control_q, procs, train_net, device)
            if games_done % args.save_every == 0 or games_done == args.num_games:
                save_model_weights(train_net, args.out)
                torch.save(train_net.state_dict(), args.out.replace('.json', '_model.pt'))
                emit(f'[save] games={games_done} 已写出 {args.out}')
            if games_done % 10 == 0 or games_done == args.num_games:
                el = time.time() - t0
                emit(f'[games {games_done}/{args.num_games}] samples={total_samples} buffer={len(buffer)} '
                     f'loss={loss:.4f}(v:{vl:.4f},p:{pl:.4f}) time={el:.0f}s elapsed/game={el/max(1,games_done):.1f}s')
        else:
            # 超时无新局：若 worker 全挂则提前收尾
            lives = [p.is_alive() for p in procs]
            if not any(lives):
                print('所有 worker 已退出，提前结束。')
                break

    stop_workers(procs, control_q)
    save_model_weights(train_net, args.out)
    torch.save(train_net.state_dict(), args.out.replace('.json', '_model.pt'))
    print('训练完成。权重已导出到:', args.out)
    print('胜负统计:', wins)


if __name__ == '__main__':
    main()
