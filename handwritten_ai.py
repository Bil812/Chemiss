# -*- coding: utf-8 -*-
"""
handwritten_ai.py
=================
把 HTML 里的「手写 AI」移植到 Python，用于在训练中让 ML 跟它对弈，生成高质量对局数据。

包含：
  - evaluate(color)：15 项手工评估（子力/位置/机动性/键合/电负性/电子经济/氢王安全/
    氢王威胁/推进/电荷/眩晕/放射性/键合完整度/离子威胁/残局/给电子潜力）—— 逐条对齐 HTML。
  - negamax：带 alpha-beta 剪枝、基本走法排序的搜索，选最佳动作。
  - choose_action(env, depth)：返回动作索引（与 chemis_env 的动作空间一致）。

用法：teacher = HandwrittenAI(depth=3)
      action = teacher.choose_action(env)   # env 为 ChemisEnv（当前执子方）
"""

import math
import time
import numpy as np

from chemis_env import ChemisEnv, COLOR_WHITE, COLOR_BLACK, METAL, NONMETAL
from chemis_env import GIVE_ELECTRON_OFFSET, NUCLEAR_OFFSET, NUCLEAR_TARGETS, NUCLEAR_TARGET_IDX
from chemis_env import build_features


class HandwrittenAI:
    def __init__(self, depth=3, color=None):
        self.depth = depth
        self.color = color   # 若固定，则只在该方执子时用；否则用 env.currentTurn
        self._deadline = None   # epoch ms；为 None 则不限时

    # ------------------------------------------------------------------
    # 子力价值
    # ------------------------------------------------------------------
    def _piece_value(self, p):
        if p.row < 0:
            return 0
        v = p.Z * 2
        if p.isHydrogenKing:
            v = 1200
        if p.noble:
            v = 25
        if p.totalElectrons > 8:
            v += (p.totalElectrons - 8) ** 1.5 * 3
        if p.radioactive:
            v -= 10
        if p.stunned:
            v *= 0.25
        if getattr(p, 'lanthanide', False):
            v -= 2
        return v

    def _is_checking(self, env, piece, oppH):
        return env._canCaptureHFrom(piece, piece.row, piece.col, oppH)

    def _is_checkmate_level(self, env, piece, row, col, oppH):
        if not env._canCaptureHFrom(piece, row, col, oppH):
            return False
        return not env._canBeCapturedByOpponent(piece, row, col)

    # ------------------------------------------------------------------
    # 15 项手工评估（from 化学棋Chemiss.html ChemissAI.evaluate）
    # 返回 color 视角的分数。
    # ------------------------------------------------------------------
    def evaluate(self, env, color):
        myPieces = env.whitePieces if color == COLOR_WHITE else env.blackPieces
        oppPieces = env.blackPieces if color == COLOR_WHITE else env.whitePieces
        if env.gameOver:
            if env.winner == color:
                return 100000000
            if env.winner == 'draw':
                return 0
            return -100000000
        score = 0.0
        myMat = sum(self._piece_value(p) for p in myPieces)
        oppMat = sum(self._piece_value(p) for p in oppPieces)
        score += myMat - oppMat

        myH = next((p for p in myPieces if p.isHydrogenKing), None)
        oppH = next((p for p in oppPieces if p.isHydrogenKing), None)

        # 2. POSITION
        for p in myPieces:
            if p.row < 0 or p.col < 0:
                continue
            centerDist = abs(p.col - 3.5) + abs(p.row - 3.5)
            psqt = max(0, 4 - centerDist)
            score += psqt * (1.5 + (0.5 if p.Z > 20 else 0))
            if (p.row <= 1 or p.row >= 6) and (p.col <= 1 or p.col >= 6):
                score -= 1
        for p in oppPieces:
            if p.row < 0 or p.col < 0:
                continue
            centerDist = abs(p.col - 3.5) + abs(p.row - 3.5)
            psqt = max(0, 4 - centerDist)
            score -= psqt * (1.5 + (0.5 if p.Z > 20 else 0))
            if (p.row <= 1 or p.row >= 6) and (p.col <= 1 or p.col >= 6):
                score += 1

        # 3. MOBILITY
        saved = env.currentTurn
        env.currentTurn = color
        myMobility = 0
        for p in myPieces:
            if p.row < 0 or p.col < 0 or p.stunned or p.totalElectrons <= 0:
                continue
            moves = env.getLegalMoves(p)
            myMobility += len(moves)
            for m in moves:
                if m['isAttack']:
                    target = env.board[m['row']][m['col']]
                    if target:
                        score += self._piece_value(target) * 0.15
        score += myMobility * 0.55
        env.currentTurn = COLOR_BLACK if color == COLOR_WHITE else COLOR_WHITE
        oppMobility = 0
        for p in oppPieces:
            if p.row < 0 or p.col < 0 or p.stunned or p.totalElectrons <= 0:
                continue
            oppMobility += len(env.getLegalMoves(p))
        score -= oppMobility * 0.55
        env.currentTurn = saved

        # 4. BOND NETWORK
        myBonded = set(); oppBonded = set()
        for p in myPieces:
            if p.bondedWith and len(p.bondedWith) > 0:
                myBonded.add(p.id)
                score += 5.5
                if any((b['p1'] == p.id or b['p2'] == p.id) and b['type'] == 'metallic' for b in env.bonds):
                    score += 2
        for p in oppPieces:
            if p.bondedWith and len(p.bondedWith) > 0:
                oppBonded.add(p.id)
                score -= 5.5
                if any((b['p1'] == p.id or b['p2'] == p.id) and b['type'] == 'metallic' for b in env.bonds):
                    score -= 2
        if len(myBonded) > len(oppBonded):
            score += 3
        elif len(oppBonded) > len(myBonded):
            score -= 3

        # 5. ELECTRONEGATIVITY
        myActive = [p for p in myPieces if p.row >= 0 and p.col >= 0 and not p.noble]
        oppActive = [p for p in oppPieces if p.row >= 0 and p.col >= 0 and not p.noble]
        myAvgEN = sum(p.electroneg for p in myActive) / len(myActive) if myActive else 0
        oppAvgEN = sum(p.electroneg for p in oppActive) / len(oppActive) if oppActive else 0
        score += (myAvgEN - oppAvgEN) * 2

        # 6. ELECTRON ECONOMY
        myE = myDirs = mySteps = 0
        oppE = oppDirs = oppSteps = 0
        for p in myPieces:
            if p.row < 0:
                continue
            myE += p.totalElectrons; myDirs += min(p.totalElectrons, 8); mySteps += p.totalElectrons
        for p in oppPieces:
            if p.row < 0:
                continue
            oppE += p.totalElectrons; oppDirs += min(p.totalElectrons, 8); oppSteps += p.totalElectrons
        score += (myE - oppE) * 1.5 + (myDirs - oppDirs) * 2 + (mySteps - oppSteps) * 1

        # 7. H SAFETY
        if myH and myH.row >= 0:
            n = env.getAdjacentPieces(myH.row, myH.col)
            friends = sum(1 for x in n if x.color == color)
            foes = sum(1 for x in n if x.color != color)
            score += friends * 12; score -= foes * 25
            if myMat < oppMat:
                score += friends * 5; score -= foes * 10
            if friends == 0 and foes > 0:
                score -= 30
        if oppH and oppH.row >= 0:
            n = env.getAdjacentPieces(oppH.row, oppH.col)
            oppColor = COLOR_BLACK if color == COLOR_WHITE else COLOR_WHITE
            oppFriends = sum(1 for x in n if x.color == oppColor)
            oppFoes = sum(1 for x in n if x.color == color)
            score -= oppFriends * 12; score += oppFoes * 25
            if oppMat < myMat:
                score -= oppFriends * 5; score += oppFoes * 10
            if oppFriends == 0 and oppFoes > 0:
                score += 30

        # 8. H THREAT
        for p in myPieces:
            if p.row < 0 or p.col < 0 or not oppH or oppH.row < 0:
                continue
            distToHK = abs(p.row - oppH.row) + abs(p.col - oppH.col)
            if self._is_checking(env, p, oppH):
                if self._is_checkmate_level(env, p, p.row, p.col, oppH):
                    score += 2000
                else:
                    score += 600
            elif distToHK <= 1:
                score += 30
            elif distToHK <= 3:
                score += (4 - distToHK) * 8
        for p in oppPieces:
            if p.row < 0 or p.col < 0 or not myH or myH.row < 0:
                continue
            distToHK = abs(p.row - myH.row) + abs(p.col - myH.col)
            if self._is_checking(env, p, myH):
                if self._is_checkmate_level(env, p, p.row, p.col, myH):
                    score -= 2000
                else:
                    score -= 600
            elif distToHK <= 1:
                score -= 30
            elif distToHK <= 3:
                score -= (4 - distToHK) * 8

        # 9. ADVANCEMENT
        for p in myPieces:
            if p.row < 0 or p.col < 0 or p.isHydrogenKing or p.noble:
                continue
            adv = (7 - p.row) if color == COLOR_WHITE else p.row
            score += adv ** 1.3 * 1.2
        for p in oppPieces:
            if p.row < 0 or p.col < 0 or p.isHydrogenKing or p.noble:
                continue
            adv = p.row if color == COLOR_WHITE else (7 - p.row)
            score -= adv ** 1.3 * 1.2

        # 10. CHARGE
        for p in myPieces:
            if p.row < 0:
                continue
            if p.charge > 0:
                score += 3
            if p.charge < 0:
                score += 3
        for p in oppPieces:
            if p.row < 0:
                continue
            if p.charge > 0:
                score -= 3
            if p.charge < 0:
                score -= 3

        # 11. STUN
        for p in myPieces:
            if p.stunned and p.row >= 0:
                score -= 10
        for p in oppPieces:
            if p.stunned and p.row >= 0:
                score += 10

        # 12. RADIOACTIVE
        oppColor = COLOR_BLACK if color == COLOR_WHITE else COLOR_WHITE
        for p in myPieces:
            if p.radioactive and p.row >= 0:
                n = env.getAdjacentPieces(p.row, p.col)
                if any(x.color != color for x in n):
                    score += 3
                if any(x.color == color for x in n):
                    score -= 2
        for p in oppPieces:
            if p.radioactive and p.row >= 0:
                n = env.getAdjacentPieces(p.row, p.col)
                if any(x.color == color for x in n):
                    score -= 3
                if any(x.color == oppColor for x in n):
                    score += 2

        # 13. BOND COMPLETENESS
        for p in myPieces:
            if p.row < 0 or p.col < 0:
                continue
            octet = 2 if p.isHydrogenKing else 8
            cur = p.valence + (p.sharedElectrons or 0) + (p.bonusElectrons or 0)
            completion = min(cur / octet, 1.5)
            score += completion * 6.5
            if p.totalElectrons > octet + 2:
                score -= (p.totalElectrons - octet - 2) * 3
        for p in oppPieces:
            if p.row < 0 or p.col < 0:
                continue
            octet = 2 if p.isHydrogenKing else 8
            cur = p.valence + (p.sharedElectrons or 0) + (p.bonusElectrons or 0)
            completion = min(cur / octet, 1.5)
            score -= completion * 6.5
            if p.totalElectrons > octet + 2:
                score += (p.totalElectrons - octet - 2) * 3

        # 14. IONIC THREAT
        for p in myPieces:
            if p.row < 0 or p.col < 0:
                continue
            for n in env.getAdjacentPieces(p.row, p.col):
                if n.color == color:
                    continue
                if (p.group == METAL and n.group == NONMETAL) or (p.group == NONMETAL and n.group == METAL):
                    metal = p if p.group == METAL else n
                    nonmetal = p if p.group == NONMETAL else n
                    if env.judgeBondType(metal, nonmetal) == 'ionic':
                        score += self._piece_value(n) * 0.18
        for p in oppPieces:
            if p.row < 0 or p.col < 0:
                continue
            for n in env.getAdjacentPieces(p.row, p.col):
                if n.color == color:
                    continue
                if (p.group == METAL and n.group == NONMETAL) or (p.group == NONMETAL and n.group == METAL):
                    metal = p if p.group == METAL else n
                    nonmetal = p if p.group == NONMETAL else n
                    if env.judgeBondType(metal, nonmetal) == 'ionic':
                        score -= self._piece_value(n) * 0.18

        # 15. ENDGAME
        totalPieces = sum(1 for p in myPieces if p.row >= 0) + sum(1 for p in oppPieces if p.row >= 0)
        if totalPieces < 16:
            if oppH and oppH.row >= 0:
                foesNearOppH = sum(1 for x in env.getAdjacentPieces(oppH.row, oppH.col) if x.color == color)
                score += foesNearOppH * 12
            if myH and myH.row >= 0:
                foesNearMyH = sum(1 for x in env.getAdjacentPieces(myH.row, myH.col) if x.color != color)
                score -= foesNearMyH * 12

        # 16. GIVE ELECTRON POTENTIAL
        for p in myPieces:
            if p.row < 0 or p.group != METAL or p.totalElectrons <= 0:
                continue
            canGive = any(n.color == color and n.group == NONMETAL and not n.noble
                          and n.totalElectrons < (2 if n.isHydrogenKing else 8)
                          for n in env.getAdjacentPieces(p.row, p.col))
            if canGive:
                score += 5
        for p in oppPieces:
            if p.row < 0 or p.group != METAL or p.totalElectrons <= 0:
                continue
            canGive = any(n.color == oppColor and n.group == NONMETAL and not n.noble
                          and n.totalElectrons < (2 if n.isHydrogenKing else 8)
                          for n in env.getAdjacentPieces(p.row, p.col))
            if canGive:
                score -= 5

        return round(score)

    # ------------------------------------------------------------------
    # 走法生成（动作索引；跳 goH 过滤加速——评估自带氢王安全惩罚）
    # ------------------------------------------------------------------
    def _actions(self, env):
        env._ai_searching = True
        acts = env.getAvailableActions()
        return acts

    def _terminal_value(self, env):
        if env.winner == 'draw' or env.winner is None:
            return 0
        return 1 if env.winner == env.currentTurn else -1

    # ------------------------------------------------------------------
    # Negamax + alpha-beta
    # ------------------------------------------------------------------
    def _negamax(self, env, depth, alpha, beta, color):
        if self._deadline is not None and time.time() * 1000 > self._deadline:
            return 0
        if env.gameOver:
            return self._terminal_value(env)
        if depth == 0:
            return self.evaluate(env, color) / 100000.0   # 归一到 [-1,1] 量级便于剪枝
        acts = self._actions(env)
        if not acts:
            return 0
        best = -1e18
        # 基本走法排序：吃子/吃氢王优先
        def order_score(a):
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
        acts = sorted(acts, key=order_score, reverse=True)
        for a in acts:
            child = env.clone()
            child._ai_searching = True
            child.step(a)
            val = -self._negamax(child, depth - 1, -beta, -alpha, child.currentTurn)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def choose_action(self, env, time_budget_ms=None):
        self.env = env
        if time_budget_ms is not None:
            self._deadline = time.time() * 1000 + time_budget_ms
        else:
            self._deadline = None
        color = env.currentTurn
        act0 = self._actions(env)
        if not act0:
            return None
        best = None; bestVal = -1e18
        alpha = -1e18
        for a in sorted(act0, key=lambda a: self._order_score(env, a), reverse=True):
            if self._deadline is not None and time.time() * 1000 > self._deadline:
                break
            child = env.clone()
            child._ai_searching = True
            child.step(a)
            val = -self._negamax(child, self.depth - 1, -1e18, -alpha, child.currentTurn)
            if val > bestVal:
                bestVal = val; best = a
            if val > alpha:
                alpha = val
        return best

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
