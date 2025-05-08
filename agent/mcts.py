"""freckers_mcts.py – Re‑revised MCTS agent.

Major fixes after reviewing full game log:
• **No backward hops.** Any move that retreats gets a massive negative bias.
• Stronger lateral penalty (`W_LAT`  → 0.4).
• Grow moves allowed *only* when they unlock ≥ 3 extra hop moves – estimated
  cheaply via `state.hop_count()` helper (added below).
• Tree policy and roll‑out share identical forward‑only logic, ensuring
  simulations reflect our strategic intent.
• Losses back‑propagated correctly (kept from last patch).

Plug‑and‑play: same public interface.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from referee.game.actions import MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord

from .bitboard import BitBoard
from .strategy import Strategy

# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------


def _action_key(action: MoveAction, res: Optional[Coord]) -> Tuple:
    return ("G",) if res is None else (action.coord.r, action.coord.c, res.r, res.c)


def _row_prog(player: int, r0: int, r1: int) -> int:
    return (r1 - r0) if player == BitBoard.RED else (r0 - r1)


# quick hop‑count cache for grow heuristic ----------------------------------
BitBoard.hop_count = lambda self: len(self.get_all_moves()) - 1  # exclude grow

# ---------------------------------------------------------------------------
#  Node class
# ---------------------------------------------------------------------------


class MonteCarloTreeSearchNode(Strategy):
    # class‑level stats
    SIMS = 0
    AVG_LEN = 0.0

    # hyper‑params
    C = 0.5
    W_PROG = 0.4
    W_LAT = 0.4
    W_BACK = 5.0  # additional penalty for backwards moves
    GROW_REQ = 6  # min extra hops a grow must unlock
    PW_K = 4
    MAX_PLY = 150
    MAX_ROLLOUT = 70

    ROLLOUT_W_PROG = 15
    ROLLOUT_W_LAT = 4

    def __init__(
        self, state: BitBoard, parent=None, parent_action=None, *, time_budget=178.0
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: List["MonteCarloTreeSearchNode"] = []
        if parent is None:
            self.root_player = state.get_current_player()
            self.time_budget = time_budget
            self.depth = 0
        else:
            self.root_player = parent.root_player
            self.time_budget = parent.time_budget
            self.depth = parent.depth + 1

        self._visits = 0
        self._score = 0
        self._rave_v: Dict[Tuple, int] = defaultdict(int)
        self._rave_s: Dict[Tuple, int] = defaultdict(int)

        self._untried = self._ordered_moves()

    # ------------------------------------------------------------------
    def n(self):
        return self._visits

    def q(self):
        return self._score

    # ------------------------------------------------------------------
    #
    def _frog_spread(self):
        # for RED, larger r = closer to goal; for BLUE vice versa
        rows = [
            r for r, c in self.state.get_all_pos(self.state.get_current_player())
        ]  # all frog origins

        return max(rows) - min(rows)

    # def _ordered_moves(self) -> List[Tuple[MoveAction, Optional[Coord]]]:
    #     moves = self.state.get_all_moves()
    #     player = self.state.get_current_player()
    #     mid = (BOARD_N - 1) // 2
    #     base_hops = self.state.hop_count()
    #     base_spread = self._frog_spread()
    #
    #     scored: List[Tuple[MoveAction, Optional[Coord], float]] = []
    #     for mv, res in moves:
    #         if res is None:
    #             # your existing grow logic…
    #             score = 50 * (self.state.move(mv, res).hop_count() - base_hops)
    #         else:
    #             # forward / lateral as before
    #             prog = _row_prog(player, mv.coord.r, res.r)
    #             lat = abs(res.c - mid)
    #             if prog < 0:
    #                 score = -10_000 - self.W_BACK * abs(prog)
    #             else:
    #                 score = 1_000 * prog - 50 * lat
    #
    #             # now add cohesion bonus:
    #             # simulate the hop, compute new spread
    #             tmp = self.state.move(mv, res)
    #             tmp.toggle_player()
    #             new_spread = MonteCarloTreeSearchNode(tmp)._frog_spread()
    #             spread_delta = new_spread - base_spread
    #             cohesion_bonus = -200.0 * spread_delta
    #             score += cohesion_bonus
    #
    #         scored.append((mv, res, score))
    #
    #     # sort by the combined score
    #     scored.sort(key=lambda x: x[2], reverse=True)
    #     # drop the scores
    #     return [(mv, res) for mv, res, _ in scored]

    def _ordered_moves(self):
        moves = self.state.get_all_moves()
        player = self.state.get_current_player()
        mid = (BOARD_N - 1) // 2

        base_hops = len(moves) - 1  # exclude grow

        def score(item):
            mv, res = item
            if res is None:
                # quick grow benefit estimate
                after = len(self.state.move(mv, res).get_all_moves()) - 1
                gain = after - base_hops
                if gain < self.GROW_REQ:
                    return -10_000
                return 50 * gain
            prog = _row_prog(player, mv.coord.r, res.r)
            lat = abs(res.c - mid)
            if prog < 0:
                return -10_000 - self.W_BACK * abs(prog)  # forbid backward
            return 1_000 * prog - 50 * lat

        # always put grow first because it is a special case that has important consequences
        # temp = moves[:-1]
        # temp.sort(key=lambda x: score(x), reverse=True)
        moves.sort(key=lambda x: score(x), reverse=True)
        # moves = moves[-1:] + temp
        return moves

    # ------------------------------------------------------------------
    def _expand(self):
        mv, res = self._untried.pop(0)
        nxt = self.state.move(mv, res)
        nxt.toggle_player()
        child = MonteCarloTreeSearchNode(nxt, parent=self, parent_action=(mv, res))
        self.children.append(child)
        return child

    def _bias(self, mv, res):
        if res is None:
            tmp = self.state.move(mv, res)
            gain = -1 * (len(self.state.get_all_moves()) - len(tmp.get_all_moves()))

            return 0.02 * gain if gain and self.depth > 2 and self.depth < 40 else -2

        p = self.state.get_current_player()
        # get a cohesion bonus, you want frogs to be close together

        rows = [r for r, c in self.state.get_all_pos(p)]
        spread_old = max(rows) - min(rows)
        rows.remove(mv.coord.r)
        rows.append(res.r)
        spread_new = max(rows) - min(rows)

        # we want a negative spread delta
        spread_delta = spread_new - spread_old

        prog = _row_prog(p, mv.coord.r, res.r)
        lat = abs(res.c - (BOARD_N - 1) // 2)
        if prog < 0:
            return -self.W_BACK * abs(prog)

        return self.W_PROG * prog - self.W_LAT * lat - 0.2 * spread_delta

    def _uct(self, child):
        exploit = child.q() / (child.n() + 1e-9)
        bias = self._bias(*child.parent_action)
        explore = self.C * math.sqrt(math.log(self.n() + 1) / (child.n() + 1e-9))
        return exploit + bias + explore

    def _select(self):
        return max(self.children, key=self._uct)

    def _tree_policy(self):
        node = self
        while not node.state.is_game_over():
            if node._untried and node.n() >= len(node.children) * self.PW_K:
                return node._expand()
            if not node.children:
                return node._expand()
            node = node._select()
        return node

    # ------------------------------------------------------------------
    @staticmethod
    def _rollout_move(state: BitBoard):
        return state.get_random_move()
        moves = state.get_all_moves()
        player = state.get_current_player()
        mid = (BOARD_N - 1) // 2
        best_val, best = -1e9, None
        for mv, res in moves:
            if res is None:
                continue  # never grow in rollout
            prog = _row_prog(player, mv.coord.r, res.r)
            if prog < 0:
                continue  # skip backwards
            lat = abs(res.c - mid)
            val = (
                MonteCarloTreeSearchNode.ROLLOUT_W_PROG * prog
                - MonteCarloTreeSearchNode.ROLLOUT_W_LAT * lat
            )
            if val > best_val:
                best_val, best = val, (mv, res)
        return best if best else random.choice(moves)  # fallback

    def _simulate(self):
        bb = BitBoard(np.copy(self.state.get_board()))
        bb.current_player = self.state.current_player
        amaf = set()

        depth = 0
        for depth in range(self.MAX_ROLLOUT):
            if bb.is_game_over():
                break

            mv, res = self._rollout_move(bb)
            if res is not None and bb.get_current_player() == self.root_player:
                amaf.add(_action_key(mv, res))
            bb = bb.move(mv, res, in_place=True)

            bb.toggle_player()

        # evaluate
        if bb.is_game_over():
            f, o = bb.frog_border_count.values()
            w = None
            if f == BOARD_N - 2 and o != BOARD_N - 2:
                w = BitBoard.RED
            elif o == BOARD_N - 2 and f != BOARD_N - 2:
                w = BitBoard.BLUE
            res = 0 if w is None else (1 if w == self.root_player else -1)
        else:
            s = bb.evaluate_position()
            good = (s > 0) == (bb.get_current_player() == self.root_player)
            res = 1 if good else -1
        MonteCarloTreeSearchNode.SIMS += 1
        MonteCarloTreeSearchNode.AVG_LEN += (
            depth - MonteCarloTreeSearchNode.AVG_LEN
        ) / MonteCarloTreeSearchNode.SIMS
        return res, list(amaf)

    # ------------------------------------------------------------------
    def _backprop(self, result, amaf):
        self._visits += 1
        self._score += result
        for k in amaf:
            self._rave_v[k] += 1
            self._rave_s[k] += result
        if self.parent:
            self.parent._backprop(-result, amaf)

    # ------------------------------------------------------------------
    def best_action(self, *, safety_margin=0.5, decay=0.975):
        t0 = time.perf_counter()
        rem = self.time_budget - safety_margin
        assert rem > 0
        move_no = min(self.state.get_ply_count() // 2, self.MAX_PLY // 2)
        geo = decay**move_no - decay ** (self.MAX_PLY // 2)
        slice_t = rem * (1 - decay) * decay**move_no / geo
        deadline = t0 + slice_t
        sims = 0

        while time.perf_counter() < deadline:
            leaf = self._tree_policy()
            res, amaf = leaf._simulate()
            leaf._backprop(res, amaf)
            sims += 1
        if sims == 0 and self._untried:
            leaf = self._expand()
            res, amaf = leaf._simulate()
            leaf._backprop(res, amaf)
            sims = 1
        best = max(self.children, key=lambda c: (c.n(), c.q() / (c.n() + 1e-9)))
        print(f"[MCTS] sims={sims}  avg_len={self.AVG_LEN:0.1f}")
        return {
            "action": best.parent_action[0],
            "res": best.parent_action[1],
            "res_node": best,
        }

    def find_child(self, action: MoveAction):
        for ch in self.children:
            if ch.parent_action[0] == action:
                return ch
        return None
