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
    C = 0.2  # Reduced exploration for more exploitation
    W_PROG = 4.0  # Extremely high weight for forward progress
    W_LAT = 0.1   # Very low weight for lateral movement
    W_BACK = 20.0  # Extremely high penalty for backwards moves
    GROW_REQ = 2  # Higher grow requirement to reduce unnecessary grows
    PW_K = 2      # Focused search
    MAX_PLY = 150
    MAX_ROLLOUT =  60 # Even shorter rollouts for faster iterations

    ROLLOUT_W_PROG = 40  # Extremely high weight for forward progress in rollouts
    ROLLOUT_W_LAT = 1    # Very low weight for lateral movement in rollouts

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

    def _ordered_moves(self):
        moves = self.state.get_all_moves()
        player = self.state.get_current_player()
        mid = (BOARD_N - 1) // 2
        base_hops = len(moves) - 1  # exclude grow

        def score(item):
            mv, res = item
            if res is None:
                # Conservative grow evaluation
                after = len(self.state.move(mv, res).get_all_moves()) - 1
                gain = after - base_hops
                if gain < self.GROW_REQ:
                    return -10_000
                
                # Only consider grows in very early game
                game_phase = self.state.get_ply_count() / 150
                if game_phase > 0.1:  # After 10% of game, grows are less valuable
                    return -5_000
                
                # Consider board state
                lily_count = bin(self.state.lilly_bits).count("1")
                board_density = lily_count / (BOARD_N * BOARD_N)
                
                # More valuable when board is very sparse
                return 100 * gain * (1 + 2 * board_density)

            prog = _row_prog(player, mv.coord.r, res.r)
            lat = abs(res.c - mid)
            
            if prog < 0:
                return -10_000 - self.W_BACK * abs(prog)  # forbid backward
            
            # Enhanced move scoring
            num_hops = len(mv.directions)
            jump_bonus = 1000 * (num_hops - 1)  # Extremely strong multi-jump bonus
            
            # Forward progress with minimal diminishing returns
            prog_score = 5000 * prog * (1 - 0.02 * (prog / BOARD_N))
            
            # Lateral movement penalty
            lat_penalty = 20 * lat  # Very low lateral penalty
            
            # Position bonus
            position_bonus = 0
            game_phase = self.state.get_ply_count() / 150
            if game_phase < 0.2:  # Early game
                player_pos = self.state.get_all_pos(player)
                if player_pos:
                    avg_r = sum(r for r, _ in player_pos) / len(player_pos)
                    r_diff = abs(res.r - avg_r)
                    position_bonus = -100 * r_diff  # Moderate penalty for spreading pieces
            
            return prog_score + jump_bonus - lat_penalty + position_bonus

        # Sort moves by score
        moves.sort(key=score, reverse=True)
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
            
            # Conservative grow bias
            game_phase = self.state.get_ply_count() / 150
            if game_phase > 0.1:  # After 10% of game, grows are less valuable
                return -5
            
            return 0.3 * gain if gain and self.depth > 2 and self.depth < 45 else 0.2 * gain

        p = self.state.get_current_player()

        prog = _row_prog(p, mv.coord.r, res.r)
        lat = abs(res.c - (BOARD_N - 1) // 2)
        if prog < 0:
            return -self.W_BACK * abs(prog)

        # Enhanced multi-jump bonus
        num_hops = len(mv.directions)
        jump_bonus = 2.0 * (num_hops - 1)  # Extremely strong multi-jump bonus

        # Position bonus
        position_bonus = 0
        game_phase = self.state.get_ply_count() / 150
        if game_phase < 0.2:  # Early game
            player_pos = self.state.get_all_pos(p)
            if player_pos:
                avg_r = sum(r for r, _ in player_pos) / len(player_pos)
                r_diff = abs(res.r - avg_r)
                position_bonus = -1.0 * r_diff  # Moderate penalty for spreading pieces

        return self.W_PROG * prog - self.W_LAT * lat + jump_bonus + position_bonus

    def _uct(self, child):
        exploit = child.q() / (child.n() + 1e-9)
        bias = self._bias(*child.parent_action)

        explore = self.C * math.sqrt(math.log(self.n() + 1) / (child.n() + 1e-9))

        # Progressive bias that decreases with visits
        prog_bias = bias / (np.log(1 + child.n()) + 1)

        # Add RAVE bonus for move actions
        if child.parent_action[1] is not None:  # Only for move actions
            key = _action_key(*child.parent_action)
            if self._rave_v[key] > 0:
                rave_score = self._rave_s[key] / self._rave_v[key]
                beta = np.sqrt(self.PW_K / (3 * self.n() + self.PW_K))
                prog_bias = (1 - beta) * prog_bias + beta * rave_score

        return exploit + prog_bias + explore

    def old_uct(self, child):
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
        moves = state.get_all_moves()
        player = state.get_current_player()
        mid = (BOARD_N - 1) // 2

        # Epsilon-greedy approach with dynamic epsilon
        game_phase = state.get_ply_count() / 150
        epsilon = 0.05 * (1 - game_phase)  # Less random in endgame
        if random.random() < epsilon:
            return state.get_random_move()

        best_val, best = -1e9, None
        for mv, res in moves:
            if res is None:
                # Very conservative grow consideration
                if random.random() < 0.05 * (1 - game_phase) and state.hop_count() < 6:
                    val = 10  # Moderate grow value
                    if val > best_val:
                        best_val, best = val, (mv, res)
                continue

            prog = _row_prog(player, mv.coord.r, res.r)
            if prog < 0:
                continue  # Skip backwards
            lat = abs(res.c - mid)
            num_hops = len(mv.directions)
            
            # Enhanced rollout evaluation
            val = (
                MonteCarloTreeSearchNode.ROLLOUT_W_PROG * prog
                - MonteCarloTreeSearchNode.ROLLOUT_W_LAT * lat
                + 20 * (num_hops - 1)  # Strong multi-jump bonus
            )
            
            if val > best_val:
                best_val, best = val, (mv, res)

        return best if best else random.choice(moves)

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
