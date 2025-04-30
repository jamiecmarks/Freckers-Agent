import random
import numpy as np
from collections import defaultdict
from .bitboard import BitBoard
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord
from .strategy import Strategy
import time


class MonteCarloTreeSearchNode(Strategy):
    avg_playout_depth = 0.0
    playouts_done = 0

    def __init__(
        self, state: BitBoard, parent=None, parent_action=None, time_budget=178.0
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        if parent is None:
            # only the root records who “we” are
            self.root_player = state.get_current_player()
            self.time_budget = time_budget
        else:
            # children inherit the real root’s identity & remaining clock
            self.root_player = parent.root_player
            self.time_budget = parent.time_budget

        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._total_reward = 0
        self._untried_actions = self.untried_actions()
        self.c = 0.5

        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def untried_actions(self):
        blocked = False
        for row in self.state.get_board():
            if np.sum(row) == 0:  # there is a 'block'
                blocked = True
        if not blocked:
            opt = self.state.get_all_optimal_moves()
            # take the top 2–3 optimal, plus 30% of the rest at random
            extra = random.sample(self.state.get_all_moves(), k=int(0.3 * len(opt)))
            return opt + extra
        return self.state.get_all_moves()

    def q(self):
        # return self._total_reward
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        # sort by heuristic priority once
        if np.random.rand() < 0.1:
            # 10% chance to expand a random untried action
            idx = random.randrange(len(self._untried_actions))
        else:
            # otherwise take the highest‐priority one
            self._untried_actions.sort(
                key=lambda mv: self.state._move_priority(mv), reverse=True
            )
            idx = 0

        action_res = self._untried_actions.pop(idx)
        action, res = action_res
        next_state = self.state.move(action, res)
        next_state.toggle_player()

        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action_res
        )

        if next_state.is_game_over():
            child_node._number_of_visits += 10
            child_node._results[1] += 10

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def fully_expand(self):
        while not self.is_fully_expanded():
            self.expand()

    def new_rollout_policy(self, state, depth=0):
        """
        epsilon-greedy + softmax-weighted heuristic playout policy.
        """
        moves = state.get_all_moves()
        num_moves = len(moves)
        # if num_moves == 0:
        #     return None

        # 1) ε-greedy: 10% of the time, explore uniformly at random
        eps = 0.1 if depth > 15 else 0.1
        if np.random.rand() < eps:
            return moves[np.random.randint(num_moves)]

        # 2) Otherwise, score each move by your heuristic
        mid_col = (BOARD_N - 1) // 2
        scores = []
        for action, res in moves:
            score = 0.0
            if isinstance(action, GrowAction):
                # early-game grow bonus, late-game penalty
                ratio = len(state.move(action, None).get_all_moves()) / len(
                    state.get_all_moves()
                )  # the increase in moves that we get if we were to grow now
                # score += +0.5 if (2 < depth and depth < 15) else 0.4
                score += ratio * 0.8
            else:
                # reward long jumps
                dist = abs(res.r - action.coord.r)
                score += dist if dist > 0 else -0.5
                # centering bonus early
                if depth < 15:
                    start_c, end_c = action.coord.c, res.c
                    if abs(start_c - mid_col) > abs(end_c - mid_col):
                        score += 0.1
            scores.append(score)

        # 3) Softmax to get a probability distribution
        x = np.array(scores, dtype=np.float64)
        # numerical stabilization
        x = x - np.max(x)
        exp_x = np.exp(x / 1.0)  # temperature = 1.0; tune if you like
        probs = exp_x / exp_x.sum()

        # 4) Sample one move according to that distribution
        choice_idx = np.random.choice(num_moves, p=probs)
        return moves[choice_idx]

    def rollout_policy(self, state, depth=0):
        weights = []

        best_score = -float("inf")
        best_move = None
        mid_col = (BOARD_N - 1) // 2
        curr_player = state.get_current_player()
        scores = []
        moves = state.get_all_moves()

        for action, res in moves:
            score = 0
            if isinstance(action, GrowAction):
                score += 0.4 if (depth < 15 and depth > 2) else -0.2
            else:
                # e.g. reward long jumps, centering, etc.

                vert_dist = abs(res.r - action.coord.r)
                # 2b) multi-jump bonus
                #
                if vert_dist < 1:
                    score -= 0.9
                else:
                    score += 1 * vert_dist

                # 2c) centering bonus
                start_c, end_c = action.coord.c, res.c

                if abs(start_c - mid_col) > abs(end_c - mid_col) and depth < 15:
                    score += 0.1

            scores.append(score)
            if score > best_score:
                best_score = score
                best_move = (action, res)

        raw = scores
        offset = -min(raw) if min(raw) < 0 else 0
        adjusted = [s + offset for s in raw]
        total = sum(adjusted)
        if total == 0:
            probs = [1 / len(raw)] * len(raw)
        else:
            probs = [a / total for a in adjusted]

        return random.choices(moves, probs)[0]

    # new version of rollout
    def simulate_playout(self):
        state = self.state
        depth = 0
        max_depth = 150 - self.state.get_ply_count()

        while not state.is_game_over() and depth < max_depth:
            if False:
                # using board eval
                next_states = {}
                for action, res in state.get_all_moves():
                    next_state = state.move(action, res)
                    next_state.toggle_player()
                    next_states[next_state] = next_state.evaluate_position()

                raw = next_states.values()
                offset = -min(raw) if min(raw) < 0 else 0
                adjusted = [s + offset for s in raw]
                total = sum(adjusted)
                if total == 0:
                    probs = [1 / len(raw)] * len(raw)
                else:
                    probs = [a / total for a in adjusted]

                state = random.choices(list(next_states.keys()), weights=probs)[0]

            if True:
                # using rollout policy
                action, res = self.rollout_policy(state, depth=depth)
                state = state.move(action, res)
                state.toggle_player()

            # state.toggle_player()
            # action, res = self.rollout_policy(state, depth=self.depth + depth // 2)
            # state = state.move(action, res)
            # state.toggle_player()
            depth += 1

        MonteCarloTreeSearchNode.playouts_done += 1
        # incremental average
        MonteCarloTreeSearchNode.avg_playout_depth += (
            depth - MonteCarloTreeSearchNode.avg_playout_depth
        ) / MonteCarloTreeSearchNode.playouts_done

        if state.is_game_over():
            # compute which side actually reached the goal
            frog_count, opp_count = (
                state.frog_border_count[BitBoard.FROG],
                state.frog_border_count[BitBoard.OPPONENT],
            )
            if frog_count == BOARD_N - 2 and opp_count != BOARD_N - 2:
                winner_piece = BitBoard.FROG
            elif opp_count == BOARD_N - 2 and frog_count != BOARD_N - 2:
                winner_piece = BitBoard.OPPONENT
            else:
                return 0  # draw
            return +1 if winner_piece == self.root_player else -1

        if not state.is_game_over():
            return +1 if self.heuristic_score(state) > 0 else -1
        return 0  # if reached max depth -> draw

    def heuristic_score(self, state):
        score = 0
        board = state.get_board()
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == self.state.get_current_player():
                    match self.state.get_current_player():
                        case BitBoard.FROG:
                            score += r
                            break
                        case BitBoard.OPPONENT:
                            score += BOARD_N - 1 - r
                            break
                elif board[r][c] not in (BitBoard.LILLY, BitBoard.EMPTY):
                    match self.state.get_current_player():
                        case BitBoard.FROG:
                            score -= BOARD_N - 1 - r
                            break
                        case BitBoard.OPPONENT:
                            score -= r
                            break
                    score -= (
                        BOARD_N - 1 - r
                    )  # still an error here, not calculated coorectl
        return score

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        # self._total_reward += result

        if self.parent:
            self.parent.backpropagate(-result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def dynamic_c(self):
        # start exploratory, then exploit more later
        return self.c / (1 + np.log(1 + self.n()))

    def new_tree_policy(self):
        widen_k = 3
        node = self
        while not node.is_terminal_node():
            # if no child yet, always expand first
            if len(node.children) == 0:
                return node.expand()

            # if we still have untried actions AND
            # we've visited this node enough times, expand
            if not node.is_fully_expanded() and node.n() > widen_k * len(node.children):
                return node.expand()

            # otherwise, pick the best child by UCB
            node = node.UCB_choose()

        return node

    def _tree_policy(self):
        node = self
        # descend until we find a node we can expand or a terminal
        while not node.is_terminal_node():
            if not node.is_fully_expanded():
                return node.expand()  # expand one child and return it
            else:
                node = node.UCB_choose()
        return node

    def UCB_choose(self):
        # return self.best_child()
        for child in self.children:
            if child.n() == 0:
                return child

        # try to add move_prio to UCB1
        scores = []
        for c in self.children:
            value = c.q() / (c.n() + 1e-5)
            exploration = self.dynamic_c() * np.sqrt(
                (2 * np.log(self.n() + 1) / (c.n() + 1e-5))
            )
            # priority_bonus = self.state._move_priority(c.parent_action)
            scores.append(value + exploration)  # tune this
        return self.children[np.argmax(scores)]

    def choose_next_action(self):
        best = max(self.children, key=lambda c: c.n())
        return best

    def best_action(self, safety_margin: float = 0.05, beta: float = 0.75):
        # how many plies have actually been played?
        moves_played = self.state.get_ply_count()
        # predict total plies from our running average, or fall back
        total_pred = (
            type(self).avg_playout_depth if type(self).playouts_done > 0 else 150.0
        )
        moves_left = max(1, int(total_pred) - moves_played)

        # compute your “ideal” alloc_time via power‐law
        raw_alloc = (self.time_budget - safety_margin) / (moves_left**beta)

        # 1) clamp to at most the actual remaining budget minus margin
        per_move_alloc = min(raw_alloc, self.time_budget - safety_margin)

        start = time.perf_counter()
        deadline = start + per_move_alloc
        sims = 0

        # 2) in‐loop guard: stop if we've used up the real budget
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            # also ensure we never run past the *global* budget
            if (now - start) >= (self.time_budget - safety_margin):
                break

            leaf = self.new_tree_policy()
            reward = leaf.simulate_playout()
            leaf.backpropagate(reward)
            sims += 1

        # how long we actually spent
        elapsed = time.perf_counter() - start
        # update the *global* remaining time budget
        self.time_budget = max(0.0, self.time_budget - elapsed)

        # pick best child and carry forward the leftover budget
        best_child = max(self.children, key=lambda c: c.n())
        best_child.time_budget = self.time_budget

        # (optional) record sims for diagnostics
        self.sims_this_move = sims
        print(f"Simulations: {sims} (avg depth: {self.avg_playout_depth:.2f})")

        return {
            "action": best_child.parent_action[0],
            "res": best_child.parent_action[1],
            "res_node": best_child,
        }

    def find_child(self, action):
        for child in self.children:
            if child.parent_action[0] == action:
                return child
