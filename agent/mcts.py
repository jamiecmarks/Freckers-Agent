import random
import numpy as np
from collections import defaultdict
from .bitboard import BitBoard
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord
from .strategy import Strategy


class MonteCarloTreeSearchNode(Strategy):
    def __init__(self, state: BitBoard, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action  # (action, res)
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._total_reward = 0
        self._untried_actions = self.untried_actions()
        self.c = 0.25

        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def untried_actions(self):
        all_moves = self.state.get_all_optimal_moves()
        if len(all_moves) > 1:
            return all_moves
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
        self._untried_actions.sort(
            key=lambda mv: self.state._move_priority(mv), reverse=True
        )
        idx = 0  # pop the very best move first
        # action_res = self._untried_actions.pop()

        # if self.depth < 5 and self.depth > 1:
        #     # try to pop a GrowAction first
        #     for i, (act, res) in enumerate(self._untried_actions):
        #         if isinstance(act, GrowAction):
        #             idx = i
        #             break
        #     else:
        #         idx = np.random.randint(len(self._untried_actions))
        # else:
        #     idx = np.random.randint(len(self._untried_actions))
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
        if num_moves == 0:
            return None

        # 1) ε-greedy: 10% of the time, explore uniformly at random
        #
        eps = 0 if depth > 15 else 0.02
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
        # if few lilies remain, let the playout run to a true end
        # lily_count = (state.get_board() == state.LILLY).sum()
        # if lily_count < BOARD_N:
        #     max_depth = BOARD_N * BOARD_N  # effectively unlimited
        # else:
        # max_depth = 20
        # state = self.state
        # depth = 0
        # max_depth = 50  # if depth < 50 else 150
        max_depth = 20

        # fast, stateless playout on BitBoard only
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
                action, res = self.new_rollout_policy(state, depth=depth)
                state = state.move(action, res)
                state.toggle_player()

            # state.toggle_player()
            # action, res = self.rollout_policy(state, depth=self.depth + depth // 2)
            # state = state.move(action, res)
            # state.toggle_player()
            depth += 1

        if state.is_game_over():
            return state.get_winner()

        # return state.evaluate_position()
        return 1 if state.evaluate_position() > 0 else -1
        # return 1 if self.heuristic_score(state) > 0 else -1

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
            priority_bonus = self.state._move_priority(c.parent_action)
            scores.append(value + exploration + 0.1 * priority_bonus)  # tune this
        return self.children[np.argmax(scores)]

        scores = [
            (c.q() / c.n()) + self.dynamic_c() * np.sqrt((2 * np.log(self.n()) / c.n()))
            for c in self.children
        ]
        return self.children[np.argmax(scores)]

    def choose_next_action(self):
        # break ties by winrate
        best = max(
            self.children,
            key=lambda c: (c.n(), (self._results[1] / c.n()) if c.n() > 0 else 0),
        )
        return best

    def best_action(self, simulation_no=75):
        for _ in range(simulation_no):
            v = self.new_tree_policy()
            reward = v.simulate_playout()
            v.backpropagate(reward)

        best = self.choose_next_action()
        return {
            "action": best.parent_action[0],
            "res": best.parent_action[1],
            "res_node": best,
        }  # if best else None

    def find_child(self, action):
        for child in self.children:
            if child.parent_action[0] == action:
                return child
