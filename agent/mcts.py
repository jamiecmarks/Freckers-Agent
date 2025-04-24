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
        self._untried_actions = self.untried_actions()
        self.c = 0.25

        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def untried_actions(self):
        return self.state.get_all_moves()

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        # action_res = self._untried_actions.pop()
        idx = np.random.randint(len(self._untried_actions))

        action_res = self._untried_actions.pop(idx)
        action, res = action_res
        next_state = self.state.move(action, res)
        next_state.toggle_player()

        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action_res
        )

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def fully_expand(self):
        while not self.is_fully_expanded():
            self.expand()

    def rollout_policy(self, state):
        weights = []

        best_score = -float("inf")
        best_move = None
        mid_col = (BOARD_N - 1) // 2
        curr_player = state.get_current_player()

        for action, res in state.get_all_moves():
            score = 0
            # bias grow early, disfavor later
            if isinstance(action, GrowAction):
                # score += 0.2 if (self.depth < 15 and self.depth > 2) else -0.2
                total_cells = BOARD_N * BOARD_N
                lily_count = (state.get_board() == state.LILLY).sum()
                score += 1.0 - (lily_count / total_cells)
            else:
                # e.g. reward long jumps, centering, etc.

                # 2b) multi-jump bonus
                jump_dist = abs(res.r - action.coord.r)
                if jump_dist > 1:
                    score += 0.2 * jump_dist

                # 2c) centering bonus
                start_c, end_c = action.coord.c, res.c
                if abs(start_c - mid_col) > abs(end_c - mid_col):
                    score += 0.1

                # # 2d) opponent’s next maximum jump penalty
                # # simulate the move, toggle to opponent, measure their best jump
                # child = state.move(action, res)
                # child.toggle_player()
                # opp_max = 0
                # for mv, mv_res in child.get_all_moves():
                #     if isinstance(mv, MoveAction):
                #         opp_max = max(opp_max, abs(mv_res.r - mv.coord.r))
                # score -= 0.1 * opp_max
            if score > best_score:
                best_score = score
                best_move = (action, res)

            # 2c) UCT score
            # exploit = child.q() / n
            # explore = self.c * math.sqrt((2 * math.log(N)) / n)
            # weights.append(mult * (exploit + explore))
        if best_move is None:
            return random.choice(state.get_all_moves())
        return best_move
        tot = sum(weights) + len(weights) * abs(min(weights))
        weights = [(w + abs(min(weights))) / tot for w in weights]

        probs = [w / sum(weights) for w in weights]
        return random.choices(node.children, weights=probs)[0]

    # new version of rollout
    def simulate_playout(self):
        state = self.state
        depth = 0
        max_depth = 10

        # fast, stateless playout on BitBoard only
        while not state.is_game_over() and depth < max_depth:
            # moves = state.get_all_moves()
            action, res = self.rollout_policy(state)
            state = state.move(action, res)
            state.toggle_player()
            depth += 1

        if state.is_game_over():
            return state.get_winner()

        return 1 if self.heuristic_score(state) > 0 else -1

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

        if self.parent:
            self.parent.backpropagate(-result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def dynamic_c(self):
        # start exploratory, then exploit more later
        return self.c / (1 + np.log(1 + self.n()))

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

        scores = [
            (c.q() / c.n()) + self.dynamic_c() * np.sqrt((2 * np.log(self.n()) / c.n()))
            for c in self.children
        ]
        return self.children[np.argmax(scores)]

    def choose_next_action(self):
        children = self.children
        return max(children, key=lambda c: c.n())

    def best_action(self, simulation_no=100):
        for _ in range(simulation_no):
            v = self._tree_policy()
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
