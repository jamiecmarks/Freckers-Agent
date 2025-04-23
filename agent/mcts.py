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
        self.c = 1.0

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

    def rollout(self):
        current_rollout_state = self.state
        max_depth = 100
        depth = 0

        while not current_rollout_state.is_game_over() and max_depth > depth:
            possible_moves = current_rollout_state.get_all_moves()

            if not possible_moves:
                break

            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action[0], action[1])
            current_rollout_state.toggle_player()
            depth += 1

        # didn't reach a terminal state
        if not current_rollout_state.is_game_over():
            # positive = good for current player, negative otherwise
            score = 0
            board = current_rollout_state.get_board()
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == current_rollout_state.FROG:
                        score += r
                    elif board[r][c] == current_rollout_state.OPPONENT:
                        score -= BOARD_N - 1 - r
            return 1 if score > 0 else (-1 if score < 0 else 0)

        return current_rollout_state.get_winner()

    def rollout1(self):
        # current_rollout_state = self.state
        current_rollout = self
        max_depth = 20 if self.depth < 80 else 40
        depth = 0

        while not current_rollout.is_terminal_node() and max_depth > depth:
            curr_player = current_rollout.state.get_current_player()
            current_rollout = current_rollout._tree_policy()
            if current_rollout.state.get_current_player() == curr_player:
                current_rollout.state.toggle_player()

            #     current_rollout = current_rollout.best_child()
            #     action = current_rollout.parent_action()
            #     # current_rollout_state = current_rollout_state.move(action[0], action[1])
            #     current_rollout.state.toggle_player()
            depth += 1

        # didn't reach a terminal state
        if not current_rollout.is_terminal_node():
            # positive = good for current player, negative otherwise
            score = 0
            board = current_rollout.state.get_board()
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
            return 1 if score > 0 else (-1 if score < 0 else 0)
        # else:
        #     print(f"I am {self.state.get_current_player()}")
        #     print(current_rollout.state.get_board())

        return current_rollout.state.get_winner()

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

    def best_child(self):
        # chooose_rand = len(self._untried_actions)
        choices_weights = []
        for c in self.children:
            if c.is_terminal_node():
                return c

            if c.n() == 0:
                choices_weights.append(float("inf"))  # prioritize unvisited nodes
                # choices_weights.append(c.q() / c.n())
            else:
                mult = 1
                if isinstance(c.parent_action[0], GrowAction):
                    mult += 0.2 if self.depth < 15 and self.depth > 2 else -0.2
                else:
                    start_r = c.parent_action[0].coord.r
                    end_r = c.parent_action[1].r
                    start_c = c.parent_action[0].coord.c
                    end_c = c.parent_action[1].c
                    vert_dist = abs(start_r - end_r)
                    if vert_dist > 1:
                        mult += 0.2 * vert_dist
                    elif vert_dist == 0:
                        mult -= 0.2
                    midboard = (BOARD_N - 1) // 2
                    if abs(start_c - midboard) > abs(end_c - midboard):
                        # moving towards the middle is good
                        mult += 0.2 if self.depth < 25 else 0

                choices_weights.append(
                    mult
                    * (
                        (c.q() / c.n())
                        + self.c * np.sqrt((2 * np.log(self.n()) / c.n()))
                    )
                )
        max_val = max(choices_weights)
        all_max = [i for i, v in enumerate(choices_weights) if v == max_val]
        return self.children[random.choice(all_max)]

    def heuristic_score(self, move, current_player):
        action, res = move
        # moving actions
        if isinstance(action, MoveAction):
            # how “forward” is the landing row?
            # frogs want to maximize r, opponents minimize r
            dist = res.r if current_player == BitBoard.FROG else (BOARD_N - 1 - res.r)
            return dist
        # growing is valuable early on (when few lilies exist)
        elif isinstance(action, GrowAction):
            return 0.5
        else:
            return 0.0

    def rollout_policy(self, possible_moves):
        # compute raw scores
        #
        eps = 0.2  # 80% follow “best” move, 20% pick random
        if np.random.rand() < eps:
            return possible_moves[np.random.randint(len(possible_moves))]

        best = max(possible_moves, key=lambda mv: self.state._move_priority(mv))

        return best

        #
        # scores = [
        #     self.heuristic_score(mv, self.state.get_current_player())
        #     for mv in possible_moves
        # ]
        # # shift to all‑positive and softmax
        # exp_scores = np.exp(np.array(scores) / 0.5)  # 0.5 is the temperature
        #
        # probs = exp_scores / exp_scores.sum()
        # idx = np.random.choice(len(possible_moves), p=probs)
        # return possible_moves[idx]

        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        # while not current_node.is_terminal_node():
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
        return current_node

    def choose_next_action(self):
        children = self.children
        num_visited = []
        for child in children:
            num_visited.append(child.n())

        return children[np.argmax(num_visited)]

    def best_action(self, simulation_no=150):
        # if not self.children and self._untried_actions:
        #     self.expand()

        for _ in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout1()
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
