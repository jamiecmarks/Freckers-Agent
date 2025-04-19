import numpy as np
from collections import defaultdict
from agent.bitboard import BitBoard


class MonteCarloTreeSearchNode:
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
        max_depth = 150
        depth = 0

        while not current_rollout_state.is_game_over() and max_depth > depth:
            possible_moves = current_rollout_state.get_all_moves()
            if not possible_moves:
                break
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action[0], action[1])
            # TODO: is this correct? I feel like the reward is not properly getting propogated upward
            current_rollout_state.toggle_player()
            depth += 1

        return current_rollout_state.get_winner()

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(-result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = []
        for c in self.children:
            if c.n() == 0:
                choices_weights.append(float("inf"))  # prioritize unvisited nodes
            else:
                choices_weights.append(
                    (c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n()))
                )

        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulation_no=100):
        for _ in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        best = self.best_child(c_param=0.0)
        return {
            "action": best.parent_action[0],
            "res": best.parent_action[1],
            "res_node": best,
        }  # if best else None

    def find_child(self, action):
        for child in self.children:
            if child.parent_action[0] == action:
                return child

        return None
