import numpy as np
from collections import defaultdict
from agent.bitboard import BitBoard


# Credit to https://ai-boson.github.io/mcts/ used as a basic starting point for this class
class MonteCarloTreeSearchNode:
    def __init__(self, state: BitBoard, parent=None, parent_action=None):
        self.state = state  # board state, in this case a BitBoard
        self.parent = parent  # parent node
        self.parent_action = parent_action  # action taken to reach this node
        self.children = []  # list of child nodes (all possible actions)
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()  # the options we didn't chose

    def untried_actions(self):
        return self.state.get_all_moves()

    def q(self):
        # wins - losses
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action, res = self._untried_actions.pop()
        next_state = self.state.move(action, res)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action
        )

        self.children.append(child_node)
        return child_node
