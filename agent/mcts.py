import numpy as np
from collections import defaultdict
from agent.bitboard import BitBoard


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
