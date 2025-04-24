import numpy as np
from referee.game.actions import GrowAction, MoveAction
from agent.strategy import Strategy


class RandomStrat(Strategy):
    def __init__(self, state):
        super().__init__(state)

    def find_child(self, action):
        next_board = self.state.move(action)
        # next_board.toggle_player()
        return RandomStrat(next_board)

    def best_action(self):
        possible_moves = self.state.get_all_moves()
        idx = np.random.randint(len(possible_moves))
        action = possible_moves[idx][0]
        # res = None
        # if not isinstance(action, GrowAction):
        #     res = action.coord
        return {"action": action}
