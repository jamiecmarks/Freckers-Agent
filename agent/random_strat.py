import numpy as np
from referee.game.actions import GrowAction, MoveAction
from .strategy import Strategy
import random
from .bitboard import BitBoard
class RandomStrat(Strategy):
    def __init__(self, state: BitBoard):
        self.weighted = True
        super().__init__(state)

    def find_child(self, action):
        next_board = self.state.move(action)
        # next_board.toggle_player()
        return RandomStrat(next_board)

    def best_action(self):
        print(self.state.current_player)
        possible_moves = self.state.get_all_moves()
        if self.weighted:
            vert_dists = []
            for mv, res in possible_moves:
                if isinstance(mv, GrowAction):
                    vert_dists.append(1)  # just count MoveAction as vert_dist of 1
                else:
                    vert_dists.append(abs(mv.coord.r - res.r))
            max_dist = max(vert_dists)
            # print([possible_moves[i] for i, dist in enumerate(vert_dists) if max_dist == dist])
            idx = random.choice(
                [i for i, dist in enumerate(vert_dists) if max_dist == dist]
            )
        else:
            idx = np.random.randint(len(possible_moves))
        action = possible_moves[idx][0]
        # res = None
        # if not isinstance(action, GrowAction):
        #     res = action.coord
        return {"action": action}

