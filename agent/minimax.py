import random
from .bitboard import BitBoard
from referee.game.actions import GrowAction
from referee.game.constants import BOARD_N
from .strategy import Strategy
import time

"""

Minimax works as follows: Expands all possible moves to a given depth.
Chooses the move that gives the best possible guaranteed perforamnce. 

"""
CUTOFF_DEPTH = 5
START_DEPTH = 1
SHORTENING_FACTOR = 1
ASTAR = False
LARGE_VALUE = 999
SPEEDUP_FACTOR = 1
EVAL = "adaptive"
RANDOM_START = 0

class MinimaxSearchNode(Strategy):
    def __init__(self, state:BitBoard, parent = None, parent_action = None,
                 time_budget = 178.0):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        if parent is None:
            self.root_player = state.get_current_player()
            self.time_budget = time_budget
        else:
            self.root_player = state.get_current_player()
            self.time_budget = parent.time_budget
        self.children = []
        self.astar = False
        self.cutoff_depth = 4
        self._logging_pv = False
        self.history = []
        self.minimax = True
        self.weights = {"centrality": 0.09349139034748077, "double_jumps": 0.08574286103248596,
                           "distance": 0.9127612113952637, "mobility": 0.03814251720905304}


    def check_gameover_next(self):
        board = self.state
        if board.get_ply_count() > 148:

            eval = 10 * self.simple_eval(board)
            with open("eval.txt", "w") as fp:
                fp.write(f"{eval}")
            return eval

        moves = board.get_all_moves()

        for action in moves:
            new_state = board.move(action[0], action[1])
            new_state.toggle_player()  # After move, opponent's turn
            if new_state.is_game_over():
                eval = 10 * self.simple_eval(new_state)

                with open("eval.txt", "w") as fp:
                    fp.write(f"{eval}")
                return eval
        return 0


    def cutoff_test(self, state:BitBoard, depth, cutoff_depth):
        if state.is_game_over():
            return True
        if depth >= cutoff_depth:
            return True
        return False

    def eval_function(self, state:BitBoard):
        """
        If cutoff test returns true, then we evaluate the position
        """
        if state.is_game_over():
            frog_count, opp_count = (
                state.frog_border_count[BitBoard.RED],
                state.frog_border_count[BitBoard.BLUE],
                )
            if frog_count == BOARD_N - 2 and opp_count != BOARD_N - 2:
                winner_piece = BitBoard.RED
            elif opp_count == BOARD_N - 2 and frog_count != BOARD_N - 2:
                winner_piece = BitBoard.BLUE
            else:
                return 0  # draw
            return float("inf") if winner_piece == self.root_player else float("-inf")
        if EVAL == "adaptive":
            return self.adaptive_eval(state)
        return self.simple_eval(state)


    def adaptive_eval(self, state: BitBoard, check = False):
        w = self.weights
        me = self.root_player



       # print("Root: ", self.root_player)
       # print("Original state:\n", self.state.render())
       # print("Current state:\n", state.render())


        you = BitBoard.RED if me == BitBoard.BLUE else BitBoard.BLUE
        board = state.get_board()
        progress = 0 

        mid = (BOARD_N - 1)/2
        centrality = 0
        
        # Centrality and comparative distance to goal
        if me == BitBoard.RED:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.RED:
                        progress += (r+1) # Starts at 1 up to 8
                        centrality += mid - abs(c - mid)
                    elif board[r][c] == BitBoard.BLUE:
                        progress -= (8 - (r)) # Starts at 1 up to 8
                        centrality -= mid - abs(c - mid)

        if me == BitBoard.BLUE:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.BLUE:
                        progress += (8 - r) # Starts at up 1 to 8
                        centrality += mid - abs(c-mid)
                    elif board[r][c] == BitBoard.RED:
                        centrality -= mid - abs(c-mid) 
                        progress -= (r+1) # starts at 1 up to 8


        # normalize and combine
        norm_prog = progress/8
        # norm_mob  = (len(moves_me) - len(moves_you)) / (len(moves_me)+ len(moves_you) + 1)
        norm_cent  = (centrality)/64
        # norm_doubles = (doubles_me - doubles_you)/(doubles_me + doubles_you + 1)
        
        
        score = (
             norm_prog
            + norm_cent
        )
       # print("And finally, current state score: ",score)
       # print()

        return score

    
    def simple_eval(self, state):
        current_player = self.root_player
        # If frog, we are moving towards the bottom
        board = state.get_board()
        progress = 0
        if current_player == BitBoard.RED:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.RED:
                        progress += (r+1)
                    elif board[r][c] == BitBoard.BLUE:
                        progress -= (8 - (r))
        if current_player == BitBoard.BLUE:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.BLUE:
                        progress += (8 - (r))
                    elif board[r][c] == BitBoard.RED:
                        progress -= (r+1)
        return progress/64

    def max_value(self, state: BitBoard, alpha, beta, depth, cutoff_depth):
        if self.cutoff_test(state, depth, cutoff_depth):
            return self.eval_function(state)
        value = float("-inf")
        moves = state.get_all_moves()

        for action in moves:
            new_position = state.move(action[0], action[1])
            new_position.toggle_player()
            value = max(value, self.min_value(new_position, alpha, beta, depth +1, cutoff_depth))
            if value >= beta:
                return value
            alpha = max(alpha, value) # Max is allowed to determine alpha
        return value

    def min_value(self, state:BitBoard, alpha, beta, depth, cutoff_depth):
        if self.cutoff_test(state, depth, cutoff_depth):
            return self.eval_function(state)
        value = float("inf")
        moves = state.get_all_moves()
        
        for action in moves:
            new_position = state.move(action[0], action[1])
            new_position.toggle_player()
            value = min(value, self.max_value(new_position, alpha, beta, depth +1, cutoff_depth))
            if value <=alpha:
                return value
            beta = min(beta, value) # Min is allowed to determine beta
        return value


    def best_action(self, safety_margin: float = 5, bt: float = 0.75):
        # 1) grab the referee‑supplied clock once
        # self.history.append((self.state.lilly_bits, self.state.frog_bits, self.state.opp_bits, self.state.get_current_player()))
        total_time = self.time_budget
        assert total_time > safety_margin, "No time to move!"
        self.check_gameover_next()
        # 2) compute moves_left as before
        moves_played = self.state.get_ply_count()
        moves_left = (150 - moves_played)//2

        # 3) ideal slice, then clamp
        raw_alloc = (total_time - safety_margin) / (moves_left**bt)
        alloc_time = min(raw_alloc, total_time - safety_margin)
        assert alloc_time >= 0

        # 4) set deadline
        t0 = time.perf_counter()
        deadline = t0 + alloc_time

        best_move = None
        depth = START_DEPTH

        # If early game random
        if self.state.get_ply_count() < RANDOM_START:
            choice = random.choice(self.state.get_all_moves())
            print("Random move for early game")
            return {"action": choice[0]}


        # Minimax is here 
        early_return_flag = False
        best_score = float("-inf")
        best_action = None
        moves = self.state.get_all_moves()
        random.shuffle(moves)
        
        depth = 1
        best_action_local = None
        while time.perf_counter() <= deadline and not early_return_flag:

            # After every loop, we asssign our result to best action
            best_action = best_action_local
            best_score_local = float("-inf")
            best_action_local = None

            for action, res in moves:
                if time.perf_counter() >= deadline:
                    early_return_flag = True
                    break
                
                alpha0, beta0 = float("-inf"), float("inf")
                # if time.perf_counter() >= deadline:
                new_state = self.state.move(action, res)
                new_state.toggle_player()
                val = self.min_value(new_state, alpha0, beta0, 1, depth)

                if val > best_score_local:
                    best_score_local = val
                    best_action_local= (action, res)

                alpha0 = max(alpha0, best_score_local)

                if alpha0 == float("inf"):
                    print("Win found at depth", depth)
                    early_return_flag = True
                    best_action = best_action_local
                    break


            depth += 1
        print("Last attempted depth: ", depth)

        


        if best_action is None:
            # fallback to first legal move
            best_action = self.state.get_all_moves()[0]

        print(f"Selected move: {best_action}")

        return {"action": best_action[0]}

