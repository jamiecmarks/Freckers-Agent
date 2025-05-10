from bitboard_io import *
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


    def adaptive_eval(self, state: BitBoard):
        w = self.weights
        me = self.root_player

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
        
        # 2) mobility
        swap_back = False
        if state.current_player != self.root_player:
            state.toggle_player() # So now we are the same as the root player
            swap_back = True # If we started as opposite to root player, remember

        moves_me  = state.get_all_moves()
        state.toggle_player() # Now we are the same as the opposition player
        moves_you = state.get_all_moves()

        if not swap_back: # Ensures that the toggle happens twice in all cases
            state.toggle_player() # 



        # 3) Total double jumps
        doubles_me = 0
        doubles_you = 0
        for move in moves_me:
            if isinstance(move[0], GrowAction):
                continue
            if abs(move[0].coord.r - move[1].r) >1:
                doubles_me +=1
        for move in moves_you:
            if isinstance(move[0], GrowAction):
                continue
            if abs(move[0].coord.r - move[1].r) > 1:
                doubles_you+=1
        

        # normalize and combine
        norm_prog = progress/64
        norm_mob  = (len(moves_me) - len(moves_you)) / (len(moves_me)+ len(moves_you) + 1)
        norm_cent  = (centrality)/64
        norm_doubles = (doubles_me - doubles_you)/(doubles_me + doubles_you + 1)
        

        score = (
            w['distance']   * norm_prog
           # + w['mobility']    * norm_mob
            +w['centrality'] * norm_cent
           # w['double_jumps'] * norm_doubles
        )
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
            if value >= beta:
                return value
            beta = min(beta, value) # Min is allowed to determine beta
        return min(beta, value)


    def best_action(self, safety_margin: float = 5, bt: float = 0.75):
        # 1) grab the referee‐supplied clock once
        self.history.append((self.state.lilly_bits, self.state.frog_bits, self.state.opp_bits, self.state.get_current_player()))


        # print("value is initially", self.adaptive_eval(self.state))
        total_time = self.time_budget
        assert total_time > safety_margin, "No time to move!"

        # 2) compute moves_left as before
        moves_played = self.state.get_ply_count()
        moves_left = (150 - moves_played)//2

        # 3) ideal slice, then clamp below
        raw_alloc = (total_time - safety_margin) / (moves_left**bt)
        alloc_time = min(raw_alloc, total_time - safety_margin)

        assert alloc_time >= 0

        # 4) establish one single *absolute* deadline
        t0 = time.perf_counter()
        hard_deadline = t0 + alloc_time
        
        best_move = None
        alpha  = float("-inf") # Optimal score
        beta = float("inf") # Opponent's optimal score (-inf from their perspective)

        all_moves = self.state.get_all_moves()
        current_state = self.state
        cutoff_depth = START_DEPTH

        if self.state.get_ply_count() < RANDOM_START:
            moves = self.state.get_all_moves()
            move_choice =  random.choice(moves)
            print("Random move for early game")
            return {"action": move_choice[0]}
        
        best_val = None
        early_return_flag = False
        while True and not early_return_flag:
            if time.perf_counter() >= hard_deadline:
                print(time.perf_counter())
                print(alloc_time)
                break

            best_at_depth = None
            best_val = float("-inf")
            all_moves = self.state.get_all_moves()


            for action in all_moves:
                if time.perf_counter() >= hard_deadline:
                    break
                new_position = current_state.move(action[0], action[1])
                new_position.toggle_player()
                value = self.min_value(new_position, alpha, beta, 1, cutoff_depth)
                if value >= best_val:
                    best_val = value
                    best_at_depth = action
                    if value > LARGE_VALUE:
                        print("early return")
                        best_move = action
                        early_return_flag = True
                        break

            if time.perf_counter() < hard_deadline and best_at_depth is not None:
                best_move = best_at_depth
                cutoff_depth += 1
            else:
                break       
        print("Best action is", best_move)
        # print("value is", self.adaptive_eval(current_state.move(best_move[0], best_move[1])))
        print("Max depth searched is: ", cutoff_depth)

        next_state = self.state.move(best_move[0], best_move[1])
        next_state.toggle_player()
        self.history.append((next_state.lilly_bits, next_state.frog_bits, next_state.opp_bits, next_state.get_current_player()))

        return {"action": best_move[0]}
