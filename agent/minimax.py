
from enum import nonmember
import random
import numpy as np
from collections import defaultdict
from .bitboard import BitBoard
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord
from .strategy import Strategy
import time
# import pandas as pd
import json


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
RANDOM_START =0

class MinimaxSearchNode(Strategy):
    def __init__(self, state:BitBoard, parent = None, parent_action = None,
                 time_budget = 178.0, weights = None):
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
        
        with open("weightsagent.json", "r") as wf:
            self.weights = json.load(wf)



    def check_gameover_next(self):
        board = self.state
        if board.get_ply_count() > 148:
            # All the neural network stuff
            eval = 10 * self.simple_eval(board)

            F = np.loadtxt("red_pv_features.csv", delimiter=",", skiprows=1)
            deltas = F[-1] - F[0]
            norm = np.abs(deltas).sum() + 1e-8
            adv = (deltas / norm) * eval
            np.savetxt("red_advantage.txt", adv, fmt="%.6f")

            with open("eval.txt", "w") as fp:
                fp.write(f"{eval}")
        moves = board.get_all_moves()
        for action in moves:
            new_state = board.move(action[0], action[1])
            new_state.toggle_player()  # After move, opponent's turn
            if new_state.is_game_over():
                eval = 10 * self.simple_eval(new_state)

                # All the neural network stuff
                F = np.loadtxt("red_pv_features.csv", delimiter=",", skiprows=1)
                deltas = F[-1] - F[0]
                norm = np.abs(deltas).sum() + 1e-8
                adv = 100 * (deltas / norm) * eval
                print("Red advantage time!")
                np.savetxt("red_advantage.txt", adv[1:], fmt="%.6f")

                with open("eval.txt", "w") as fp:
                    fp.write(f"{eval}")
                return 


    def cutoff_test(self, state:BitBoard, depth, cutoff_depth):
        if state.is_game_over():
            return True
        if depth >= cutoff_depth:
            # print("Cutoff depth exceeded")
            # print(state.render())
            return True
        return False

    def eval_function(self, state:BitBoard):
        """
        If cutoff test returns true, then we evaluate the position
        """
        if state.is_game_over():
            frog_count, opp_count = (
                state.frog_border_count[BitBoard.FROG],
                state.frog_border_count[BitBoard.OPPONENT],
                )
            if frog_count == BOARD_N - 2 and opp_count != BOARD_N - 2:
                winner_piece = BitBoard.FROG
            elif opp_count == BOARD_N - 2 and frog_count != BOARD_N - 2:
                winner_piece = BitBoard.OPPONENT
            else:
                return 0  # draw
            return float("inf") if winner_piece == self.root_player else float("-inf")
        if EVAL == "adaptive":
            return self.adaptive_eval(state)
        return self.simple_eval(state)

    
    def adaptive_eval(self, state: BitBoard):
        w = self.weights
        me = self.root_player
        you = BitBoard.FROG if me == BitBoard.OPPONENT else BitBoard.OPPONENT
        board = state.get_board()
        progress = 0 

        mid = (BOARD_N - 1)/2
        cent_me = cent_you = 0
        
        # Centrality and comparative distance to goal
        if me == BitBoard.FROG:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.FROG:
                        progress += (r+1) # Starts at 1 up to 8
                        cent_me += mid - abs(c - mid)
                    elif board[r][c] == BitBoard.OPPONENT:
                        progress -= (8 - (r)) # Starts at 1 up to 8
                        cent_you += mid - abs(c - mid)

        if me == BitBoard.OPPONENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.OPPONENT:
                        progress += (8 - r) # Starts at up 1 to 8
                        cent_me += mid - abs(c-mid)
                    elif board[r][c] == BitBoard.FROG:
                        cent_you += mid - abs(c-mid) 
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
        

        # What other types of adversarial measures?
        # Cluster level?

        # normalize and combine
        norm_mob  = (len(moves_me) - len(moves_you)) / (len(moves_me)+ len(moves_you) + 1)
        norm_cent  = (cent_me - cent_you) / (BOARD_N * BOARD_N)
        norm_doubles = (doubles_me - doubles_you)/(doubles_me + doubles_you + 1)
        raw = np.array([norm_cent,norm_doubles,progress, norm_mob], dtype=float)

        if self._logging_pv:
            print("Logging")
            move_idx = state.get_ply_count()
            with open("red_pv_features.csv", "a") as pf:
                target_str = f"{move_idx},{raw[0]},{raw[1]},{raw[2]},{raw[3]}\n"
                print(target_str)
                pf.write(f"{move_idx},{raw[0]},{raw[1]},{raw[2]},{raw[3]}\n")

        # print("Current player is: ", state.current_player)
        # print("Root player is: ", self.root_player)
        score = (
            w['distance']   * progress
           + w['mobility']    * norm_mob
           + w['centrality'] * norm_cent
           + w['double_jumps'] * norm_doubles
        )
        # print(f"Scores are (distance {progress}, mobility {norm_mob}, centrality {norm_cent}, doubles {norm_doubles}): ")
        # print(state.render())
        return score


    def astar_eval(self, state):
        current_player = self.root_player
        # If frog, we are moving towards the bottom
        board = state.board
        astar_check = 0
        if current_player == BitBoard.FROG:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.FROG:
                        astar_check += (r+1)
        if current_player == BitBoard.OPPONENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.OPPONENT:
                        astar_check += (8- (r+1))
        return astar_check

    
    def simple_eval(self, state):
        current_player = self.root_player
        # If frog, we are moving towards the bottom
        board = state.board
        progress = 0
        if current_player == BitBoard.FROG:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.FROG:
                        progress += (r+1)
                    elif board[r][c] == BitBoard.OPPONENT:
                        progress -= (8 - (r))
        if current_player == BitBoard.OPPONENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.OPPONENT:
                        progress += (8 - (r))
                    elif board[r][c] == BitBoard.FROG:
                        progress -= (r+1)
        return progress/64

    def max_value(self, state: BitBoard, alpha, beta, depth, cutoff_depth):
        if self.cutoff_test(state, depth, cutoff_depth):
            return self.eval_function(state)
        value = float("-inf")
        if self.astar == True:
            moves = state.get_all_optimal_moves()
            newmoves = state.get_all_moves()
            moves.extend(random.sample(newmoves, len(newmoves)//SHORTENING_FACTOR))
        else:
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
        if self.astar == True:
            moves = state.get_all_optimal_moves()
            newmoves = state.get_all_moves()
            moves.extend(random.sample(newmoves, len(newmoves)//SHORTENING_FACTOR))
        else:
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
        using_astar = self.state.get_all_optimal_moves()
        if len(using_astar)>=2 and ASTAR and self.state.get_ply_count() >=40:
            self.astar = True
        else:
            self.astar = False


        total_time = self.time_budget
        assert total_time > safety_margin, "No time to move!"

        # 2) compute moves_left as before
        moves_played = self.state.get_ply_count()
        moves_left = (150 - moves_played)//2

        # 3) ideal slice, then clamp below
        raw_alloc = (total_time - safety_margin) / (moves_left**bt)
        alloc_time = min(raw_alloc, total_time - safety_margin)
        alloc_time /= SPEEDUP_FACTOR

        assert alloc_time >= 0

        # 4) establish one single *absolute* deadline
        t0 = time.perf_counter()
        hard_deadline = t0 + alloc_time
        
        best_move = None
        alpha  = float("-inf") # Optimal score
        beta = float("inf") # Opponent's optimal score (-inf from their perspective)
        best_action = None
        all_moves = self.state.get_all_moves()
        current_state = self.state
        cutoff_depth = START_DEPTH
        if self.state.get_ply_count() < RANDOM_START:
            moves = self.state.get_all_moves()
            move_choice =  random.choice(moves)
            print("Random move for early game")
            return {"action": move_choice[0]}
        
        
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

                if value > best_val:
                    best_val = value
                    best_at_depth = action
                    if value > LARGE_VALUE:
                        best_move = action
                        early_return_flag = True
                        break

            if time.perf_counter() < hard_deadline and best_at_depth is not None:
                best_move = best_at_depth
                cutoff_depth += 1
            else:
                break       
        print("Best action is", best_move)
        print("Max depth searched is: ", cutoff_depth-1)

        # 2) PV logging of chosen move only
        self._pv_features = []
        self._logging_pv = True
        # log root features
        _ = self.adaptive_eval(self.state)

        # apply and log child features

        next_state = self.state.move(best_move[0], best_move[1])
        next_state.toggle_player()
        _ = self.adaptive_eval(next_state)
        # stop logging
        self._logging_pv = False

        # 3) check for end-of-game & advantage computation
        self.check_gameover_next()

        return {"action": best_move[0]}


       #  while True:
       #      now = time.perf_counter()
       #      if now >= hard_deadline:
       #          break

            # since hard_deadline = t0 + alloc_time ≤ t0 + (total_time - safety_margin),
            # we are guaranteed never to go beyond the referee’s remaining clock.



