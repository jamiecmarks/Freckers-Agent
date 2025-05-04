
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
import json


"""

Minimax works as follows: Expands all possible moves to a given depth.
Chooses the move that gives the best possible guaranteed perforamnce. 

"""
START_DEPTH = 1
SHORTENING_FACTOR = 1
ASTAR = False
LARGE_VALUE = 999
SPEEDUP_FACTOR = 5
EVAL = "adaptive"

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
        
        default = {'W_DIST':0.4, 'W_MOB':0.3, 'W_BORDER':0.2, 'W_CENTRAL':0.1}
        with open("weights2.json", "r") as wf:
            self.weights = json.load(wf)


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
        if me == BitBoard.FROG:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.FROG:
                        progress += (r+1)
                    elif board[r][c] == BitBoard.OPPONENT:
                        progress -= (8 - (r+1))
        if me == BitBoard.OPPONENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.OPPONENT:
                        progress += (8 - (r+1))
                    elif board[r][c] == BitBoard.FROG:
                        progress -= (r+1)
        # 2) mobility

        moves_me  = len(state.get_all_moves())
        state.toggle_player()
        moves_you = len(state.get_all_moves())
        state.toggle_player()

        # 4) centralization
        mid = (BOARD_N - 1) / 2
        cent_me = cent_you = 0.0
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == me:
                    cent_me  += mid - abs(c - mid)
                elif board[r][c] == you:
                    cent_you += mid - abs(c - mid)

        # normalize and combine
        norm_mob  = (moves_me - moves_you) / (moves_me + moves_you + 1)
        norm_cent  = (cent_me - cent_you) / (BOARD_N * BOARD_N)

        score = (
            w['W_DIST']   * progress
           + w['W_MOB']    * norm_mob
           + w['W_CENTRAL']* norm_cent
        )
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
                        progress -= (8 - (r+1))
        if current_player == BitBoard.OPPONENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.OPPONENT:
                        progress += (8 - (r+1))
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
        if self.state.get_ply_count() < 6:
            moves = self.state.get_all_moves()
            move_choice =  random.choice(moves)
            print("Random move for early game")
            return {"action": move_choice[0]}

        while True:
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
                    best_at_depth = action[0]
                    if value > LARGE_VALUE:
                        print("early return")
                        return {"action": best_at_depth}

            if time.perf_counter() < hard_deadline and best_at_depth is not None:
                best_move = best_at_depth
                cutoff_depth += 1
            else:
                break       
        print("Best action is", best_move)
        print("Max depth searched is: ", cutoff_depth-1)
        return {"action": best_move}


       #  while True:
       #      now = time.perf_counter()
       #      if now >= hard_deadline:
       #          break

            # since hard_deadline = t0 + alloc_time ≤ t0 + (total_time - safety_margin),
            # we are guaranteed never to go beyond the referee’s remaining clock.



