
SUBMITTING = False
from bitboard_io import *
from enum import nonmember
import random
import numpy as np
import torch as to
from collections import defaultdict
from .bitboard import BitBoard
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord
from .strategy import Strategy
import time
import os
# import pandas as pd
if not SUBMITTING:
    import json

# Need to punish: leaving one of our frogs behind. Generally the cause of losses. 
# Priorities at different states of the game:
#   - Early game: Expand forwards as quickly as possible. This is generally done
    # via double jumps. 
    # Midgame - distance. we wnat moves which make us advance more than the other team
    # Endgame - More of a stress on average mobility (mobility of pieces not already finished)


"""

Minimax works as follows: Expands all possible moves to a given depth.
Chooses the move that gives the best possible guaranteed perforamnce. 

"""
START_DEPTH = 1
SHORTENING_FACTOR = 1
ASTAR = False
LARGE_VALUE = 999
SPEEDUP_FACTOR = 1
EVAL = "normal"
RANDOM_START = 0

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
        self.history = []

        self.model =  BitboardNet()
        self.model.load_state_dict(torch.load("bitboard_model.pt", map_location="cpu"))
        self.model.eval()


        self.new_weights = {
                'distance':1,
                'goal_count': 1,
                'connectivity': 1,
                'dispersion': 1,
                'mobility_diff': 1,
                'jump_mobility':1 
        }


        if SUBMITTING:
            # Earlygame
            if self.state.get_ply_count() <20:
                self.weights = {"centrality": 0.3, "double_jumps": 0.8,
                           "distance": 10.5, "mobility": 0.1}
            # Midgame
            elif self.state.get_ply_count() <40:
                self.weights = {"centrality": 0.4, "double_jumps": 0.4,
                           "distance": 10.8, "mobility": 0.3}
            # Endgame
            else:   
                self.weights = {"centrality": 0.09349139034748077, "double_jumps": 0.08574286103248596,
                           "distance": 10.9127612113952637, "mobility": 0.03814251720905304}
        else:
            with open("weights.json", "r") as wf:
                self.weights = json.load(wf)



    def neural_eval(self, state, device="cpu"):
        self.model.to(device)
        self.model.eval()
        lilly = state.lilly_bits
        red = state.frog_bits
        blue = state.opp_bits
        tensor = bitboard_to_tensor(lilly, red, blue)
        X = torch.tensor(tensor[None, ...]).to(device)  # shape (1, 3, 8, 8)

        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Example 1: return red's win probability
        if self.root_player != state.current_player:
            return -probs[1]  # assuming class 1 = red wins
        return probs[1]



    def check_gameover_next(self):
        board = self.state
        nd_board = self.state.get_board()
        if board.get_ply_count() > 148:
            # All the neural network stuff
            eval = 10 * self.simple_eval(board)

            # F = np.loadtxt("red_pv_features.csv", delimiter=",", skiprows=1)
            # deltas = F[-1] - F[0]
            # norm = np.abs(deltas).sum() + 1e-8
            # adv = (deltas / norm) * eval
           # np.savetxt("red_advantage.txt", adv, fmt="%.6f")

            with open("eval.txt", "w") as fp:
                fp.write(f"{eval}")
            return eval

        moves = board.get_all_moves()
        for action in moves:
            new_state = board.move(action[0], action[1])
            new_state.toggle_player()  # After move, opponent's turn
            compatible_state = new_state.get_board()
            if new_state.is_game_over():
                eval = 10 * self.simple_eval(new_state)

                # All the neural network stuff
                # F = np.loadtxt("red_pv_features.csv", delimiter=",", skiprows=1)
                # deltas = F[-1] - F[0]
                # norm = np.abs(deltas).sum() + 1e-8
                # adv = 100 * (deltas / norm) * eval
                # print("Red advantage time!")
                # np.savetxt("red_advantage.txt", adv[1:], fmt="%.6f")

                with open("eval.txt", "w") as fp:
                    fp.write(f"{eval}")
                return eval
        return 0


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
            return self.simple_eval(state)
        return self.simple_eval(state)

    def extract_features(self, state: BitBoard, root_player: int) -> dict:
        """
        Compute adversarial feature differences in one board scan.
        Returns a dict mapping each feature to (root_player - opponent).
        Features: distance, goal_count, connectivity, dispersion, mobility_diff, jump_mobility
        """
        board = state.get_board()
        halfway = BOARD_N // 2

        # Aggregates for both players
        agg = {
            BitBoard.RED: {'distance': 0, 'goal_count': 0, 'connectivity': 0, 'min_row': BOARD_N, 'max_row': -1},
            BitBoard.BLUE: {'distance': 0, 'goal_count': 0, 'connectivity': 0, 'min_row': BOARD_N, 'max_row': -1}
        }

        # Single pass: distance, goal_count, connectivity, dispersion bounds
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                piece = board[r][c]
                if piece in agg:
                    data = agg[piece]
                    # Distance/progress
                    data['distance'] += (r + 1) if piece == BitBoard.RED else (BOARD_N - r)
                    # Goal occupancy
                    if (piece == BitBoard.RED and r >= halfway) or (piece == BitBoard.BLUE and r < halfway):
                        data['goal_count'] += 1
                    # Connectivity: right & down
                    if c + 1 < BOARD_N and board[r][c + 1] == piece:
                        data['connectivity'] += 1
                    if r + 1 < BOARD_N and board[r + 1][c] == piece:
                        data['connectivity'] += 1
                    # Dispersion bounds
                    data['min_row'] = min(data['min_row'], r)
                    data['max_row'] = max(data['max_row'], r)

        # Initialize raw feature dicts
        raw = {p: {} for p in agg}
        for player, data in agg.items():
            dispersion = (data['max_row'] - data['min_row']) if data['max_row'] >= data['min_row'] else 0
            raw[player].update({
                'distance': data['distance'],
                'goal_count': data['goal_count'],
                'connectivity': data['connectivity'],
                'dispersion': dispersion
            })

        # Compute mobility_diff and jump_mobility for each
        current = state.get_current_player()
        moves = {}
        for player in (BitBoard.RED, BitBoard.BLUE):
            if state.get_current_player() != player:
                state.toggle_player()
            moves[player] = state.get_all_moves()
        if state.get_current_player() != current:
            state.toggle_player()

        for player, my_moves in moves.items():
            my_count = len(my_moves)
            opp = BitBoard.RED if player == BitBoard.BLUE else BitBoard.BLUE
            opp_count = len(moves[opp])
            mobility_diff = (my_count - opp_count) / (my_count + opp_count + 1)
            seen = set()
            for action, dest in my_moves:
                if not isinstance(action, GrowAction) and abs(action.coord.r - dest.r) > 1:
                    seen.add((action.coord.r, action.coord.c))
            jump_mobility = len(seen) / (my_count + 1)
            raw[player].update({
                'mobility_diff': mobility_diff,
                'jump_mobility': jump_mobility
            })

        # Adversarial differences
        opp_player = BitBoard.RED if root_player == BitBoard.BLUE else BitBoard.BLUE
        return {feat: raw[root_player][feat] - raw[opp_player][feat] for feat in raw[root_player]}




    def evaluate_with_weights(self, state: BitBoard, root_player: int) -> float:
        """
        Compute weighted heuristic score: sum(weight_i * feature_i).
        Uses adversarial features from extract_features.
        """
        feats = self.extract_features(state, root_player)
        score = 0.0
        for name, value in feats.items():
            w = self.weights.get(name, 0.0)
            score += w * value
        return score



    def adaptive_eval(self, state: BitBoard):
        w = self.weights
        me = self.root_player
        you = BitBoard.RED if me == BitBoard.BLUE else BitBoard.BLUE
        board = state.get_board()
        progress = 0 

        mid = (BOARD_N - 1)/2
        cent_me = cent_you = 0
        
        # Centrality and comparative distance to goal
        if me == BitBoard.RED:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.RED:
                        progress += (r+1) # Starts at 1 up to 8
                        cent_me += mid - abs(c - mid)
                    elif board[r][c] == BitBoard.BLUE:
                        progress -= (8 - (r)) # Starts at 1 up to 8
                        cent_you += mid - abs(c - mid)

        if me == BitBoard.BLUE:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.BLUE:
                        progress += (8 - r) # Starts at up 1 to 8
                        cent_me += mid - abs(c-mid)
                    elif board[r][c] == BitBoard.RED:
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


        # Calculate the percent contribution of each metric to the overall heuristic return
        weighted_centrality = abs(w['centrality'] * raw[0])
        weighted_doubles = abs(w['double_jumps'] * raw[1])
        weighted_distance = abs(w['distance'] * raw[2])
        weighted_mobility = abs(w['mobility'] * raw[3])

        epsilon = 1e-4
        normaliser = weighted_centrality + weighted_doubles + weighted_distance + weighted_mobility + epsilon

        weighted_centrality = weighted_centrality/normaliser
        weighted_doubles = weighted_doubles/normaliser
        weighted_mobility = weighted_mobility/normaliser
        weighted_distance = weighted_distance/normaliser



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
        if current_player == BitBoard.RED:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.RED:
                        astar_check += (r+1)
        if current_player == BitBoard.BLUE:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.BLUE:
                        astar_check += (8- (r+1))
        return astar_check

    
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

        #print("bit lengths:", 
        #    int(self.state.lilly_bits).bit_length(), 
        #    int(self.state.frog_bits).bit_length(), 
        #    int(self.state.opp_bits).bit_length())

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
                if value >= best_val:
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
        print("Max depth searched is: ", cutoff_depth)

        # 2) PV logging of chosen move only
        self._pv_features = []
        self._logging_pv = True
        # log root features
        # _ = self.adaptive_eval(self.state)

        # apply and log child features

        next_state = self.state.move(best_move[0], best_move[1])
        next_state.toggle_player()
        self.history.append((next_state.lilly_bits, next_state.frog_bits, next_state.opp_bits, next_state.get_current_player()))

       # print("bit lengths:", 
       #     self.state.lilly_bits.bit_length(), 
       #     self.state.frog_bits.bit_length(), 
       #     self.state.opp_bits.bit_length())
        # 3) check for end-of-game & advantage computation
        winner = self.check_gameover_next()
        fname = "bitboards_win.bin" if winner > 0 else "bitboards_loss.bin"
        flag_path = "bitboards_logged.flag"
        if not os.path.exists(flag_path) and winner:
            if winner >0:
                print("We have a winner: red")
            else:
                print("We have a winner: blue")
            fname = "bitboards_win.bin" if winner > 0 else "bitboards_loss.bin"
            print(fname)
            print("History saved", self.history)
            save_game_record(fname, self.history)
            # create the flag
            with open(flag_path, "w") as f:
                f.write("done")




        return {"action": best_move[0]}


       #  while True:
       #      now = time.perf_counter()
       #      if now >= hard_deadline:
       #          break

            # since hard_deadline = t0 + alloc_time ≤ t0 + (total_time - safety_margin),
            # we are guaranteed never to go beyond the referee’s remaining clock.
