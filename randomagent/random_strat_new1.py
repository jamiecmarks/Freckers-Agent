import numpy as np
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from ..agent.strategy import Strategy
import random


class RandomStrat(Strategy):
    def __init__(self, state):
        super().__init__(state)
        self.mid_col = (BOARD_N - 1) // 2
        self.game_phase = 0  # Will be updated in best_action

    def find_child(self, action):
        next_board = self.state.move(action)
        return RandomStrat(next_board)

    def _score_move(self, move, res):
        """Score a move based on multiple heuristics"""
        action, res = move
        score = 0

        # Game phase (0 to 1) based on ply count
        self.game_phase = min(1.0, self.state.get_ply_count() / 150)

        if isinstance(action, GrowAction):
            # Grow action scoring
            if self.game_phase < 0.2:  # Early game
                # Count current lily pads
                lily_count = bin(self.state.lilly_bits).count("1")
                board_density = lily_count / (BOARD_N * BOARD_N)
                
                # More valuable when board is sparse
                score = 100 * (1 - board_density)
            else:
                # Less valuable in mid/late game
                score = -50
        else:
            # Move action scoring
            start_r, start_c = action.coord.r, action.coord.c
            end_r, end_c = res.r, res.c
            
            # 1. Forward progress (most important)
            forward = 1 if self.state.current_player == self.state.RED else -1
            vert_progress = (end_r - start_r) * forward
            score += 1000 * vert_progress
            
            # 2. Multi-jump bonus
            num_hops = len(action.directions)
            if num_hops > 1:
                score += 500 * (num_hops - 1)
            
            # 3. Centralization bonus
            center_gain = abs(start_c - self.mid_col) - abs(end_c - self.mid_col)
            score += 100 * center_gain
            
            # 4. Position relative to other pieces
            player_positions = self.state.get_all_pos(self.state.current_player)
            if player_positions:
                avg_r = sum(r for r, _ in player_positions) / len(player_positions)
                r_diff = abs(end_r - avg_r)
                # Penalize spreading pieces too much
                score -= 50 * r_diff
            
            # 5. Endgame considerations
            if self.game_phase > 0.8:  # Late game
                # Strongly prefer moves that get pieces to the final row
                if (self.state.current_player == self.state.RED and end_r == BOARD_N - 1) or \
                   (self.state.current_player == self.state.BLUE and end_r == 0):
                    score += 2000

        return score

    def best_action(self):
        possible_moves = self.state.get_all_moves()
        
        # Score all moves
        scored_moves = [(move, self._score_move(move, res)) for move, res in possible_moves]
        
        # Get the best score
        best_score = max(score for _, score in scored_moves)
        
        # Filter moves within 80% of best score
        threshold = best_score * 0.8
        good_moves = [move for move, score in scored_moves if score >= threshold]
        
        # Randomly choose from good moves
        chosen_move = random.choice(good_moves)
        
        return {"action": chosen_move[0]}
