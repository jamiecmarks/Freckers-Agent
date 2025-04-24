import numpy as np
from referee.game.actions import MoveAction, GrowAction
from referee.game.coord import Coord, Direction
from referee.game.constants import BOARD_N
import math
from scipy.spatial import ConvexHull
import numpy as np
from itertools import combinations
from math import exp

class BitBoard:
    LILLY = 0b01
    FROG = 0b10
    EMPTY = 0b00
    OPPONENT = 0b11

    def __init__(self, board=None):
        if board is None:
            board = self.get_start_board()
        self.board = board
        self.current_player = self.FROG
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}

    def get_board(self):
        return self.board

    def get_current_player(self):
        return self.current_player

    def toggle_player(self):
        self.current_player = (
            self.OPPONENT if self.current_player == self.FROG else self.FROG
        )

    def is_game_over(self):
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}
        for c in range(BOARD_N):
            if self.board[BOARD_N - 1][c] == self.FROG:
                self.frog_border_count[self.FROG] += 1
            elif self.board[0][c] == self.OPPONENT:
                self.frog_border_count[self.OPPONENT] += 1
        return (
            self.frog_border_count[self.FROG] == BOARD_N - 2
            or self.frog_border_count[self.OPPONENT] == BOARD_N - 2
        )

    def get_winner(self):
        if max(self.frog_border_count.values()) == BOARD_N - 2:
            if (
                self.frog_border_count[self.FROG] == BOARD_N - 2
                and self.frog_border_count[self.OPPONENT] != BOARD_N - 2
            ):
                return 1
            elif (
                self.frog_border_count[self.OPPONENT] == BOARD_N - 2
                and self.frog_border_count[self.FROG] != BOARD_N - 2
            ):
                return -1
            else:
                return 0
        return 0

    def move(self, action: MoveAction, res: Coord | None = None):
        board_copy = np.copy(self.board)
        fill = self.current_player

        if isinstance(action, MoveAction):
            fill = self.board[action.coord.r][action.coord.c]

        if isinstance(action, GrowAction):
            for pos in self.get_all_pos(fill):
                for direction in Direction:
                    next_c = pos.c + direction.c
                    next_r = pos.r + direction.r
                    if (
                        0 <= next_c < BOARD_N
                        and 0 <= next_r < BOARD_N
                        and board_copy[next_r][next_c] == self.EMPTY
                    ):
                        board_copy[next_r][next_c] = self.LILLY
        elif res is not None:
            board_copy[action.coord.r][action.coord.c] = self.EMPTY
            board_copy[res.r][res.c] = fill
        else:
            next_coord = action.coord
            for direction in action.directions:
                found_move = False
                while not found_move:
                    next_coord = next_coord + direction
                    if board_copy[next_coord.r][next_coord.c] == self.LILLY:
                        found_move = True
            board_copy[action.coord.r][action.coord.c] = self.EMPTY
            board_copy[next_coord.r][next_coord.c] = fill

        new_board = BitBoard(board_copy)
        new_board.current_player = self.current_player
        return new_board

    def get_all_pos(self, pos_type):
        out = []
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if self.board[r][c] == pos_type:
                    out.append(Coord(r, c))
        return out

    def get_all_moves(self):
        possible_moves = []
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if self.board[r][c] == self.current_player:
                    coord = Coord(r, c)
                    moves = self.get_possible_move(coord)
                    possible_moves.extend(moves)

        # GrowAction always an option
        possible_moves.append((GrowAction(), None))
        # cache the base list, but return a copy to callers
        return possible_moves.copy()

    def get_start_board(self):
        board = np.full((BOARD_N, BOARD_N), self.EMPTY, dtype=int)
        for r in [0, BOARD_N - 1]:
            for c in range(1, BOARD_N - 1):
                board[r][c] = self.FROG if r == 0 else self.OPPONENT
        for r in [1]:
            opp_r = BOARD_N - 1 - r
            for c in range(1, BOARD_N - 1):
                if board[r][c] == self.EMPTY:
                    board[r][c] = self.LILLY
                    board[opp_r][c] = self.LILLY
        for r in [0]:
            for c in [0, BOARD_N - 1]:
                board[r][c] = self.LILLY
                board[BOARD_N - 1 - r][c] = self.LILLY
        return board

    def get_possible_move(self, coord: Coord):
        visited = set()
        return self.get_possible_move_rec(coord, visited)

    def get_possible_move_rec(self, coord, visited, in_jump=False):
        possible_jumps = []
        visited.add(coord)
        forward = 1 if self.current_player == self.FROG else -1

        for direction in Direction:
            single_jump = MoveAction(coord, [direction])
            if self.is_valid_move(single_jump, forward) and not in_jump:
                possible_jumps.append((single_jump, coord + direction))

            try:
                first_jump = coord + direction
                double_jump_res = coord + direction + direction
            except ValueError:
                continue
            if (
                self.is_valid_move(MoveAction(first_jump, [direction]), forward)
                and (double_jump_res not in visited)
                and self.board[first_jump.r][first_jump.c] in [self.FROG, self.OPPONENT]
            ):
                double_jump = MoveAction(coord, [direction])
                possible_jumps.append((double_jump, double_jump_res))
                for jump, res in self.get_possible_move_rec(
                    double_jump_res, visited, True
                ):
                    possible_jumps.append(
                        (MoveAction(coord, [direction] + list(jump.directions)), res)
                    )

        return possible_jumps

    def is_valid_move(self, move: MoveAction, forward=1):
        coord = move.coord
        for i, direction in enumerate(move.directions):
            if direction.r not in (forward, 0):
                return False
            next_c = coord.c + direction.c
            next_r = coord.r + direction.r
            if next_c < 0 or next_c >= BOARD_N or next_r < 0 or next_r >= BOARD_N:
                return False
            if i < len(move.directions) - 1:
                # intermediate hop must jump over a piece
                if self.board[next_r][next_c] not in (self.FROG, self.OPPONENT):
                    return False
            else:
                # final landing cell must be a lily
                if self.board[next_r][next_c] != self.LILLY:
                    return False
            coord = Coord(next_r, next_c)
        return True

    def _move_priority(self, move):
        """
        Heuristic priority: forward moves first, then grow, then horizontal.
        """
        action, res = move
        # GROW action prioritized after forward moves
        if isinstance(action, GrowAction):
            return 0.5
        # compute vertical delta: forward (+), horizontal (0), backward (-)
        delta = res.r - action.coord.r
        # adjust sign if opponent
        if self.current_player != BitBoard.FROG:
            delta = -delta
        return delta


    def quick_eval(self, move, total_moves):
        if isinstance(move[0], GrowAction):
            if total_moves < 8:
                return ((8-total_moves))**2+1
            return 1
        start = move[0].coord.r
        end = move[1].r
        distance_covered  = start - end
        if self.current_player == BitBoard.FROG:
            distance_covered = -distance_covered

        return distance_covered


    
    
    def render(self) -> str:
        """
        Returns a visualisation of the game board as a multiline string, with
        optional ANSI color codes and Unicode characters (if applicable).
        """

        def apply_ansi(str, bold=True):
            bold_code = "\033[1m" if bold else ""
            color_code = ""
            if str == "R":
                color_code = "\033[31m"
            if str == "B":
                color_code = "\033[34m"
            if str == "*":
                color_code = "\033[32m"
            return f"{bold_code}{color_code}{str}\033[0m"

        board = self.get_board()
        output = ""
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                state = board[r][c]
                if state == self.LILLY:
                    text = "*"
                elif state == self.OPPONENT or state == self.FROG:
                    text = "B" if state == self.OPPONENT else "R"
                elif state == self.EMPTY:
                    text = "." 
                else:
                    text = " "
                output += apply_ansi(text,  bold=False)
                output += " "
            output += "\n"
        return output



    def get_adjacent_leapfrog(self, coord: Coord, player):
        """
        Return a list of Coord positions where a lily pad is adjacent to the given Coord.
        """
        adjacent = []
        for direction in Direction:
            if player == BitBoard.FROG:
                if direction.c<=0:
                    continue
            if player == BitBoard.OPPONENT:
                if direction.c>=0:
                    continue
                    
            next_r = coord.r + direction.r + direction.r
            next_c = coord.c + direction.c + direction.c
            if 0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N:
                if self.board[next_r][next_c] == BitBoard.LILLY:
                    adjacent.append(Coord(next_r, next_c))
        return len(adjacent)
    
    def evaluate_position(self, continuous = False):
        """ Heuristic function that figures out 
            whether a move is generally better for 
            a given side
        """

        # Available ways to double jump (more better for us)
        current_player = self.current_player
        player_skips = 0
        opponent_skips = 0
        board = self.get_board()
        players = [BitBoard.FROG, BitBoard.OPPONENT]
        opponent_player = [x for x in players if x!=self.current_player].pop()

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == current_player:
                    location = Coord(r, c)
                    player_skips += self.get_adjacent_leapfrog(location, current_player)
                elif board[r][c] == opponent_player:
                    location = Coord(r, c)
                    opponent_skips += self.get_adjacent_leapfrog(location, opponent_player)
        skip_advantage = scaled_sigmoid(player_skips-opponent_skips, input_range = 5)

        # Now we want advancement level:
        score = 0
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == self.current_player:
                    match self.current_player:
                        case BitBoard.FROG:
                            score += r
                            break
                        case BitBoard.OPPONENT:
                            score += BOARD_N - 1 - r
                            break
                elif board[r][c] == opponent_player:
                    match self.current_player:
                        case BitBoard.FROG:
                            score -= BOARD_N - 1 - r
                            break
                        case BitBoard.OPPONENT:
                            score -= r
                            break
                    score -= (
                        BOARD_N - 1 - r
                    )  # still an error here, not calculated coorectl
        score = scaled_sigmoid(score, input_range = 10)
        cluster_score = self.clustering_score()
        
        weighted_score = (2 * score + skip_advantage)/3
        # print(weighted_score)
        # print(self.render())
        if continuous:
            return weighted_score
        if weighted_score > 0.5:
            return 1
        return -1
        # Clustering level?


    def get_coordinates(self):
        board = self.get_board()
        coordinates = []
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == self.current_player:
                    coordinates.append((r,c))
        return coordinates

    def clustering_score(self, ideal_dist=2, sigma=0.5):
        coords = self.get_coordinates()
        # Calculate the centroid of the points (mean of coordinates)
        centroid = np.mean(coords, axis=0)
        # Calculate the Manhattan distance from each point to the centroid
        distances = [np.abs(np.array(p) - centroid).sum() for p in coords]
        avg_dist_to_centroid = np.mean(distances)
        # Gaussian penalty for the difference from the ideal distance
        penalty = ((avg_dist_to_centroid - ideal_dist) ** 2) / (2 * sigma ** 2)
        
        # Compute the score using the Gaussian function
        score = np.exp(-penalty)
        # Clamp the score to be between 0 and 1
        return max(0, min(1, score))





def scaled_sigmoid(x, input_range=10, output_range=(0, 1)):
    """
    Scales input `x` using a sigmoid function.
    
    Parameters:
    - x: The input value (can be any real number).
    - input_range: Controls how wide the input range is (e.g. 10 means -10 to 10 is the key range).
    - output_range: Tuple (min, max) to map the sigmoid output to a subrange like (0.2, 0.8).
    
    Returns:
    - A float in the specified output range (default is (0, 1)).
    """
    # Standard sigmoid centered at 0, scaled to input_range
    normalized = 1 / (1 + math.exp(-x * (2 / input_range)))

    # Map to custom output range if needed
    out_min, out_max = output_range
    return out_min + normalized * (out_max - out_min)


