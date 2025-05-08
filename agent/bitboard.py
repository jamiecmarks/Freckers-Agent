import numpy as np
from referee.game.actions import MoveAction, GrowAction
from referee.game.coord import Coord, Direction
from referee.game.constants import BOARD_N
import math
import heapq
import itertools
import random
from functools import lru_cache


class BitBoard:
    # Cell type constants - kept for interface compatibility
    EMPTY = 0b00
    LILLY = 0b01
    FROG = 0b10
    OPPONENT = 0b11

    # Direction constants
    DOWN = 1
    UP = -1
    LEFT = -1
    RIGHT = 1
    SAME = 0

    # Precompute (dr, dc) and Direction enum pairs once at import
    _OFFSETS: list[tuple[int, int, Direction]] = [
        (d.value.r, d.value.c, d) for d in Direction
    ]

    def __init__(self, board=None):
        # Initialize three 64-bit integers to represent the entire board
        # Each bit corresponds to one cell, with bit position = r*BOARD_N + c
        self.lilly_bits = 0  # Bits for lily pads
        self.frog_bits = 0  # Bits for frog pieces
        self.opp_bits = 0  # Bits for opponent pieces

        if board is None:
            # Initialize a fresh bitboard
            self._create_start_bitboard()
        elif isinstance(board, np.ndarray):
            # Convert numpy array to bitboard
            self._convert_numpy_to_bitboard(board)
        else:
            # Assume it's already a bitboard (copy from another BitBoard)
            self.lilly_bits = board[0]
            self.frog_bits = board[1]
            self.opp_bits = board[2]

        self.current_player = self.FROG
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}
        self.ply_count = 0

    def _create_start_bitboard(self):
        """Creates the starting bitboard configuration using bit operations"""
        # Start with empty board
        self.lilly_bits = 0
        self.frog_bits = 0
        self.opp_bits = 0

        # Set top row (FROG pieces)
        for c in range(1, BOARD_N - 1):
            self.frog_bits |= 1 << (0 * BOARD_N + c)

        # Set bottom row (OPPONENT pieces)
        for c in range(1, BOARD_N - 1):
            self.opp_bits |= 1 << ((BOARD_N - 1) * BOARD_N + c)

        # Set lily pads in row 1 and BOARD_N - 2
        for c in range(1, BOARD_N - 1):
            self.lilly_bits |= 1 << (1 * BOARD_N + c)
            self.lilly_bits |= 1 << ((BOARD_N - 2) * BOARD_N + c)

        # Set corner lily pads
        self.lilly_bits |= 1 << (0 * BOARD_N + 0)  # Top left
        self.lilly_bits |= 1 << (0 * BOARD_N + (BOARD_N - 1))  # Top right
        self.lilly_bits |= 1 << ((BOARD_N - 1) * BOARD_N + 0)  # Bottom left
        self.lilly_bits |= 1 << (
            (BOARD_N - 1) * BOARD_N + (BOARD_N - 1)
        )  # Bottom right

    def _convert_numpy_to_bitboard(self, np_board):
        """Converts a numpy array board to a bitboard representation"""
        self.lilly_bits = 0
        self.frog_bits = 0
        self.opp_bits = 0

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                pos = r * BOARD_N + c
                cell_value = np_board[r][c]

                if cell_value == self.LILLY:
                    self.lilly_bits |= 1 << pos
                elif cell_value == self.FROG:
                    self.frog_bits |= 1 << pos
                elif cell_value == self.OPPONENT:
                    self.opp_bits |= 1 << pos

    def bitboard(self):
        """Return a representation of the bitboard for compatibility with copy operations"""
        return [self.lilly_bits, self.frog_bits, self.opp_bits]

    def _get_cell(self, r, c):
        """Gets the value of a cell at (r,c) using bit operations"""
        if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
            return None

        pos = r * BOARD_N + c
        bit_pos = 1 << pos

        if self.lilly_bits & bit_pos:
            return self.LILLY
        elif self.frog_bits & bit_pos:
            return self.FROG
        elif self.opp_bits & bit_pos:
            return self.OPPONENT
        else:
            return self.EMPTY

    def _set_cell(self, r, c, value):
        """Sets the value of a cell at (r,c) using bit operations"""
        if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
            return

        pos = r * BOARD_N + c
        bit_pos = 1 << pos

        # Clear the cell first
        self.lilly_bits &= ~bit_pos
        self.frog_bits &= ~bit_pos
        self.opp_bits &= ~bit_pos

        # Set the new value
        if value == self.LILLY:
            self.lilly_bits |= bit_pos
        elif value == self.FROG:
            self.frog_bits |= bit_pos
        elif value == self.OPPONENT:
            self.opp_bits |= bit_pos
        # EMPTY case: all bits already cleared

    def get_ply_count(self):
        return self.ply_count

    def get_board(self):
        """Returns the board as a 2D numpy array for compatibility"""
        board = np.zeros((BOARD_N, BOARD_N), dtype=int)
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                board[r][c] = self._get_cell(r, c)
        return board

    def get_current_player(self):
        return self.current_player

    def toggle_player(self):
        self.current_player = (
            self.OPPONENT if self.current_player == self.FROG else self.FROG
        )

    def is_game_over(self):
        """Check if the game is over - using bit operations for efficiency"""
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}

        # Check bottom row for FROG pieces
        bottom_row_mask = 0
        for c in range(BOARD_N):
            bottom_row_mask |= 1 << ((BOARD_N - 1) * BOARD_N + c)

        # Count frogs in bottom row
        frogs_in_bottom = bin(self.frog_bits & bottom_row_mask).count("1")
        self.frog_border_count[self.FROG] = frogs_in_bottom

        # Check top row for OPPONENT pieces
        top_row_mask = 0
        for c in range(BOARD_N):
            top_row_mask |= 1 << (0 * BOARD_N + c)

        # Count opponents in top row
        opps_in_top = bin(self.opp_bits & top_row_mask).count("1")
        self.frog_border_count[self.OPPONENT] = opps_in_top

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
                if self.current_player == self.FROG:
                    return 1
                else:
                    return -1
            elif (
                self.frog_border_count[self.OPPONENT] == BOARD_N - 2
                and self.frog_border_count[self.FROG] != BOARD_N - 2
            ):
                if self.current_player == self.OPPONENT:
                    return 1
                else:
                    return -1
            else:
                return 0
        return 0

    def move(self, action: MoveAction, res: Coord | None = None, in_place=False):
        if in_place:
            new_board = self
        else:
            # Create a new BitBoard with the same state
            new_board = BitBoard()
            new_board.lilly_bits = self.lilly_bits
            new_board.frog_bits = self.frog_bits
            new_board.opp_bits = self.opp_bits

        fill = self.current_player

        if isinstance(action, MoveAction):
            fill = self._get_cell(action.coord.r, action.coord.c)

        if isinstance(action, GrowAction):
            # Handle GrowAction - add lily pads next to current pieces
            for pos in self.get_all_pos(fill):
                for direction in Direction:
                    next_c = pos.c + direction.value.c
                    next_r = pos.r + direction.value.r
                    if (
                        0 <= next_c < BOARD_N
                        and 0 <= next_r < BOARD_N
                        and self._get_cell(next_r, next_c) == self.EMPTY
                    ):
                        # Set the cell to lily pad
                        new_board._set_cell(next_r, next_c, self.LILLY)
        elif res is not None:
            # Handle move with specified result
            new_board._set_cell(action.coord.r, action.coord.c, self.EMPTY)
            new_board._set_cell(res.r, res.c, fill)
        else:
            # Handle move with computed result
            next_coord = action.coord
            for direction in action.directions:
                found_move = False
                while not found_move:
                    next_coord = next_coord + direction
                    if self._get_cell(next_coord.r, next_coord.c) == self.LILLY:
                        found_move = True

            new_board._set_cell(action.coord.r, action.coord.c, self.EMPTY)
            new_board._set_cell(next_coord.r, next_coord.c, fill)

        new_board.current_player = self.current_player
        new_board.ply_count = self.ply_count + 1
        return new_board

    def get_all_pos(self, pos_type):
        """Find all positions of a given piece type using bit operations"""
        out = []
        # Choose the correct bitboard based on piece type
        if pos_type == self.LILLY:
            bits = self.lilly_bits
        elif pos_type == self.FROG:
            bits = self.frog_bits
        elif pos_type == self.OPPONENT:
            bits = self.opp_bits
        else:
            return out  # Empty not tracked directly

        # Iterate through all set bits
        temp_bits = bits
        while temp_bits:
            # Extract least significant 1-bit
            lsb = temp_bits & -temp_bits
            # Calculate position index
            pos_idx = bin(lsb).count("0") - 1
            # Convert to row, col
            r, c = pos_idx // BOARD_N, pos_idx % BOARD_N
            out.append(Coord(r, c))
            # Clear the bit
            temp_bits &= ~lsb

        return out

    def get_start_board(self):
        """Returns the starting board as a numpy array for compatibility"""
        # Create a temporary board with start position
        temp_board = BitBoard()
        temp_board._create_start_bitboard()
        return temp_board.get_board()

    @staticmethod
    def _cached_moves(board_bytes: bytes, player: int, dtype_str: str):
        """
        Reconstruct a bitboard from its raw bytes + player, then
        call the *uncached* move generator.
        """
        # rebuild the numpy array in the right dtype & shape:
        arr = np.frombuffer(board_bytes, dtype=np.dtype(dtype_str)).copy()
        arr = arr.reshape((BOARD_N, BOARD_N))

        bb = BitBoard(arr)
        bb.current_player = player
        return bb._all_moves_uncached()

    def _all_moves_uncached(self):
        """Get all possible moves (uncached)"""
        possible_moves = []

        # Use bit operations to efficiently find all pieces of current player
        player_positions = self.get_all_pos(self.current_player)

        for pos in player_positions:
            possible_moves.extend(self.get_possible_move(pos))

        possible_moves.append((GrowAction(), None))
        return possible_moves

    def get_random_move(self):
        """Get a random valid move"""
        all_pos = self.get_all_pos(self.current_player)

        random.shuffle(all_pos)

        # find the first position with at least one possible move
        while True:
            if not all_pos:
                return (GrowAction(), None)

            rand_pos = all_pos.pop()
            if rand_pos.r == 0 and self.current_player == self.OPPONENT:
                continue
            if rand_pos.r == BOARD_N - 1 and self.current_player == self.FROG:
                continue

            possible_moves = self.get_possible_move(rand_pos, lazy_ret=True)

            if possible_moves:
                break

        possible_moves.append((GrowAction(), None))
        return random.choice(possible_moves)

    def get_all_moves(self):
        """Get all possible moves for the current player"""
        # Convert to numpy array for compatibility with the original function
        board_np = self.get_board()
        bts = board_np.tobytes()

        moves = BitBoard._cached_moves(bts, self.current_player, board_np.dtype.str)
        return moves.copy()

    def get_possible_move(
        self, coord: Coord, lazy_ret=False
    ) -> list[tuple[MoveAction, Coord]]:
        """
        Move-generation using bit operations for faster processing
        """
        possible_moves: list[tuple[MoveAction, Coord]] = []
        forward = 1 if self.current_player == self.FROG else -1

        start_r, start_c = coord.r, coord.c
        # Each stack frame: (r, c, path_dirs, visited_set, in_jump)
        stack: list[tuple[int, int, list[Direction], set[tuple[int, int]], bool]] = [
            (start_r, start_c, [], {(start_r, start_c)}, False)
        ]

        # Localize references for speed
        _get_cell = self._get_cell
        is_valid = self.is_valid_move

        if lazy_ret:
            random.shuffle(self._OFFSETS)  # so that the first move is random

        while stack:
            r, c, path_dirs, visited, in_jump = stack.pop()
            for dr, dc, direction in self._OFFSETS:
                # Single-step (only if not already jumping)
                if not in_jump:
                    move = MoveAction(Coord(r, c), [direction])
                    if is_valid(move, forward):
                        dest_r, dest_c = r + dr, c + dc
                        possible_moves.append((move, Coord(dest_r, dest_c)))

                # Attempt a jump
                mid_r, mid_c = r + dr, c + dc
                land_r, land_c = mid_r + dr, mid_c + dc

                # Bounds + occupied + not yet visited
                if (
                    0 <= mid_r < BOARD_N
                    and 0 <= mid_c < BOARD_N
                    and 0 <= land_r < BOARD_N
                    and 0 <= land_c < BOARD_N
                    and _get_cell(mid_r, mid_c) in (self.FROG, self.OPPONENT)
                    and (land_r, land_c) not in visited
                ):
                    # Validate hop
                    hop_check = MoveAction(Coord(mid_r, mid_c), [direction])
                    if is_valid(hop_check, forward):
                        new_dirs = path_dirs + [direction]
                        action = MoveAction(coord, new_dirs)
                        possible_moves.append((action, Coord(land_r, land_c)))

                        # Create a new visited set for this path to avoid interference
                        new_visited = visited.copy()
                        new_visited.add((land_r, land_c))
                        stack.append((land_r, land_c, new_dirs, new_visited, True))

                if possible_moves and lazy_ret:
                    return possible_moves  # get the first move

        return possible_moves

    def is_valid_move(self, move: MoveAction, forward: int = 1) -> bool:
        # Localize variables for speed
        r, c = move.coord.r, move.coord.c
        directions = move.directions
        n_dirs = len(directions)
        _get_cell = self._get_cell

        # Loop without enumerate for better performance
        for idx in range(n_dirs):
            d = directions[idx]
            dr, dc = d.value.r, d.value.c

            # Fast forward check
            if dr != forward and dr != 0:
                return False

            # Compute new row/col
            nr = r + dr
            nc = c + dc

            # Single bounds check
            if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
                return False

            cell = _get_cell(nr, nc)

            # Intermediate or final check
            if idx < n_dirs - 1:
                # must be a piece (frog or opp)
                if cell != self.FROG and cell != self.OPPONENT:
                    return False
            else:
                # landing must be lily
                if cell != self.LILLY:
                    return False

            # Advance for the next iteration
            r, c = nr, nc

        return True

    def _move_priority(self, move):
        """
        A heuristic for prioritizing moves when expanding
        """
        action, res = move

        # Get true board state
        board = self.get_board()  # Convert to numpy for heuristic calculation
        me = self.current_player
        mid = (BOARD_N - 1) / 2

        # 1) GrowAction: more valuable the fewer lilies on board
        if isinstance(action, GrowAction):
            # Count lilies using bit operations
            lily_count = bin(self.lilly_bits).count("1")

            # scale [0..1]: when lily_count=0 → 1.0, when full → 0.0
            grow_score = 1.0 - (lily_count / (BOARD_N * BOARD_N))
            # give grow a solid boost early
            return 2.0 * grow_score

        # 2) it's a MoveAction: compute forward distance
        start_r, start_c = action.coord.r, action.coord.c
        end_r, end_c = res.r, res.c
        # forward delta (frogs down, opponent up)
        delta = (end_r - start_r) if me == BitBoard.FROG else (start_r - end_r)

        # 3) multi-jump bonus
        num_hops = len(action.directions)
        jump_bonus = 0.3 * (num_hops - 1)  # 0 hops → 0, double‐jump → +0.3, etc.

        # 4) centralization bonus: how much closer to mid column?
        center_gain = abs(start_c - mid) - abs(end_c - mid)
        center_bonus = center_gain / BOARD_N  # normalize

        # 5) blocking penalty: if opponent can jump far from landing
        temp = BitBoard()
        temp.lilly_bits = self.lilly_bits
        temp.frog_bits = self.frog_bits
        temp.opp_bits = self.opp_bits
        temp.current_player = me

        # Use bit operations to set cells efficiently
        start_pos = start_r * BOARD_N + start_c
        end_pos = end_r * BOARD_N + end_c

        # Clear the start position
        if me == self.FROG:
            temp.frog_bits &= ~(1 << start_pos)
        else:
            temp.opp_bits &= ~(1 << start_pos)

        # Set the end position
        if me == self.FROG:
            temp.frog_bits |= 1 << end_pos
        else:
            temp.opp_bits |= 1 << end_pos

        temp.toggle_player()

        opp_max = 0
        for mv, mv_res in temp.get_all_moves():
            if isinstance(mv, MoveAction):
                dist = abs(mv_res.r - mv.coord.r)
                opp_max = max(opp_max, dist)
        block_penalty = 0.1 * opp_max  # more opp options → worse

        # combine with weights
        priority = 1.0 * delta + jump_bonus + center_bonus - block_penalty
        return priority

    def old_move_priority(self, move):
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

        output = ""
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                state = self._get_cell(r, c)
                if state == self.LILLY:
                    text = "*"
                elif state == self.OPPONENT or state == self.FROG:
                    text = "B" if state == self.OPPONENT else "R"
                elif state == self.EMPTY:
                    text = "."
                else:
                    text = " "
                output += apply_ansi(text, bold=False)
                output += " "
            output += "\n"
        return output

    def get_adjacent_leapfrog(self, coord: Coord, player):
        """
        Return a count of lily pad positions where a leap is possible from the given Coord.
        """
        adjacent_count = 0
        for direction in Direction:
            if player == BitBoard.FROG:
                if direction.value.c <= 0:
                    continue
            if player == BitBoard.OPPONENT:
                if direction.value.c >= 0:
                    continue

            next_r = coord.r + direction.value.r + direction.value.r
            next_c = coord.c + direction.value.c + direction.value.c
            if 0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N:
                if self._get_cell(next_r, next_c) == BitBoard.LILLY:
                    adjacent_count += 1

        return adjacent_count

    def evaluate_position(self):
        """Heuristic function that evaluates a position"""
        # Available ways to double jump
        current_player = self.current_player
        player_skips = 0
        opponent_skips = 0
        players = [BitBoard.FROG, BitBoard.OPPONENT]
        opponent_player = [x for x in players if x != self.current_player].pop()

        # Get positions for current player and opponent
        current_player_positions = self.get_all_pos(current_player)
        opponent_player_positions = self.get_all_pos(opponent_player)

        # Calculate skip options
        for pos in current_player_positions:
            player_skips += self.get_adjacent_leapfrog(pos, current_player)

        for pos in opponent_player_positions:
            opponent_skips += self.get_adjacent_leapfrog(pos, opponent_player)

        skip_advantage = scaled_sigmoid(player_skips - opponent_skips, input_range=5)

        # Calculate advancement level using bit operations
        score = 0

        # For FROG player - score is higher for pieces further down the board
        if self.current_player == BitBoard.FROG:
            frog_pieces = self.get_all_pos(BitBoard.FROG)
            for pos in frog_pieces:
                score += pos.r

            opponent_pieces = self.get_all_pos(BitBoard.OPPONENT)
            for pos in opponent_pieces:
                score -= BOARD_N - 1 - pos.r
        else:
            # For OPPONENT player - score is higher for pieces further up the board
            opponent_pieces = self.get_all_pos(BitBoard.OPPONENT)
            for pos in opponent_pieces:
                score += BOARD_N - 1 - pos.r

            frog_pieces = self.get_all_pos(BitBoard.FROG)
            for pos in frog_pieces:
                score -= pos.r

        score = scaled_sigmoid(score, input_range=10)
        cluster_score = self.clustering_score()

        weighted_score = (5 * score + 3 * skip_advantage + cluster_score) / 9
        return 2 * weighted_score - 1

    def get_coordinates(self):
        """Get coordinates of all pieces for the current player"""
        coords = self.get_all_pos(self.current_player)
        return [(coord.r, coord.c) for coord in coords]

    def clustering_score(self, ideal_dist=2, sigma=0.5):
        """Calculate how well the player's pieces are clustered"""
        coords = self.get_coordinates()

        # Calculate the centroid of the points
        centroid = np.mean(coords, axis=0)

        # Calculate the Manhattan distance from each point to the centroid
        distances = [np.abs(np.array(p) - centroid).sum() for p in coords]
        avg_dist_to_centroid = np.mean(distances)

        # Gaussian penalty for the difference from the ideal distance
        penalty = ((avg_dist_to_centroid - ideal_dist) ** 2) / (2 * sigma**2)

        # Compute the score using the Gaussian function
        score = np.exp(-penalty)

        # Clamp the score to be between 0 and 1
        return max(0, min(1, score))

    def dijkstra_algorithm(self, compressions, start_row):
        """
        Performs Dijkstra's algorithm search on the simplified representation,
        to return a heuristic.
        """
        # Defines the winning condition
        if self.current_player == BitBoard.FROG:
            target = BOARD_N - 1
        else:
            target = 0

        # Initializes dijkstra, assuming all distances are inf initially
        distances = {i: float("inf") for i in range(start_row, BOARD_N)}
        distances[start_row] = 0
        queue = [(0, start_row)]

        while queue:
            # Explore the most promising search node
            cost, current = heapq.heappop(queue)

            # If target is at the frontier, we found shortest path
            if current == target:
                return cost

            next_row = current + 1
            if next_row < BOARD_N and cost + 1 < distances[next_row]:
                distances[next_row] = cost + 1
                heapq.heappush(queue, (cost + 1, next_row))

            # Update graph based on any available compressions
            if current in compressions:
                for jump in compressions[current]:
                    next_row = current + jump
                    if next_row < BOARD_N and cost + 1 < distances[next_row]:
                        distances[next_row] = cost + 1
                        heapq.heappush(queue, (cost + 1, next_row))

        # Return infinity if solution not possible
        return float("inf")

    def get_all_optimal_moves(self):
        """Get all optimal moves using A* search for each piece"""
        all_moves = []
        for coord in self.get_all_pos(self.current_player):
            move_set = self.a_star_new(coord)
            if move_set:
                all_moves.append(
                    move_set[0]
                )  # just need the first move in the optimal seq

        all_moves.append((GrowAction(), None))
        return all_moves

    def a_star_new(self, coord):
        """A* search algorithm for finding optimal path to goal"""
        start = coord
        if self.current_player == BitBoard.FROG:
            target = BOARD_N - 1
        else:
            target = 0

        # precompute heuristics per row
        compressions = self.get_all_compressions()
        # h = [self.jump_h


def scaled_sigmoid(x, input_range=10, output_range=(0, 1)):
    normalized = 1 / (1 + math.exp(-x * (2 / input_range)))
    lo, hi = output_range
    return lo + normalized * (hi - lo)
