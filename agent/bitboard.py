import numpy as np
from referee.game.actions import MoveAction, GrowAction
from referee.game.coord import Coord, Direction
from referee.game.constants import BOARD_N
import math
import random


class BitBoard:
    # Cell type constants
    EMPTY = 0b00
    LILLY = 0b01
    RED = 0b10
    BLUE = 0b11

    # Direction constants
    DOWN = 1
    UP = -1
    LEFT = -1
    RIGHT = 1
    SAME = 0

    # Precompute (dr, dc) and Direction enum pairs once at import
    # Avoid expensive Direction enum access during move generation
    _OFFSETS = [(d.value.r, d.value.c, d) for d in Direction]

    # Precompute row masks for common operations
    # this allows us to get any row in O(1) time
    _ROW_MASKS = [0] * BOARD_N
    for r in range(BOARD_N):
        mask = 0
        for c in range(BOARD_N):
            mask |= 1 << (r * BOARD_N + c)
        _ROW_MASKS[r] = mask

    ILLEGAL_RED_DIRECTIONS = set([
        Direction.Up,
        Direction.UpRight,
        Direction.UpLeft,
    ])

    ILLEGAL_BLUE_DIRECTIONS = set([
        Direction.Down,
        Direction.DownRight,
        Direction.DownLeft,
    ])

    def __init__(self, board=None):
        # Initialize three 64-bit integers to represent the entire board
        # Each bit corresponds to one cell, with bit position = r*BOARD_N + c

        self.lilly_bits = 0  # Bits for lily pads
        self.frog_bits = 0  # Bits for frog pieces
        self.opp_bits = 0  # Bits for opponent pieces

        if board is None:
            # Initialize a fresh bitboard according to game spec
            self._create_start_bitboard()
        elif isinstance(board, np.ndarray):
            # Convert numpy array to bitboard
            self._convert_numpy_to_bitboard(board)
        else:
            # Assume it's already a bitboard as we know it
            self.lilly_bits = board[0]
            self.frog_bits = board[1]
            self.opp_bits = board[2]

        self.current_player = self.RED  # default current player
        self.frog_border_count = {self.RED: 0, self.BLUE: 0}
        self.ply_count = 0

        # Calculate occupied cells mask for faster move generation
        self._update_occupied_mask()

    def _update_occupied_mask(self):
        """Updates the mask of all occupied cells"""
        self.occupied_mask = (
            self.lilly_bits | self.frog_bits | self.opp_bits
        )  # just the OR of all three gives the full picture

    def _create_start_bitboard(self):
        """Creates the starting bitboard configuration using bit operations, and the configuration outlined in the spec"""
        # Start with empty boards
        self.lilly_bits = 0
        self.frog_bits = 0
        self.opp_bits = 0

        # Frog pieces
        row_mask = 0
        for c in range(1, BOARD_N - 1):  # Insert all the necessary frogs in row 0
            row_mask |= 1 << c
        self.frog_bits |= row_mask  # Top row is at r=0

        # Set bottom row (BLUE pieces)
        bottom_row_idx = (BOARD_N - 1) * BOARD_N
        self.opp_bits |= row_mask << bottom_row_idx

        # lilly pads in row 1 and BOARD_N - 2
        self.lilly_bits |= row_mask << BOARD_N  # Row 1
        self.lilly_bits |= row_mask << ((BOARD_N - 2) * BOARD_N)  # Row BOARD_N - 2

        # Corner lily pads
        self.lilly_bits |= 1
        self.lilly_bits |= 1 << (BOARD_N - 1)  # Top right
        self.lilly_bits |= 1 << ((BOARD_N - 1) * BOARD_N)  # Bottom left
        self.lilly_bits |= 1 << (
            (BOARD_N - 1) * BOARD_N + (BOARD_N - 1)
        )  # Bottom right

        # Update occupied cells mask
        self._update_occupied_mask()

    def _convert_numpy_to_bitboard(self, np_board):
        """Converts a given numpy 2-d array to our internal bitboard representation, used because the 2d array is much more human-readable"""
        self.lilly_bits = 0
        self.frog_bits = 0
        self.opp_bits = 0

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                pos = r * BOARD_N + c
                cell_value = np_board[r][c]

                if cell_value == self.LILLY:
                    self.lilly_bits |= 1 << pos
                elif cell_value == self.RED:
                    self.frog_bits |= 1 << pos
                elif cell_value == self.BLUE:
                    self.opp_bits |= 1 << pos

        # Update occupied cells mask
        self._update_occupied_mask()

    def bitboard(self):
        """Returns our internal representation of the bitboard"""
        return [self.lilly_bits, self.frog_bits, self.opp_bits]

    def _get_cell(self, r, c):
        """Return the cell type [lilly, empty, frog or opponent] at (r, c) using bit operations"""
        if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
            return None

        pos = r * BOARD_N + c
        bit_pos = 1 << pos

        if self.lilly_bits & bit_pos:
            return self.LILLY
        elif self.frog_bits & bit_pos:
            return self.RED
        elif self.opp_bits & bit_pos:
            return self.BLUE
        else:
            return self.EMPTY

    def _set_cell(self, r, c, value):
        """Sets the cell at (r,c) to value: `value` using bit operations"""
        if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
            return

        pos = r * BOARD_N + c
        bit_pos = 1 << pos

        # Clear the cell in all representations
        mask = ~bit_pos  # bitwise NOT, flips the bits
        self.lilly_bits &= mask
        self.frog_bits &= mask
        self.opp_bits &= mask

        # Set the new value
        if value == self.LILLY:
            self.lilly_bits |= bit_pos
        elif value == self.RED:
            self.frog_bits |= bit_pos
        elif value == self.BLUE:
            self.opp_bits |= bit_pos

        # Update occupied cells mask
        self._update_occupied_mask()

    def get_ply_count(self):
        """Ply count here is the number of turns played so far in the game"""
        return self.ply_count

    def get_board(self):
        """Returns the board as a 2D numpy array for compatibility"""
        board = np.zeros((BOARD_N, BOARD_N), dtype=int)

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                pos = r * BOARD_N + c
                bit_pos = 1 << pos

                if self.lilly_bits & bit_pos:
                    board[r][c] = self.LILLY
                elif self.frog_bits & bit_pos:
                    board[r][c] = self.RED
                elif self.opp_bits & bit_pos:
                    board[r][c] = self.BLUE

        return board

    def get_current_player(self):
        return self.current_player

    def toggle_player(self):
        """Toggles player in place for this bitboard"""
        self.current_player = self.BLUE if self.current_player == self.RED else self.RED

    def is_game_over(self):
        """Check if the game is over"""
        self.frog_border_count = {self.RED: 0, self.BLUE: 0}

        # Get bottom row mask
        bottom_row_mask = self._ROW_MASKS[BOARD_N - 1]

        # Count frogs in bottom row
        frogs_in_bottom = bin(self.frog_bits & bottom_row_mask).count(
            "1"
        )  # how many 1s are there ?
        self.frog_border_count[self.RED] = frogs_in_bottom  # frogs here means RED frogs

        # Get top row mask
        top_row_mask = self._ROW_MASKS[0]

        # Count opponents in top row using bit operations
        opps_in_top = bin(self.opp_bits & top_row_mask).count(
            "1"
        )  # blue frogs, all from pov of red for understanding
        self.frog_border_count[self.BLUE] = opps_in_top

        return (
            self.frog_border_count[self.RED] == BOARD_N - 2
            or self.frog_border_count[self.BLUE] == BOARD_N - 2
        )

    def get_winner(self):
        """Returns 1 if the current player won, -1 if the opponent won, and 0 if it's a draw"""
        if max(self.frog_border_count.values()) == BOARD_N - 2:
            if (
                self.frog_border_count[self.RED] == BOARD_N - 2
                and self.frog_border_count[self.BLUE] != BOARD_N - 2
            ):
                if self.current_player == self.RED:
                    return 1
                else:
                    return -1
            elif (
                self.frog_border_count[self.BLUE] == BOARD_N - 2
                and self.frog_border_count[self.RED] != BOARD_N - 2
            ):
                if self.current_player == self.BLUE:
                    return 1
                else:
                    return -1
            else:
                return 0
        return 0

    def move(self, action: MoveAction, res: Coord | None = None, in_place=False):
        """Moves a piece according to the action described in the action object combined with an optiional result coordinate for quicker calculation"""
        if in_place:
            new_board = self
        else:
            # Create a new BitBoard with the same state
            new_board = BitBoard()
            new_board.lilly_bits = self.lilly_bits
            new_board.frog_bits = self.frog_bits
            new_board.opp_bits = self.opp_bits
            new_board._update_occupied_mask()  # Update the occupied mask

        # Get the piece type to move
        if isinstance(action, MoveAction):
            # Directly compute bit position without using _get_cell
            start_pos = action.coord.r * BOARD_N + action.coord.c
            start_bit = 1 << start_pos

            # find out what type of piece we're working with here
            if self.frog_bits & start_bit:
                fill = self.RED
            elif self.opp_bits & start_bit:
                fill = self.BLUE
            else:
                fill = self.LILLY  # Fallback, should not happen
        else:
            fill = self.current_player

        if isinstance(action, GrowAction):
            # Handle GrowAction using bit operations
            pieces_to_check = (
                self.frog_bits if self.current_player == self.RED else self.opp_bits
            )  # assume grows happen from the perspective of the current player
            empty_cells = (
                ~self.occupied_mask
            )  # get all unoccupied cells using bitwise NOT

            # For each direction, shift the player pieces and get potential growth spots
            growth_spots = 0
            for dr, dc, _ in self._OFFSETS:
                # Calculate shift amount based on direction
                if dr == 0 and dc == 0:
                    continue  # Skip no movement

                shift = dr * BOARD_N + dc

                # Calculate potential growth spots (empty neighbors)
                if shift > 0:
                    shifted = pieces_to_check << shift
                else:
                    shifted = pieces_to_check >> abs(shift)

                # Mask to keep cells in bounds after shift
                if dc > 0:  # RIGHT shift - remove leftmost column
                    edge_mask = ~0
                    for r in range(BOARD_N):
                        edge_mask &= ~(1 << (r * BOARD_N))
                    shifted &= edge_mask
                elif dc < 0:  # LEFT shift - remove rightmost column
                    edge_mask = ~0
                    for r in range(BOARD_N):
                        edge_mask &= ~(1 << (r * BOARD_N + BOARD_N - 1))
                    shifted &= edge_mask

                # Add these spots to growth spots if they're empty
                growth_spots |= shifted & empty_cells

            # Set all growth spots to lily pads
            new_board.lilly_bits |= growth_spots
            new_board._update_occupied_mask()

        elif res is not None:
            # Handle move with specified result
            # Get the bit positions
            start_pos = action.coord.r * BOARD_N + action.coord.c
            end_pos = res.r * BOARD_N + res.c
            start_bit = 1 << start_pos
            end_bit = 1 << end_pos

            # Easy case, just move the piece leaving an empty spot
            if fill == self.RED:
                new_board.frog_bits &= ~start_bit
                new_board.frog_bits |= end_bit
            elif fill == self.BLUE:
                new_board.opp_bits &= ~start_bit
                new_board.opp_bits |= end_bit

            # Make sure destination is not a lily pad anymore
            new_board.lilly_bits &= ~end_bit
            new_board._update_occupied_mask()

        else:
            # Handle move with computed result
            start_r, start_c = action.coord.r, action.coord.c
            next_r, next_c = start_r, start_c

            # We don't know where this frog ended up, so we go in all the directions greedily until we find the resultant position
            for direction in action.directions:
                dr, dc = direction.value.r, direction.value.c
                found_move = False
                while not found_move:
                    next_r += dr
                    next_c += dc
                    pos = next_r * BOARD_N + next_c
                    bit_pos = 1 << pos

                    if self.lilly_bits & bit_pos:
                        found_move = True

            # Apply the move
            start_pos = start_r * BOARD_N + start_c
            end_pos = next_r * BOARD_N + next_c
            start_bit = 1 << start_pos
            end_bit = 1 << end_pos

            # Clear start position
            if fill == self.RED:
                new_board.frog_bits &= ~start_bit
                new_board.frog_bits |= end_bit
            elif fill == self.BLUE:
                new_board.opp_bits &= ~start_bit
                new_board.opp_bits |= end_bit

            new_board.lilly_bits &= ~end_bit
            new_board._update_occupied_mask()

        new_board.current_player = self.current_player
        new_board.ply_count = self.ply_count + 1
        return new_board

    def get_all_pos(self, pos_type):
        """Find all positions of a given piece type using bit operations"""
        out = []

        # Choose the correct bitboard based on piece type
        if pos_type == self.LILLY:
            bits = self.lilly_bits
        elif pos_type == self.RED:
            bits = self.frog_bits
        elif pos_type == self.BLUE:
            bits = self.opp_bits
        else:
            return out  # Empty not tracked directly

        # Directly iterate over set bits for better performance
        temp_bits = bits
        while temp_bits:
            # Extract least significant 1-bit
            lsb = temp_bits & -temp_bits
            # Calculate position index using fast bit counting
            pos_idx = lsb.bit_length() - 1
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
        call the function to get uncached moves
        """
        # Rebuild the numpy array in the right dtype & shape:
        arr = np.frombuffer(board_bytes, dtype=np.dtype(dtype_str)).copy()
        arr = arr.reshape((BOARD_N, BOARD_N))

        bb = BitBoard(arr)
        bb.current_player = player
        return bb._all_moves_uncached()

    def _all_moves_uncached(self):
        """Get all possible moves using iterative logic"""
        possible_moves = []

        # Use bit operations to efficiently find all pieces of current player
        player_positions = self.get_all_pos(self.current_player)

        for pos in player_positions:
            possible_moves.extend(self.get_possible_move(pos))

        # GrowAction is always possible
        possible_moves.append((GrowAction(), None))
        return possible_moves

    def get_random_move(self):
        """Get a random valid move lazily, for efficiency"""
        # Get current player's pieces efficiently
        if self.current_player == self.RED:
            player_bits = self.frog_bits
            forbidden_row = BOARD_N - 1
        else:
            player_bits = self.opp_bits
            forbidden_row = 0

        # Convert to positions list
        all_pos = []
        temp_bits = player_bits
        while temp_bits:
            lsb = temp_bits & -temp_bits
            pos_idx = lsb.bit_length() - 1
            r, c = pos_idx // BOARD_N, pos_idx % BOARD_N
            if r != forbidden_row:  # Skip positions in forbidden row
                all_pos.append(Coord(r, c))
            temp_bits &= ~lsb

        random.shuffle(all_pos)

        # Find the first position with at least one possible move
        while all_pos:
            rand_pos = all_pos.pop()
            possible_moves = self.get_possible_move(rand_pos, lazy_ret=True)
            if possible_moves:
                break
        else:
            # No moves found, return GrowAction
            return (GrowAction(), None)

        # Add GrowAction and choose randomly
        if rand_pos.r == forbidden_row:
            return possible_moves[
                0
            ]  # Try to avoid GrowAction loops when near the end of the game

        possible_moves.append((GrowAction(), None))
        return random.choices(possible_moves, k=1)[0]

    def get_all_moves(self):
        """Get all possible moves for the current player"""
        board_np = self.get_board()
        bts = board_np.tobytes()

        moves = BitBoard._cached_moves(bts, self.current_player, board_np.dtype.str)
        return moves.copy()

    def get_possible_move(
        self, coord: Coord, lazy_ret=False
    ) -> list[tuple[MoveAction, Coord]]:
        """
        Iterative move generation using a stack-based approach. If lazy_ret = True we just return the first move we come across for efficiency
        """
        possible_moves = []
        forward = 1 if self.current_player == self.RED else -1

        start_r, start_c = coord.r, coord.c

        stack = [(start_r, start_c, [], {(start_r, start_c)}, False)]

        # Pre-shuffle offsets for random first move
        if lazy_ret:
            offsets = list(self._OFFSETS)
            random.shuffle(offsets)
        else:
            offsets = self._OFFSETS

        while stack:
            r, c, path_dirs, visited, in_jump = stack.pop()

            for dr, dc, direction in offsets:
                # Single-step (only if not already jumping)
                if not in_jump:
                    # Directly check validity instead of calling is_valid_move
                    next_r, next_c = r + dr, c + dc

                    # Quick bounds and forward direction check
                    if (
                        0 <= next_r < BOARD_N
                        and 0 <= next_c < BOARD_N
                        and (dr == forward or dr == 0)
                    ):
                        # Fast check for lily pad at destination
                        next_pos = next_r * BOARD_N + next_c
                        if self.lilly_bits & (1 << next_pos):
                            move = MoveAction(Coord(r, c), [direction])
                            possible_moves.append((move, Coord(next_r, next_c)))

                            if lazy_ret:
                                return possible_moves

                # Attempt a jump
                mid_r, mid_c = r + dr, c + dc
                land_r, land_c = mid_r + dr, mid_c + dc

                # Compute bit positions
                mid_pos = (
                    mid_r * BOARD_N + mid_c
                    if 0 <= mid_r < BOARD_N and 0 <= mid_c < BOARD_N
                    else -1
                )
                land_pos = (
                    land_r * BOARD_N + land_c
                    if 0 <= land_r < BOARD_N and 0 <= land_c < BOARD_N
                    else -1
                )

                # Quick bounds checks
                if mid_pos == -1 or land_pos == -1 or (land_r, land_c) in visited:
                    continue

                # Check if mid cell has a piece (any player) and land cell is a lily pad
                mid_bit = 1 << mid_pos
                land_bit = 1 << land_pos

                if (
                    (self.frog_bits & mid_bit or self.opp_bits & mid_bit)
                    and (self.lilly_bits & land_bit)
                    and (dr == forward or dr == 0)  # Forward direction check
                ):
                    new_dirs = path_dirs + [direction]
                    action = MoveAction(coord, new_dirs)
                    possible_moves.append((action, Coord(land_r, land_c)))

                    if lazy_ret:
                        return possible_moves

                    # Add to stack for next jump
                    new_visited = visited.copy()
                    new_visited.add((land_r, land_c))
                    stack.append((land_r, land_c, new_dirs, new_visited, True))

        return possible_moves

    def is_valid_move(self, move: MoveAction, forward: int = 1):
        """
        Efficiently validates moves using bit operations
        """
        r, c = move.coord.r, move.coord.c
        directions = move.directions
        n_dirs = len(directions)

        # Process each direction
        for idx in range(n_dirs):
            d = directions[idx]
            dr, dc = d.value.r, d.value.c

            # Fast forward check, only forward moves are valid
            if dr != forward and dr != 0:
                return False

            # Compute new coordinates
            nr, nc = r + dr, c + dc

            # Bounds check
            if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
                return False

            # Get bit position
            cell_pos = nr * BOARD_N + nc
            cell_bit = 1 << cell_pos

            # Check cell type
            if idx < n_dirs - 1:
                # Intermediate hop must be a piece (frog or opp)
                if not ((self.frog_bits & cell_bit) or (self.opp_bits & cell_bit)):
                    return False
            else:
                # Final landing must be lily
                if not (self.lilly_bits & cell_bit):
                    return False

            # Update for next iteration
            r, c = nr, nc

        return True

    def _move_priority(self, move):
        """
        A heuristic for prioritizing moves when expanding
        """
        action, res = move

        # Get true board state
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
        delta = (end_r - start_r) if me == BitBoard.RED else (start_r - end_r)

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
        temp._update_occupied_mask()

        # Use bit operations to set cells efficiently
        start_pos = start_r * BOARD_N + start_c
        end_pos = end_r * BOARD_N + end_c
        start_bit = 1 << start_pos
        end_bit = 1 << end_pos

        # Clear the start position
        if me == self.RED:
            temp.frog_bits &= ~start_bit
            temp.frog_bits |= end_bit
        else:
            temp.opp_bits &= ~start_bit
            temp.opp_bits |= end_bit

        # Clear lily at destination
        temp.lilly_bits &= ~end_bit
        temp._update_occupied_mask()

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
        if self.current_player != BitBoard.RED:
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
                # Get state using bit operations
                pos = r * BOARD_N + c
                bit_pos = 1 << pos

                if self.lilly_bits & bit_pos:
                    text = "*"
                elif self.frog_bits & bit_pos:
                    text = "R"
                elif self.opp_bits & bit_pos:
                    text = "B"
                else:
                    text = "."

                output += apply_ansi(text, bold=False)
                output += " "
            output += "\n"
        return output

    def get_adjacent_leapfrog(self, coord: Coord, player):
        """
        Return a count of lily pad positions where a leap is possible from the given Coord.
        """
        adjacent_count = 0
        r, c = coord.r, coord.c

        for dr, dc, _ in self._OFFSETS:
            # Skip directions based on player
            if player == BitBoard.RED and dc <= 0:
                continue
            if player == BitBoard.BLUE and dc >= 0:
                continue

            # Calculate double jump position
            next_r = r + 2 * dr
            next_c = c + 2 * dc

            if 0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N:
                # Check if destination is a lily pad
                pos = next_r * BOARD_N + next_c
                if self.lilly_bits & (1 << pos):
                    adjacent_count += 1

        return adjacent_count

    def evaluate_position(self):
        """Heuristic function that evaluates a position"""
        # Available ways to double jump
        current_player = self.current_player
        player_skips = 0
        opponent_skips = 0
        players = [BitBoard.RED, BitBoard.BLUE]
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

        # For RED player - score is higher for pieces further down the board
        if self.current_player == BitBoard.RED:
            frog_pieces = self.get_all_pos(BitBoard.RED)
            for pos in frog_pieces:
                score += pos.r

            opponent_pieces = self.get_all_pos(BitBoard.BLUE)
            for pos in opponent_pieces:
                score -= BOARD_N - 1 - pos.r
        else:
            # For BLUE player - score is higher for pieces further up the board
            opponent_pieces = self.get_all_pos(BitBoard.BLUE)
            for pos in opponent_pieces:
                score += BOARD_N - 1 - pos.r

            frog_pieces = self.get_all_pos(BitBoard.RED)
            for pos in frog_pieces:
                score -= pos.r

        score = scaled_sigmoid(score, input_range=10)
        cluster_score = self.clustering_score()

        weighted_score = (5 * score + 3 * skip_advantage + cluster_score) / 9
        return 2 * weighted_score - 1

    def get_coordinates(self):
        coords = self.get_all_pos(self.current_player)
        return [(coord.r, coord.c) for coord in coords]

    def clustering_score(self, ideal_dist=2, sigma=0.5):
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


def scaled_sigmoid(x, input_range=10, output_range=(0, 1)):
    normalized = 1 / (1 + math.exp(-x * (2 / input_range)))
    lo, hi = output_range
    return lo + normalized * (hi - lo)
