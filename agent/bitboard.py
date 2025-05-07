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
    # True bitboard constants - each cell is 2 bits (00 = EMPTY, 01 = LILLY, 10 = FROG, 11 = OPPONENT)
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
        if board is None:
            # Initialize a fresh bitboard
            self.bitboard = self._create_start_bitboard()
        elif isinstance(board, np.ndarray):
            # Convert numpy array to bitboard
            self.bitboard = self._convert_numpy_to_bitboard(board)
        else:
            # Assume it's already a bitboard (int array)
            self.bitboard = board.copy()

        self.current_player = self.FROG
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}
        self.ply_count = 0

    def _create_start_bitboard(self):
        """Creates the starting bitboard configuration"""
        # Create a board of all EMPTYs (each row is a 2N-bit integer)
        bitboard = [0] * BOARD_N

        # Set top row (FROG pieces - 10)
        for c in range(1, BOARD_N - 1):
            bitboard[0] |= self.FROG << (c * 2)

        # Set bottom row (OPPONENT pieces - 11)
        for c in range(1, BOARD_N - 1):
            bitboard[BOARD_N - 1] |= self.OPPONENT << (c * 2)

        # Set lily pads in row 1 and BOARD_N - 2
        for c in range(1, BOARD_N - 1):
            bitboard[1] |= self.LILLY << (c * 2)
            bitboard[BOARD_N - 2] |= self.LILLY << (c * 2)

        # Set corner lily pads
        bitboard[0] |= self.LILLY << 0  # Top left
        bitboard[0] |= self.LILLY << ((BOARD_N - 1) * 2)  # Top right
        bitboard[BOARD_N - 1] |= self.LILLY << 0  # Bottom left
        bitboard[BOARD_N - 1] |= self.LILLY << ((BOARD_N - 1) * 2)  # Bottom right

        return bitboard

    def _convert_numpy_to_bitboard(self, np_board):
        """Converts a numpy array board to a bitboard representation"""
        bitboard = [0] * BOARD_N
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                # Set 2 bits at the appropriate position
                bitboard[r] |= np_board[r][c] << (c * 2)
        return bitboard

    def _get_cell(self, r, c):
        """Gets the value of a cell at (r,c)"""
        if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
            return None
        # Extract 2 bits at position c
        return (self.bitboard[r] >> (c * 2)) & 0b11

    def _set_cell(self, r, c, value):
        """Sets the value of a cell at (r,c)"""
        if not (0 <= r < BOARD_N and 0 <= c < BOARD_N):
            return
        # Clear the 2 bits at position c
        self.bitboard[r] &= ~(0b11 << (c * 2))
        # Set the 2 bits to the new value
        self.bitboard[r] |= value << (c * 2)

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
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}

        # Check bottom row for FROG pieces
        row = self.bitboard[BOARD_N - 1]
        for c in range(BOARD_N):
            if ((row >> (c * 2)) & 0b11) == self.FROG:
                self.frog_border_count[self.FROG] += 1

        # Check top row for OPPONENT pieces
        row = self.bitboard[0]
        for c in range(BOARD_N):
            if ((row >> (c * 2)) & 0b11) == self.OPPONENT:
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
            board_copy = self.bitboard
        else:
            board_copy = self.bitboard.copy()

        fill = self.current_player

        if isinstance(action, MoveAction):
            fill = self._get_cell(action.coord.r, action.coord.c)

        if isinstance(action, GrowAction):
            # Handle GrowAction
            for pos in self.get_all_pos(fill):
                for direction in Direction:
                    next_c = pos.c + direction.value.c
                    next_r = pos.r + direction.value.r
                    if (
                        0 <= next_c < BOARD_N
                        and 0 <= next_r < BOARD_N
                        and self._get_cell(next_r, next_c) == self.EMPTY
                    ):
                        # Set the cell in the copied board
                        new_board = BitBoard(board_copy)
                        new_board._set_cell(next_r, next_c, self.LILLY)
                        board_copy = new_board.bitboard
        elif res is not None:
            # Handle move with specified result
            new_board = BitBoard(board_copy)
            new_board._set_cell(action.coord.r, action.coord.c, self.EMPTY)
            new_board._set_cell(res.r, res.c, fill)
            board_copy = new_board.bitboard
        else:
            # Handle move with computed result
            next_coord = action.coord
            for direction in action.directions:
                found_move = False
                while not found_move:
                    next_coord = next_coord + direction
                    if self._get_cell(next_coord.r, next_coord.c) == self.LILLY:
                        found_move = True

            new_board = BitBoard(board_copy)
            new_board._set_cell(action.coord.r, action.coord.c, self.EMPTY)
            new_board._set_cell(next_coord.r, next_coord.c, fill)
            board_copy = new_board.bitboard

        new_board = BitBoard(board_copy)
        new_board.current_player = self.current_player
        new_board.ply_count = self.ply_count + 1
        return new_board

    def get_all_pos(self, pos_type):
        """Find all positions of a given piece type"""
        out = []
        for r in range(BOARD_N):
            row = self.bitboard[r]
            for c in range(BOARD_N):
                # Extract cell value (2 bits)
                cell = (row >> (c * 2)) & 0b11
                if cell == pos_type:
                    out.append(Coord(r, c))
        return out

    def get_start_board(self):
        """Returns the starting board as a numpy array for compatibility"""
        bitboard = self._create_start_bitboard()
        board = np.zeros((BOARD_N, BOARD_N), dtype=int)

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                board[r][c] = (bitboard[r] >> (c * 2)) & 0b11

        return board

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
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if self._get_cell(r, c) == self.current_player:
                    possible_moves.extend(self.get_possible_move(Coord(r, c)))

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
                        if lazy_ret:
                            return possible_moves

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

                        if lazy_ret:
                            return possible_moves

                        # Create a new visited set for this path to avoid interference
                        new_visited = visited.copy()
                        new_visited.add((land_r, land_c))
                        stack.append((land_r, land_c, new_dirs, new_visited, True))

        return possible_moves

    def is_valid_move(self, move: MoveAction, forward: int = 1) -> bool:
        # Localize variables for speed
        r, c = move.coord.r, move.coord.c
        directions = move.directions
        n_dirs = len(directions)
        _get_cell = self._get_cell

        # Precompute a bitmask for "is frog or opponent?"
        foe_mask = self.FROG | self.OPPONENT

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
                if (cell & foe_mask) == 0:
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
            # Count lilies by iterating through each cell
            lily_count = 0
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if self._get_cell(r, c) == BitBoard.LILLY:
                        lily_count += 1

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
        temp = BitBoard(self.bitboard.copy())
        temp.current_player = me
        temp._set_cell(start_r, start_c, BitBoard.EMPTY)
        temp._set_cell(end_r, end_c, me)
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
        return 0
        # Available ways to double jump
        current_player = self.current_player
        player_skips = 0
        opponent_skips = 0
        players = [BitBoard.FROG, BitBoard.OPPONENT]
        opponent_player = [x for x in players if x != self.current_player].pop()

        # Use bit operations to quickly find all pieces and calculate skip options
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                cell = self._get_cell(r, c)
                if cell == current_player:
                    location = Coord(r, c)
                    player_skips += self.get_adjacent_leapfrog(location, current_player)
                elif cell == opponent_player:
                    location = Coord(r, c)
                    opponent_skips += self.get_adjacent_leapfrog(
                        location, opponent_player
                    )

        skip_advantage = scaled_sigmoid(player_skips - opponent_skips, input_range=5)

        # Calculate advancement level
        score = 0
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                cell = self._get_cell(r, c)
                if cell == self.current_player:
                    if self.current_player == BitBoard.FROG:
                        score += r
                    else:
                        score += BOARD_N - 1 - r
                elif cell == opponent_player:
                    if self.current_player == BitBoard.FROG:
                        score -= BOARD_N - 1 - r
                    else:
                        score -= r

        score = scaled_sigmoid(score, input_range=10)
        cluster_score = self.clustering_score()

        weighted_score = (5 * score + 3 * skip_advantage + cluster_score) / 9
        return 2 * weighted_score - 1

    def get_coordinates(self):
        """Get coordinates of all pieces for the current player"""
        coordinates = []
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if self._get_cell(r, c) == self.current_player:
                    coordinates.append((r, c))
        return coordinates

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
        h = [self.jump_heuristic(r, compressions) for r in range(BOARD_N)]

        # priority queue holds (f, counter, coord, bitboard)
        open_heap = []
        # make a fresh copy of self for the root
        root_bb = BitBoard(self.bitboard.copy())
        root_bb.current_player = self.current_player

        counter = itertools.count()

        heapq.heappush(open_heap, (h[start.r], next(counter), start, root_bb))

        came_from = {}
        g_score = {start: 0}

        while open_heap:
            f, _, current, bb = heapq.heappop(open_heap)
            if current.r == target:
                return self.reconstruct_path(came_from, current)

            # generate all hops *from this exact board*
            for action, neighbor in bb.get_possible_move(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = (current, action)
                    new_f = tentative_g + h[neighbor.r]

                    # make the hopped board
                    new_bb = bb.move(action, neighbor)
                    # preserve player so forward stays consistent
                    new_bb.current_player = bb.current_player

                    heapq.heappush(open_heap, (new_f, next(counter), neighbor, new_bb))

        return None

    def a_star(self, coord):
        """Legacy A* search function - keeping for compatibility"""
        start = coord

        # Finds all compressions for the board
        compression_dict = self.get_all_compressions()

        # Pre computes heuristic values
        heuristic_dict = {}
        for i in range(BOARD_N):
            heuristic_dict[i] = self.jump_heuristic(i, compression_dict)

        if start is None:
            return None

        closed_set = set()
        came_from = {}
        g_score = {start: 0}

        open_heap = []
        heapq.heappush(open_heap, (heuristic_dict[start.r], start))

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current.r == BOARD_N - 1:
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)
            moves = self.get_possible_move(current)

            for move, neighbor in moves:
                if neighbor in closed_set:
                    continue

                # Save the current state
                saved_bitboard = self.bitboard.copy()

                # Create a temporary board with the move applied
                tmp = BitBoard(saved_bitboard)
                tmp.current
