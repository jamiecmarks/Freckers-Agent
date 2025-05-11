import numpy as np
from referee.game.actions import MoveAction, GrowAction
from referee.game.coord import Coord, Direction
from referee.game.constants import BOARD_N
import math
import numpy as np
import heapq
import itertools


class BitBoard:
    LILLY = 0b01
    FROG = 0b10
    EMPTY = 0b00
    OPPONENT = 0b11

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
            board = self.get_start_board()
        self.board = board
        self.current_player = self.FROG
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}
        self.ply_count = 0

    def get_ply_count(self):
        return self.ply_count

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
            if self.board[0][c] == self.OPPONENT:
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
            board_copy = self.board
        else:
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
        new_board.ply_count = self.ply_count + 1
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
                    # print(coord)
                    # for move, res in moves:
                    #     print(move, res)
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

    # def get_possible_move(self, coord: Coord):
    #     visited = set()
    #     return self.get_possible_move_rec(coord, visited)





    def get_possible_move(self, coord: Coord) -> list[tuple[MoveAction, Coord]]:
        """
        Move-generation using primitive ints and per-path visited sets to avoid shared-state bugs.
        """
        possible_moves: list[tuple[MoveAction, Coord]] = []
        forward = 1 if self.current_player == self.FROG else -1

        start_r, start_c = coord.r, coord.c
        # Each stack frame: (r, c, path_dirs, visited_set, in_jump)
        stack: list[tuple[int, int, list[Direction], set[tuple[int, int]], bool]] = [
            (start_r, start_c, [], {(start_r, start_c)}, False)
        ]

        # Localize attributes for speed
        board = self.board
        is_valid = self.is_valid_move
        F, O = self.FROG, self.OPPONENT
        N = BOARD_N

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
                    0 <= mid_r < N
                    and 0 <= mid_c < N
                    and 0 <= land_r < N
                    and 0 <= land_c < N
                    and board[mid_r][mid_c] in (F, O)
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

        return possible_moves

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

    # has been optimized

    def is_valid_move(self, move: MoveAction, forward: int = 1) -> bool:
        # 1) Localize everything
        r, c = move.coord.r, move.coord.c
        directions = move.directions
        n_dirs = len(directions)
        board = self.board
        FROG, OPP, LILLY = self.FROG, self.OPPONENT, self.LILLY
        bn = BOARD_N

        # 2) Precompute a bitmask for “is frog or opponent?”
        #    (if your enums are bit‐flags, else omit this)
        foe_mask = FROG | OPP

        # 3) Loop without enumerate(), avoid tuple unpacking & attribute lookups
        for idx in range(n_dirs):
            d = directions[idx]
            dr, dc = d.r, d.c

            # 4) Fast forward check
            if dr != forward and dr != 0:
                return False

            # 5) Compute new row/col
            nr = r + dr
            nc = c + dc

            # 6) Single bounds check
            if not (0 <= nr < bn and 0 <= nc < bn):
                return False

            cell = board[nr][nc]

            # 7) Intermediate or final check
            if idx < n_dirs - 1:
                # must be a piece (frog or opp)
                # using bitmask test is slightly faster than "cell != A and cell != B"
                if (cell & foe_mask) == 0:
                    return False
            else:
                # landing must be lily
                if cell != LILLY:
                    return False

            # 8) Advance for the next iteration
            r, c = nr, nc

        return True

    def _move_priority(self, move):
        """
        A stronger heuristic for prioritizing moves when expanding:
        - Grow actions early get top billing
        - Forward-move distance, with extra for multi-jumps
        - Bonus for moving toward center columns
        - Small penalty if landing spot lets your opponent have big jumps
        Returns a float: higher == better.
        """
        action, res = move
        board = self.board
        me = self.current_player
        mid = (BOARD_N - 1) / 2

        # 1) GrowAction: more valuable the fewer lilies on board
        if isinstance(action, GrowAction):
            lily_count = (board == BitBoard.LILLY).sum()
            # scale [0..1]: when lily_count=0 → 1.0, when full → 0.0
            grow_score = 1.0 - (lily_count / (BOARD_N * BOARD_N))
            # give grow a solid boost early (you could even taper by turn)
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

        # 5) blocking penalty: if the opponent on next turn can jump far from this landing
        #    quick approximation: count their max hop from this cell
        #    (you can comment this out if it's too slow)
        temp = BitBoard(board.copy())
        temp.current_player = me
        temp.board[start_r][start_c] = BitBoard.EMPTY
        temp.board[end_r][end_c] = me
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
                output += apply_ansi(text, bold=False)
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
                if direction.c <= 0:
                    continue
            if player == BitBoard.OPPONENT:
                if direction.c >= 0:
                    continue

            next_r = coord.r + direction.r + direction.r
            next_c = coord.c + direction.c + direction.c
            if 0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N:
                if self.board[next_r][next_c] == BitBoard.LILLY:
                    adjacent.append(Coord(next_r, next_c))
        return len(adjacent)

    def evaluate_position(self, simple = True):
        """Heuristic function that figures out
        whether a move is generally better for
        a given side
        """

        # Available ways to double jump (more better for us)
        current_player = self.current_player
        player_skips = 0
        opponent_skips = 0
        board = self.get_board()
        players = [BitBoard.FROG, BitBoard.OPPONENT]
        opponent_player = [x for x in players if x != self.current_player].pop()

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == current_player:
                    location = Coord(r, c)
                    player_skips += self.get_adjacent_leapfrog(location, current_player)
                elif board[r][c] == opponent_player:
                    location = Coord(r, c)
                    opponent_skips += self.get_adjacent_leapfrog(
                        location, opponent_player
                    )
        skip_advantage = scaled_sigmoid(player_skips - opponent_skips, input_range=5)

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
        if simple:
            return score/64

        score = scaled_sigmoid(score, input_range=10)
        cluster_score = self.clustering_score()

        weighted_score = (5 * score + 3 * skip_advantage + cluster_score) / 9
        if weighted_score < 0.1:
            return 0.1
        if weighted_score > 0.9:
             return 0.9
        return weighted_score
        # print(weighted_score)
        # print(self.render())
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
                    coordinates.append((r, c))
        return coordinates

    def clustering_score(self, ideal_dist=2, sigma=0.5):
        coords = self.get_coordinates()
        # Calculate the centroid of the points (mean of coordinates)
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
        Performs Dijkstra's algorithm (or uniform cost search in this case)
        search on the simplified representation of the problem, to return a heuristic.
        Using Dijkstra allows for the option of non-uniform costs in the future.
        """

        # Defines the winning condition
        if self.current_player == BitBoard.FROG:
            target = BOARD_N - 1
        else:
            target = 0

        # Initialises dijkstra, assuming all distances are inf initially
        distances = {i: float("inf") for i in range(start_row, BOARD_N)}
        distances[start_row] = 0
        queue = [(0, start_row)]

        while queue:
            # Explore the most promising search node
            cost, current = heapq.heappop(queue)

            # If target is at the frontier, we must have found shortest the path to it
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

        # Return infinity if solution not possible (although technically always should be)
        return float("inf")

    def get_all_optimal_moves(self):
        all_moves = []
        for coord in self.get_all_pos(self.current_player):
            # print(coord)
            move_set = self.a_star_new(coord)
            if move_set:
                all_moves.append(
                    move_set[0]
                )  # just need the first move in the optimal seq
                # print(coord)
                # for move, res in move_set:
                #     print(move, res)
        all_moves.append((GrowAction(), None))

        return all_moves

    def a_star_new(self, coord):
        start = coord
        if self.current_player == BitBoard.FROG:
            target = BOARD_N - 1
        else:
            target = 0

        # precompute heuristics per row
        compressions = self.get_all_compressions()
        h = [self.jump_heuristic(r, compressions) for r in range(BOARD_N)]

        # priority queue holds (f, coord, bitboard)
        open_heap = []
        # make a fresh copy of self for the root
        root_bb = BitBoard(self.board.copy())
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
                    # preserve the same player so forward stays 1
                    new_bb.current_player = bb.current_player

                    heapq.heappush(open_heap, (new_f, next(counter), neighbor, new_bb))

        return None

    def a_star(self, coord):
        """The main A* search function.
        It uses a priority queue along with the informed
        heuristic above to find the optimal path to the goal."""
        start = coord

        # Finds all compressions for the board
        compression_dict = self.get_all_compressions()

        # Pre computes heuristic values so we don't repeatedly calculate them
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

        # --------------- Search logic -------------#
        while open_heap:
            # don't need f_score, just for the heap
            _, current = heapq.heappop(open_heap)

            # If we are
            if current.r == BOARD_N - 1:
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            moves = self.get_possible_move(current)

            for move, neighbor in moves:
                if neighbor in closed_set:
                    continue

                saved_board = self.board
                tmp = BitBoard(saved_board.copy())
                # tmp = tmp.move(move, neighbor)
                tmp.current_player = self.current_player
                tmp.board[current.r][current.c] = BitBoard.EMPTY
                tmp.board[neighbor.r][neighbor.c] = self.current_player

                # 2) swap it in so get_possible_move() will see the frog up at `neighbor`
                self.board = tmp.board

                possible_g_score = g_score[current] + 1  # each move costs 1

                if neighbor not in g_score or possible_g_score < g_score[neighbor]:
                    came_from[neighbor] = (current, move)
                    g_score[neighbor] = possible_g_score
                    new_f_score = possible_g_score + heuristic_dict[neighbor.r]
                    heapq.heappush(open_heap, (new_f_score, neighbor))

                self.board = saved_board

        return None

    def reconstruct_path(self, came_from, current: Coord):
        """Uses the came_from dictionary to reconstruct the path from the start"""
        path = []
        node = current
        # Walk backwards from the goal
        while node in came_from:
            parent, action = came_from[node]
            path.append((action, node))
            node = parent

        path.reverse()
        return path

        total_path = []
        while current in came_from:
            current, move = came_from[current]
            total_path.append(move)

        return total_path[::-1]

    def get_all_compressions(self) -> dict:
        """A function that takes the board and a list of blue locations and returns a
        graph of the possible connections between rows on the board via jumps in the form of a dictionary."""

        locations = self.get_all_pos(self.current_player)

        if not locations:
            return {}

        compressions = {}

        if self.current_player == BitBoard.FROG:
            offsets = [
                (self.SAME, self.RIGHT),
                (self.SAME, self.LEFT),
                (self.UP, self.LEFT),
                (self.UP, self.SAME),
                (self.UP, self.RIGHT),
            ]
        else:
            offsets = [
                (self.SAME, self.RIGHT),
                (self.SAME, self.LEFT),
                (self.DOWN, self.LEFT),
                (self.DOWN, self.SAME),
                (self.DOWN, self.RIGHT),
            ]

        coords = []
        for location in locations:
            for offset in offsets:
                try:
                    r = offset[0]
                    c = offset[1]
                    coords.append(location + Coord(r, c))
                except ValueError:
                    # Skip invalid coordinates
                    continue
                # r = offset[0]
                # c = offset[1]
                # safe_col = (c + location.c >= 0) and (c + location.c < BOARD_N)
                # safe_row = (r + location.r >= 0) and (r + location.r < BOARD_N)
                # if safe_col and safe_row:
                #     coords.append(location + Coord(r, c))

        coordsfiltered = [
            coord
            for coord in coords
            if self.board[coord.r][coord.c] in [self.LILLY, self.OPPONENT, self.FROG]
        ]

        for coord in coordsfiltered:
            moves = self.get_possible_move(coord)
            for move_action, neighbor in moves:
                distance = neighbor.r - coord.r
                if distance > 1:
                    if coord.r not in compressions.keys():
                        compressions[coord.r] = [distance]
                    elif distance not in compressions[coord.r]:
                        compressions[coord.r].append(distance)

        return compressions

    def jump_heuristic(self, row: int, compression_dict):
        """
        A function that takes the board, a coordinate, and
        returns a heurstic for the estimated distance to the end
        """
        return self.dijkstra_algorithm(compression_dict, row)


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
