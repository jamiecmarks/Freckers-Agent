import numpy as np
from referee.game.actions import MoveAction, GrowAction
from referee.game.coord import Coord, Direction
from referee.game.constants import BOARD_N

# Global cache: maps (board_tuple, current_player) to move lists
t_move_cache: dict[tuple, list[tuple[MoveAction, Coord]]] = {}


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
        # use tuple(board.flatten()) and player to key cache
        key = (tuple(self.board.flatten()), self.current_player)
        if key in t_move_cache:
            # return a fresh list so callers can mutate without affecting cache
            return t_move_cache[key].copy()

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
        t_move_cache[key] = possible_moves
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
