import numpy as np
from referee.game.actions import MoveAction, GrowAction
from referee.game.coord import Coord, Direction
from referee.game.board import CellState
from referee.game.constants import BOARD_N


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
        # if test:
        #     print("possible jumps")
        #     poss = self.get_possible_move(Coord(0, 2))
        #     for jump, res in poss:
        #         print(jump, res)
        #     print(f"Doing action {poss[0][0]}")
        #     print(self.move(poss[0][0], poss[0][1]).get_board())

    def get_board(self):
        return self.board

    def get_current_player(self):
        return self.current_player

    def toggle_player(self):
        if self.current_player == self.FROG:
            self.current_player = self.OPPONENT
        else:
            self.current_player = self.FROG

    def is_game_over(self):
        self.frog_border_count = {self.FROG: 0, self.OPPONENT: 0}
        # for r in [0, BOARD_N - 1]:
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
        # if self.is_game_over():
        # frog_count = {self.FROG: 0, self.OPPONENT: 0}
        # for r in [0, BOARD_N - 1]:
        #     for c in range(BOARD_N):
        #         if self.board[r][c] == self.FROG:
        #             frog_count[self.FROG] += 1
        #         elif self.board[r][c] == self.OPPONENT:
        #             frog_count[self.OPPONENT] += 1
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

    # return None

    def move(self, action: MoveAction | GrowAction, res: Coord | None = None):
        """Move the frog to the new position"""

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
                        next_c >= 0
                        and next_c < BOARD_N
                        and next_r >= 0
                        and next_r < BOARD_N
                        and board_copy[next_r][next_c] == self.EMPTY
                    ):
                        board_copy[next_r][next_c] = self.LILLY
        elif res is not None:
            board_copy = np.copy(self.board)
            board_copy[action.coord.r][action.coord.c] = self.EMPTY
            board_copy[res.r][res.c] = fill
        else:
            next_coord = action.coord
            for direction in action.directions:
                found_move = False
                direction_test = direction
                while not found_move:
                    next_coord = next_coord + direction_test
                    # this shouldn't ever give an error because ref is supposed to check
                    if board_copy[next_coord.r][next_coord.c] == self.LILLY:
                        found_move = True
            board_copy[action.coord.r][action.coord.c] = self.EMPTY
            board_copy[next_coord.r][next_coord.c] = fill

        return BitBoard(board_copy)

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

        return possible_moves

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

    def get_possible_move(self, coord: Coord) -> list[tuple[MoveAction, Coord]]:
        """Get all possible moves from a given coordinate the output format is a list of tuples, each tuple contains a MoveAction and the resulting coordinate"""
        visited = set()
        moves = self.get_possible_move_rec(coord, visited)

        return moves

    def get_possible_move_rec(
        self, coord, visited, in_jump=False
    ) -> list[tuple[MoveAction, Coord]]:
        """Get all possible moves from a given coordinate the output format is a list of tuples, each tuple contains a MoveAction and the resulting coordinate"""
        possible_jumps = []

        visited.add(coord)

        if self.current_player == self.FROG:
            forward = 1
        else:
            forward = -1

        for direction in Direction:
            single_jump = MoveAction(coord, [direction])
            if self.is_valid_move(single_jump, forward) and not in_jump:
                possible_jumps.append((single_jump, coord + direction))

            try:
                first_jump = coord + direction
                double_jump_res = coord + direction + direction
            except ValueError:
                # Skip invalid coordinates
                continue
            if (
                self.is_valid_move(MoveAction(first_jump, direction), forward)
                and (double_jump_res not in visited)
                and self.board[first_jump.r][first_jump.c]
                in [
                    self.FROG,
                    self.OPPONENT,
                ]
            ):
                double_jump = MoveAction(coord, direction)

                # add the double jump
                possible_jumps.append((double_jump, double_jump_res))

                # add the possible jumps from the double jump
                for jump, res in self.get_possible_move_rec(
                    double_jump_res, visited, True
                ):
                    possible_jumps.append(
                        (MoveAction(coord, [direction] + list(jump.directions)), res)
                    )

        return possible_jumps

    def is_valid_move(self, move: MoveAction, forward=1) -> bool:
        """Check if a move is valid"""
        coord = move.coord
        directions = move.directions

        for direction in directions:
            # immediately deny if the move does not go 'forward'
            if direction.r not in (forward, 0):
                return False
            next_c = coord.c + direction.c
            next_r = coord.r + direction.r
            if next_c < 0 or next_c >= BOARD_N or next_r < 0 or next_r >= BOARD_N:
                return False
            if self.board[(coord + direction).r][(coord + direction).c] != self.LILLY:
                # not a possible jump
                return False

            coord += direction
        return True
