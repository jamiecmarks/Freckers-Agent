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
        print("possible jumps")
        for jump, res in self.get_all_moves():
            print(jump, res)

    def get_board(self):
        return self.board

    def get_all_moves(self):
        possible_moves = []
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if self.board[r][c] == self.FROG:
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
            for c in range(1, BOARD_N):
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

        for direction in Direction:
            single_jump = MoveAction(coord, [direction])
            if self.is_valid_move(single_jump) and not in_jump:
                possible_jumps.append((single_jump, coord + direction))

            try:
                first_jump = coord + direction
                double_jump_res = coord + direction + direction
            except ValueError:
                # Skip invalid coordinates
                continue
            if (
                self.is_valid_move(MoveAction(first_jump, direction))
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
                        (MoveAction(coord, [direction] + jump.directions), res)
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
