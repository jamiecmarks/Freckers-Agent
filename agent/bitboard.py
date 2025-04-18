import numpy as np


class BitBoard:
    LILLY = 0b01
    FROG = 0b10
    EMPTY = 0b00
    OPPONENT = 0b11

    def __init__(self, board=None):
        if board is None:
            board = self.get_start_board()
        self.board = board

    def get_board(self):
        return self.board

    def get_start_board(self):
        board = np.full((8, 8), self.EMPTY, dtype=int)

        # Initialize lily pads and starting frog positions based on game rules [66, Figure 3]
        for r in [0, 7]:
            for c in range(1, 7):
                board[r][c] = self.FROG if r == 0 else self.OPPONENT

        for r in [1]:
            opp_r = 7 - r
            for c in range(1, 7):
                if board[r][c] == self.EMPTY:
                    board[r][c] = self.LILLY
                    board[opp_r][c] = self.LILLY

        for r in [0]:
            for c in [0, 7]:
                board[r][c] = self.LILLY
                board[7 - r][c] = self.LILLY
        return board
