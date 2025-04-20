# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
import numpy as np
import random
from .bitboard import BitBoard
from .mcts import MonteCarloTreeSearchNode
from referee.game.constants import MAX_TURNS


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color

        self.total_moves = 0
        bitboard = BitBoard()

        self.test_count = 0
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")
        self.root = MonteCarloTreeSearchNode(bitboard)

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.

        if self.test_count < 75:
            self.test_count += 1
            print("Testing MCTS")
            action_out = self.root.best_action(
                100
            )  # simulate only as many moves as possible

            # print(action_out["res_node"].state.get_board())
            return action_out["action"]

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """

        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.

        self.total_moves += 1
        child = self.root.find_child(action)

        if child is not None:
            print("Found child")

        if child is None:
            # create a new child node
            new_board = self.root.state.move(action)
            new_board.toggle_player()
            child = MonteCarloTreeSearchNode(new_board)

        self.root = child
