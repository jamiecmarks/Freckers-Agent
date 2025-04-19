# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
import numpy as np
import random
from .internal_state import FreckersState
from .bitboard import BitBoard
from .mcts import MonteCarloTreeSearchNode


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

        if self.test_count < 1:
            self.test_count += 1
            print("Testing MCTS")
            action_out = self.root.best_action()
            # self.root.state.toggle_player()

            return action_out["action"]

        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a MOVE action")
                return MoveAction(Coord(0, 3), [Direction.Down])
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a GROW action")
                return GrowAction()

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """

        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.
        child = self.root.find_child(action)

        if child is None:
            # create a new child node
            new_board = self.root.state.move(action)
            new_board.toggle_player()
            child = MonteCarloTreeSearchNode(
                new_board
            )  # default perspective is from blue

        self.root = child

        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
            case GrowAction():
                print(f"Testing: {color} played GROW action")
            case _:
                raise ValueError(f"Unknown action type: {action}")
