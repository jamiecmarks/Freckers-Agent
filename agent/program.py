# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from .minimax import MinimaxSearchNode
from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from .bitboard import BitBoard
from .mcts import MonteCarloTreeSearchNode
import cProfile
import pstats


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
        print("I am an mcts agent")

        self.total_moves = 0
        bitboard = BitBoard()
        self.minimax = False
        self.root = MonteCarloTreeSearchNode(bitboard)
        # self.root = MinimaxSearchNode(bitboard)

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.

        profiler = cProfile.Profile()
        profiler.enable()

        self.root.time_budget = referee["time_remaining"]
        action_out = self.root.best_action()  # simulate only as many moves as possible
        if self.root.minimax ==True:
            print("minimax!")
            self.minimax = True
        profiler.disable()

        stats = pstats.Stats(profiler)

        stats.strip_dirs().sort_stats("cumulative")
        # Save profiling data for later analysis
        stats.dump_stats("mcts_profile.prof")
        
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

        if self.root.minimax:
            new_board = self.root.state.move(action)
            new_board.toggle_player()
            child = MinimaxSearchNode(new_board)
            child.time_budget = referee["time_remaining"]
            self.root = child
            return
        child = self.root.find_child(action)
        if child is None:
            # create a new child node
            new_board = self.root.state.move(action)
            new_board.toggle_player()
            child = MonteCarloTreeSearchNode(new_board)
        
        child.time_budget = referee["time_remaining"]

        self.root = child
