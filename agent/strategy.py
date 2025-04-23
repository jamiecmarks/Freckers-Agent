import abc
from referee.game.actions import GrowAction, MoveAction
from referee.game.coord import Coord


class Strategy:
    def __init__(self, state):
        self.state = state

    @abc.abstractmethod
    def best_action(self) -> dict:  # tuple["GrowAction | MoveAction", "Coord | None"]:
        pass

    @abc.abstractmethod
    def find_child(self, action) -> "Strategy | None":
        pass
