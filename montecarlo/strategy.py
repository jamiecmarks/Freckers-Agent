import abc

class Strategy:
    def __init__(self, state):
        self.state = state
        self.time_budget = 180

    @abc.abstractmethod
    def best_action(self) -> dict:  # tuple["GrowAction | MoveAction", "Coord | None"]:
        pass

    @abc.abstractmethod
    def find_child(self, action) -> "Strategy | None":
        pass
