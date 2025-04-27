import random
import numpy as np
from collections import defaultdict
from .bitboard import BitBoard
from .random_strat import RandomStrat
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord
from .strategy import Strategy
import random 

LATE_DEPTH = 150
EARLY_DEPTH = 150
DEPTH_THRESHOLD = 60
SIMULATIONS = 60
BRUTE_FORCE_SIMS = 400

GREEDY_THRESHOLD = 30
BRUTE_THRESHOLD = 5
GREEDY = -1
MONTE = 1
BRUTE = 2


class BruteForceSearchNode(Strategy):
    def __init__(self, state: BitBoard, parent = None, parent_action = None):
        self.state =state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._untried_actions  = self.untried_actions()
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
    def untried_actions(self):
        return self.state.get_all_moves()
    


    def expand(self):
        # action_res = self._untried_actions.pop()
        idx = np.random.randint(len(self._untried_actions))

        action_res = self._untried_actions.pop(idx)
        action, res = action_res
        next_state = self.state.move(action, res)
        next_state.toggle_player()

        child_node = BruteForceSearchNode(
            next_state, parent=self, parent_action=action_res
        )

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        moves_made = []
        limit =40; depth = 0
        current_rollout = self.state
        while (not current_rollout.is_game_over()) and depth < limit:
            depth +=1
            curr_player = current_rollout.get_current_player()
            legal_moves = current_rollout.get_all_moves()
            
            # Weighted random choice
            legal_filtered = [move for move in legal_moves if move[0] != GrowAction()]
            target_move = [move for move in legal_filtered if move[0].coord.r != BOARD_N-1]
            target_move.append((GrowAction(), None))
            chosen_move = random.choice(target_move)



            next_move_mv, next_move_res = chosen_move
            moves_made.append((next_move_mv, next_move_res))
            current_rollout = current_rollout.move(next_move_mv, next_move_res)
            
            if current_rollout.get_current_player() == curr_player:
                current_rollout.toggle_player()

        # didn't reach a terminal state
        if not current_rollout.is_game_over():
            if current_rollout.get_current_player!=self.state.get_current_player:
                current_rollout.toggle_player()
            return 0, moves_made


        return current_rollout.get_winner(), moves_made

    def best_action(self, simulation_no=BRUTE_FORCE_SIMS):
        # if not self.children and self._untried_actions:
        #     self.expand()
        best_move_total = float("inf")
        best_result = -1
        best_move = None
        for _ in range(simulation_no):
            v = self
            reward, moves = v.rollout()
            if len(moves) < best_move_total and reward == 1:
                best_move_total = min(len(moves), best_move_total)
                best_move = moves[0][0]
        print("Estimated distance to finish: ", best_move_total)
        if not best_move:
            best_move = GrowAction
        print(best_move)
        return {
            "action": best_move
        }













class MonteCarloTreeSearchNode(Strategy):
    def __init__(self, state: BitBoard, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action  # (action, res)
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()
        self.c = 1.0

        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def untried_actions(self):
        return self.state.get_all_moves()

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        # action_res = self._untried_actions.pop()
        idx = np.random.randint(len(self._untried_actions))

        action_res = self._untried_actions.pop(idx)
        action, res = action_res
        next_state = self.state.move(action, res)
        next_state.toggle_player()

        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action_res
        )

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()


    def rollout1(self):

        current_rollout = self.state
        max_depth = EARLY_DEPTH if self.depth < DEPTH_THRESHOLD else LATE_DEPTH
        depth = 0
        while (not current_rollout.is_game_over()) and (max_depth > depth):
            if depth > 75:
                break

            move_values = []
            curr_player = current_rollout.get_current_player()
            legal_moves = current_rollout.get_all_moves()
            total_moves = len(legal_moves)
            
            for move in legal_moves:
                result = current_rollout.move(move[0], move[1])
                score = result.quick_eval(move, total_moves)
                move_values.append((score, move))

            # Make all scores positive by shifting if needed
            min_score = min(score for score, _ in move_values)
            weights = [(score - min_score + 1e-3) for score, _ in move_values]

            # Weighted random choice
            chosen_move = random.choices([m for _, m in move_values], weights=weights, k=1)[0]
            next_move_mv, next_move_res = chosen_move

            current_rollout = current_rollout.move(next_move_mv, next_move_res)
            
            if current_rollout.get_current_player() == curr_player:
                current_rollout.toggle_player()

            depth += 1

        # didn't reach a terminal state
        if not current_rollout.is_game_over():
            if current_rollout.get_current_player!=self.state.get_current_player:
                current_rollout.toggle_player()
            return current_rollout.evaluate_position()
        return current_rollout.get_winner()
        



    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1

        if self.parent:
            self.parent.backpropagate(-result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def dynamic_c(self):
        # start exploratory, then exploit more later
        return self.c / (1 + np.log(1 + self.n()))

    def best_child(self):
        # chooose_rand = len(self._untried_actions)
        choices_weights = []
        for c in self.children:
            if c.is_terminal_node():
                return c

            if c.n() == 0:
                choices_weights.append(float("inf"))  # prioritize unvisited nodes
                # choices_weights.append(c.q() / c.n())
            else:
                mult = 1
                if isinstance(c.parent_action[0], GrowAction):
                    mult += 0.2 if self.depth < 15 and self.depth > 2 else -0.2
                else:
                    start_r = c.parent_action[0].coord.r
                    end_r = c.parent_action[1].r
                    start_c = c.parent_action[0].coord.c
                    end_c = c.parent_action[1].c
                    vert_dist = abs(start_r - end_r)
                    if vert_dist > 1:
                        mult += 0.2 * vert_dist
                    elif vert_dist == 0:
                        mult -= 0.2

                    midboard = (BOARD_N - 1) // 2

                    if abs(start_c - midboard) > abs(end_c - midboard):
                        # moving towards the middle is good
                        mult += 0.2 if self.depth < 25 else 0

                    max_adv_dist = 0
                    for mv, res in c.state.get_all_moves():
                        if isinstance(mv, MoveAction):
                            max_adv_dist = max(max_adv_dist, abs(res.r - mv.coord.r))

                    mult -= 0.1 * (
                        max_adv_dist - 1
                    )  # try and be a menace and minimize opponents vertical movement

                choices_weights.append(
                    mult
                    * (
                        (c.q() / c.n())
                        + self.c * np.sqrt((2 * np.log(self.n()) / c.n()))
                    )
                )
        max_val = max(choices_weights)
        all_max = [i for i, v in enumerate(choices_weights) if v == max_val]
        return self.children[random.choice(all_max)]


    def rollout_policy(self, possible_moves):
        # compute raw scores
        #
        eps = 0.2  # 80% follow “best” move, 20% pick random
        if np.random.rand() < eps:
            return possible_moves[np.random.randint(len(possible_moves))]

        best = max(possible_moves, key=lambda mv: self.state._move_priority(mv))

        return best

    def _tree_policy(self):
        current_node = self
        while current_node.is_fully_expanded():
            current_node = current_node.best_child()
        current_node = current_node.expand()

        return current_node

    def choose_next_action(self):
        children = self.children
        num_visited = []
        for child in children:
            num_visited.append(child.n())

        return children[np.argmax(num_visited)]

    def check_brute_force(self):
        """
            sees whether we should stop doing monte 
            carlo -- for now, if we are only 3 moves vertically from finishing
        """
        board = self.state.get_board()

        score =  0

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == self.state.current_player:
                    match self.state.current_player:
                        case BitBoard.FROG:
                            score += BOARD_N - 1 - r
                        case BitBoard.OPPONENT:
                            score += r
        print("Brute force estimate: ", score)
        if score < BRUTE_THRESHOLD:
            print("brute active")
            return BRUTE
        elif score > GREEDY_THRESHOLD:
            print("greedy active")
            return GREEDY
        print("normal active")
        return False
            
        
    def best_action(self, simulation_no=SIMULATIONS):
        # if not self.children and self._untried_actions:
        #     self.expand()

        brute_force_check = self.check_brute_force()

        if brute_force_check == BRUTE:
            print("We are now solving via brute force!")
            node = BruteForceSearchNode(self.state)
            return node.best_action()
        elif brute_force_check == GREEDY:
            print("We are now solving via greedy first!")
            node = RandomStrat(self.state)
            return node.best_action()

        for _ in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout1()
            v.backpropagate(reward)

        best = self.choose_next_action()
        return {
            "action": best.parent_action[0],
            "res": best.parent_action[1],
            "res_node": best,
        }  # if best else None

    def find_child(self, action):
        for child in self.children:
            if child.parent_action[0] == action:
                return child


