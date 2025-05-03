
from enum import nonmember
import random
import numpy as np
from collections import defaultdict
from .bitboard import BitBoard
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord
from .strategy import Strategy
import time
import json


"""

Minimax works as follows: Expands all possible moves to a given depth.
Chooses the move that gives the best possible guaranteed perforamnce. 

"""

CUTOFF_DEPTH = 4
SHORTENING_FACTOR = 1
ASTAR = False

class MinimaxSearchNode(Strategy):
    def __init__(self, state:BitBoard, parent = None, parent_action = None,
                 time_budget = 178.0, weights = None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        if parent is None:
            self.root_player = state.get_current_player()
            self.time_budget = time_budget
        else:
            self.root_player = state.get_current_player()
            self.time_budget = parent.time_budget
        self.children = []
        self.astar = False
        
        default = {'W_DIST':0.4, 'W_MOB':0.3, 'W_BORDER':0.2, 'W_CENTRAL':0.1}
        with open("weights.json", "r") as wf:
            self.weights = json.load(wf)

    def cutoff_test(self, state:BitBoard, depth, cutoff_depth = CUTOFF_DEPTH):
        if state.is_game_over():
            return True
        if depth >= cutoff_depth:
            # print("Cutoff depth exceeded")
            # print(state.render())
            return True
        return False

    def eval_function(self, state:BitBoard):
        """
        If cutoff test returns true, then we evaluate the position
        """
        if state.is_game_over():
            frog_count, opp_count = (
                state.frog_border_count[BitBoard.FROG],
                state.frog_border_count[BitBoard.OPPONENT],
                )
            if frog_count == BOARD_N - 2 and opp_count != BOARD_N - 2:
                winner_piece = BitBoard.FROG
            elif opp_count == BOARD_N - 2 and frog_count != BOARD_N - 2:
                winner_piece = BitBoard.OPPONENT
            else:
                return 0  # draw
            return 1 if winner_piece == self.root_player else -1
        return self.adaptive_eval(state)

    def adaptive_eval(self, state: BitBoard):
        w = self.weights
        me = self.root_player
        you = BitBoard.FROG if me == BitBoard.OPPONENT else BitBoard.OPPONENT
        board = state.get_board()

        # 1) goal distance
        dist_me = dist_you = 0
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == me:
                    dist_me  += (BOARD_N - 1 - r)
                elif board[r][c] == you:
                    dist_you += r

        # 2) mobility

        moves_me  = len(state.get_all_moves())
        state.toggle_player()
        moves_you = len(state.get_all_moves())
        state.toggle_player()

        # 3) border control
        border_me  = state.frog_border_count[me]
        border_you = state.frog_border_count[you]

        # 4) centralization
        mid = (BOARD_N - 1) / 2
        cent_me = cent_you = 0.0
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == me:
                    cent_me  += mid - abs(c - mid)
                elif board[r][c] == you:
                    cent_you += mid - abs(c - mid)

        # normalize and combine
        norm_dist = (dist_you - dist_me) / (BOARD_N * BOARD_N)
        norm_mob  = (moves_me - moves_you) / (moves_me + moves_you + 1)
        norm_border = (border_me - border_you) / BOARD_N
        norm_cent  = (cent_me - cent_you) / (BOARD_N * BOARD_N)

        score = (
            w['W_DIST']   * norm_dist
          + w['W_MOB']    * norm_mob
          + w['W_BORDER'] * norm_border
          + w['W_CENTRAL']* norm_cent
        )
        return score


    def astar_eval(self, state):
        current_player = self.root_player
        # If frog, we are moving towards the bottom
        board = state.board
        astar_check = 0
        if current_player == BitBoard.FROG:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.FROG:
                        astar_check += (r+1)
        if current_player == BitBoard.OPPONENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.OPPONENT:
                        astar_check += (8- (r+1))
        return astar_check

    
    def simple_eval(self, state):
        current_player = self.root_player
        # If frog, we are moving towards the bottom
        board = state.board
        progress = 0
        if current_player == BitBoard.FROG:
            for r in range(BOARD_N): 
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.FROG:
                        progress += (r+1)
                    elif board[r][c] == BitBoard.OPPONENT:
                        progress -= (8 - (r+1))
        if current_player == BitBoard.OPPONENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    if board[r][c] == BitBoard.OPPONENT:
                        progress += (8 - (r+1))
                    elif board[r][c] == BitBoard.FROG:
                        progress -= (r+1)
        return progress/64

    def max_value(self, state: BitBoard, alpha, beta, depth):
        if self.cutoff_test(state, depth):
            return self.eval_function(state)
        value = float("-inf")
        if self.astar == True:
            moves = state.get_all_optimal_moves()
            newmoves = state.get_all_moves()
            moves.extend(random.sample(newmoves, len(newmoves)//SHORTENING_FACTOR))
        else:
            moves = state.get_all_moves()

        for action in moves:
            new_position = state.move(action[0], action[1])
            new_position.toggle_player()
            value = max(value, self.min_value(new_position, alpha, beta, depth +1))
            if value >= beta:
                return value
            alpha = max(alpha, value) # Max is allowed to determine alpha
        return value

    def min_value(self, state:BitBoard, alpha, beta, depth):
        if self.cutoff_test(state, depth):
            return self.eval_function(state)
        value = float("inf")
        if self.astar == True:
            moves = state.get_all_optimal_moves()
            newmoves = state.get_all_moves()
            moves.extend(random.sample(newmoves, len(newmoves)//SHORTENING_FACTOR))
        else:
            moves = state.get_all_moves()
        for action in moves:
            new_position = state.move(action[0], action[1])
            new_position.toggle_player()
            value = min(value, self.max_value(new_position, alpha, beta, depth +1))
            if value >= beta:
                return value
            beta = min(beta, value) # Min is allowed to determine beta
        return min(beta, value)


    def best_action(self, safety_margin: float = 5, bt: float = 0.75):
        # 1) grab the referee‐supplied clock once
        using_astar = self.state.get_all_optimal_moves()
        if len(using_astar)>=2 and ASTAR:
            self.astar = True
        else:
            self.astar = False


        total_time = self.time_budget
        assert total_time > safety_margin, "No time to move!"

        # 2) compute moves_left as before
        moves_played = self.state.get_ply_count()
        moves_left = (150 - moves_played)//2

        # 3) ideal slice, then clamp below
        raw_alloc = (total_time - safety_margin) / (moves_left**bt)
        alloc_time = min(raw_alloc, total_time - safety_margin)

        assert alloc_time >= 0

        # 4) establish one single *absolute* deadline
        t0 = time.perf_counter()
        hard_deadline = t0 + alloc_time
        
        best_move = None
        alpha  = float("-inf") # Optimal score
        beta = float("inf") # Opponent's optimal score (-inf from their perspective)
        current_state = self.state
        best_action = None

        for action in current_state.get_all_moves():
            new_position = current_state.move(action[0], action[1])
            new_position.toggle_player()
            value = self.min_value(new_position, alpha, beta, depth = 1)
            if value > alpha:
                alpha = value
                best_action = action[0]
        print("Best action is", best_action)
        return {"action": best_action}


       #  while True:
       #      now = time.perf_counter()
       #      if now >= hard_deadline:
       #          break

            # since hard_deadline = t0 + alloc_time ≤ t0 + (total_time - safety_margin),
            # we are guaranteed never to go beyond the referee’s remaining clock.
















































class MonteCarloTreeSearchNode(Strategy):
    avg_playout_depth = 0.0
    playouts_done = 0

    def __init__(
        self, state: BitBoard, parent=None, parent_action=None, time_budget=178.0
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        if parent is None:
            # only the root records who “we” are
            self.root_player = state.get_current_player()
            self.time_budget = time_budget
        else:
            # children inherit the real root’s identity & remaining clock
            self.root_player = parent.root_player
            self.time_budget = parent.time_budget

        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._total_reward = 0
        self._untried_actions = self.untried_actions()
        self.c = 0.2

        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def untried_actions(self):
        blocked = False
        for row in self.state.get_board():
            if np.sum(row) == 0:  # there is a 'block'
                blocked = True
        if not blocked:
            opt = self.state.get_all_optimal_moves()
            # take the top 2–3 optimal, plus 30% of the rest at random
            extra = random.sample(self.state.get_all_moves(), k=int(0.3 * len(opt)))
            return opt + extra
        return self.state.get_all_moves()

    def q(self):
        # return self._total_reward
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        # sort by heuristic priority once
        self._untried_actions.sort(
            key=lambda mv: self.state._move_priority(mv), reverse=True
        )
        idx = 0  # pop the very best move first
        # action_res = self._untried_actions.pop()

        # if self.depth < 5 and self.depth > 1:
        #     # try to pop a GrowAction first
        #     for i, (act, res) in enumerate(self._untried_actions):
        #         if isinstance(act, GrowAction):
        #             idx = i
        #             break
        #     else:
        #         idx = np.random.randint(len(self._untried_actions))
        # else:
        #     idx = np.random.randint(len(self._untried_actions))
        action_res = self._untried_actions.pop(idx)
        action, res = action_res
        next_state = self.state.move(action, res)
        next_state.toggle_player()

        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action_res
        )

        if next_state.is_game_over():
            child_node._number_of_visits += 10
            child_node._results[1] += 10

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def fully_expand(self):
        while not self.is_fully_expanded():
            self.expand()

    def new_rollout_policy(self, state, depth=0):
        """
        epsilon-greedy + softmax-weighted heuristic playout policy.
        """
        moves = state.get_all_moves()
        num_moves = len(moves)
        if num_moves == 0:
            return None

        # 1) ε-greedy: 10% of the time, explore uniformly at random
        #
        eps = 0 if depth > 15 else 0.02
        if np.random.rand() < eps:
            return moves[np.random.randint(num_moves)]

        # 2) Otherwise, score each move by your heuristic
        mid_col = (BOARD_N - 1) // 2
        scores = []
        for action, res in moves:
            score = 0.0
            if isinstance(action, GrowAction):
                # early-game grow bonus, late-game penalty
                if self.state.get_ply_count() < 6:
                    score = 0
                else:
                    score = 0.1
                    # ratio = len(state.move(action, None).get_all_moves()) / len(
                    #     state.get_all_moves()
                    # )  # the increase in moves that we get if we were to grow now
                    # score += +0.0 if (2 < depth and depth < 15) else ratio * 0.01
            else:
                # reward long jumps
                dist = abs(res.r - action.coord.r)
                score += dist if dist > 0 else -0.5
                # centering bonus early
                if depth < 15:
                    start_c, end_c = action.coord.c, res.c
                    if abs(start_c - mid_col) > abs(end_c - mid_col):
                        score += 0.1
            scores.append(score)

        # 3) Softmax to get a probability distribution
        x = np.array(scores, dtype=np.float64)
        # numerical stabilization
        x = x - np.max(x)
        exp_x = np.exp(x / 1.0)  # temperature = 1.0; tune if you like
        probs = exp_x / exp_x.sum()

        # 4) Sample one move according to that distribution
        choice_idx = np.random.choice(num_moves, p=probs)
        return moves[choice_idx]

    def rollout_policy(self, state, depth=0):
        weights = []

        best_score = -float("inf")
        best_move = None
        mid_col = (BOARD_N - 1) // 2
        curr_player = state.get_current_player()
        scores = []
        moves = state.get_all_moves()

        for action, res in moves:
            score = 0
            if isinstance(action, GrowAction):
                score += 0.4 if (depth < 15 and depth > 2) else -0.2
            else:
                # e.g. reward long jumps, centering, etc.

                vert_dist = abs(res.r - action.coord.r)
                # 2b) multi-jump bonus
                #
                if vert_dist < 1:
                    score -= 0.9
                else:
                    score += 1 * vert_dist

                # 2c) centering bonus
                start_c, end_c = action.coord.c, res.c

                if abs(start_c - mid_col) > abs(end_c - mid_col) and depth < 15:
                    score += 0.1

            scores.append(score)
            if score > best_score:
                best_score = score
                best_move = (action, res)

        raw = scores
        offset = -min(raw) if min(raw) < 0 else 0
        adjusted = [s + offset for s in raw]
        total = sum(adjusted)
        if total == 0:
            probs = [1 / len(raw)] * len(raw)
        else:
            probs = [a / total for a in adjusted]

        return random.choices(moves, probs)[0]

    # new version of rollout
    def simulate_playout(self):
        state = BitBoard(np.copy(self.state.get_board()))
        state.current_player = self.state.current_player
        depth = 0
        max_depth = 150 - self.state.get_ply_count()

        # fast, stateless playout on BitBoard only
        while not state.is_game_over() and depth < max_depth:
            if False:
                # using board eval
                next_states = {}
                for action, res in state.get_all_moves():
                    next_state = state.move(action, res)
                    next_state.toggle_player()
                    next_states[next_state] = next_state.evaluate_position()

                raw = next_states.values()
                offset = -min(raw) if min(raw) < 0 else 0
                adjusted = [s + offset for s in raw]
                total = sum(adjusted)
                if total == 0:
                    probs = [1 / len(raw)] * len(raw)
                else:
                    probs = [a / total for a in adjusted]

                state = random.choices(list(next_states.keys()), weights=probs)[0]

            if True:
                # using rollout policy
                action, res = self.new_rollout_policy(state, depth=depth)
                state = state.move(action, res, in_place=True)  # for performance
                state.toggle_player()

            # state.toggle_player()
            # action, res = self.rollout_policy(state, depth=self.depth + depth // 2)
            # state = state.move(action, res)
            # state.toggle_player()
            depth += 1

        MonteCarloTreeSearchNode.playouts_done += 1
        # incremental average
        MonteCarloTreeSearchNode.avg_playout_depth += (
            depth - MonteCarloTreeSearchNode.avg_playout_depth
        ) / MonteCarloTreeSearchNode.playouts_done

        if state.is_game_over():
            # compute which side actually reached the goal
            frog_count, opp_count = (
                state.frog_border_count[BitBoard.FROG],
                state.frog_border_count[BitBoard.OPPONENT],
            )
            if frog_count == BOARD_N - 2 and opp_count != BOARD_N - 2:
                winner_piece = BitBoard.FROG
            elif opp_count == BOARD_N - 2 and frog_count != BOARD_N - 2:
                winner_piece = BitBoard.OPPONENT
            else:
                return 0  # draw
            return +1 if winner_piece == self.root_player else -1

        return 1 if state.evaluate_position() > 0 else -1
        # return 1 if self.heuristic_score(state) > 0 else -1

    def heuristic_score(self, state):
        score = 0
        board = state.get_board()
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if board[r][c] == self.state.get_current_player():
                    match self.state.get_current_player():
                        case BitBoard.FROG:
                            score += r
                            break
                        case BitBoard.OPPONENT:
                            score += BOARD_N - 1 - r
                            break
                elif board[r][c] not in (BitBoard.LILLY, BitBoard.EMPTY):
                    match self.state.get_current_player():
                        case BitBoard.FROG:
                            score -= BOARD_N - 1 - r
                            break
                        case BitBoard.OPPONENT:
                            score -= r
                            break
                    score -= (
                        BOARD_N - 1 - r
                    )  # still an error here, not calculated coorectl
        return score

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        # self._total_reward += result

        if self.parent:
            self.parent.backpropagate(-result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def dynamic_c(self):
        # start exploratory, then exploit more later
        return self.c / (1 + np.log(1 + self.n()))

    def new_tree_policy(self):
        widen_k = 3
        node = self
        while not node.is_terminal_node():
            # if no child yet, always expand first
            if len(node.children) == 0:
                return node.expand()

            # if we still have untried actions AND
            # we've visited this node enough times, expand
            if not node.is_fully_expanded() and node.n() > widen_k * len(node.children):
                return node.expand()

            # otherwise, pick the best child by UCB
            node = node.UCB_choose()

        return node

    def _tree_policy(self):
        node = self
        # descend until we find a node we can expand or a terminal
        while not node.is_terminal_node():
            if not node.is_fully_expanded():
                return node.expand()  # expand one child and return it
            else:
                node = node.UCB_choose()
        return node

    def UCB_choose(self):
        # return self.best_child()
        for child in self.children:
            if child.n() == 0:
                return child

        # try to add move_prio to UCB1
        scores = []
        for c in self.children:
            value = c.q() / (c.n() + 1e-5)
            exploration = self.dynamic_c() * np.sqrt(
                (2 * np.log(self.n() + 1) / (c.n() + 1e-5))
            )
            priority_bonus = self.state._move_priority(c.parent_action)
            scores.append(value + exploration + 0.1 * priority_bonus)  # tune this
        return self.children[np.argmax(scores)]

        scores = [
            (c.q() / c.n()) + self.dynamic_c() * np.sqrt((2 * np.log(self.n()) / c.n()))
            for c in self.children
        ]
        return self.children[np.argmax(scores)]

    def choose_next_action(self):
        # break ties by winrate
        best = max(
            self.children,
            key=lambda c: (c.n(), (self._results[1] / c.n()) if c.n() > 0 else 0),
        )
        return best

    def best_action(self, safety_margin: float = 5, beta: float = 0.75):
        # 1) grab the referee‐supplied clock once
        total_time = self.time_budget
        assert total_time > safety_margin, "No time to move!"

        # 2) compute moves_left as before
        moves_played = self.state.get_ply_count()
        total_pred = (
            type(self).avg_playout_depth if type(self).playouts_done > 0 else 150.0
        )
        moves_left = max(1, int(total_pred) - moves_played)

        # 3) ideal slice, then clamp below
        raw_alloc = (total_time - safety_margin) / (moves_left**beta)
        alloc_time = min(raw_alloc, total_time - safety_margin)

        assert alloc_time >= 0

        # 4) establish one single *absolute* deadline
        t0 = time.perf_counter()
        hard_deadline = t0 + alloc_time

        sims = 0
        # 5) loop until *either* we hit the per‐move slice *or* the referee clock
        while True:
            now = time.perf_counter()
            if now >= hard_deadline:
                break
            # since hard_deadline = t0 + alloc_time ≤ t0 + (total_time - safety_margin),
            # we are guaranteed never to go beyond the referee’s remaining clock.
            leaf = self.new_tree_policy()
            reward = leaf.simulate_playout()
            leaf.backpropagate(reward)
            sims += 1

        # 6) if you managed zero sims, force one expansion so you always return something
        if sims == 0:
            self.new_tree_policy()

        # 7) pick the child with most visits
        best_child = max(self.children, key=lambda c: c.n())

        print(f"[MCTS] sims this move: {sims}")

        return {
            "action": best_child.parent_action[0],
            "res": best_child.parent_action[1],
            "res_node": best_child,
        }

    def find_child(self, action):
        for child in self.children:
            if child.parent_action[0] == action:
                return child

