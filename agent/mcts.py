import random
import numpy as np
from collections import defaultdict
from .bitboard import BitBoard
from referee.game.actions import GrowAction, MoveAction
from referee.game.constants import BOARD_N
from referee.game.coord import Coord
from .strategy import Strategy
import math
import time


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
        self.c = 0.22

        # trying rave stuff
        self._rave_visits = defaultdict(int)
        self._rave_results = defaultdict(int)

        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def untried_actions(self):
        blocked = False
        # for row in self.state.get_board():
        #     if np.sum(row) == 0:  # there is a 'block'
        #         blocked = True
        #         break
        #
        # if not blocked:
        #     opt = self.state.get_all_optimal_moves()
        #     # take the top 2–3 optimal, plus 30% of the rest at random
        #     extra = random.sample(
        #         self.state.get_all_moves()[:-1], k=int(0.3 * len(opt))
        #     )
        #     all_moves = opt[:-1] + extra + [(GrowAction(), None)]
        # else:
        all_moves = self.state.get_all_moves()

        hops = all_moves[:-1]
        grows = all_moves[-1:]

        # early in the search: explore hops first
        # if self.n() < 5 and hops:
        #     return hops

        # once we've visited this node a handful of times,
        # add Grow back in, sorted by priority
        actions = hops + grows
        # actions.sort(key=lambda mv: self.state._move_priority(mv), reverse=True)

        return actions

    def kuntried_actions(self):
        blocked = False
        for row in self.state.get_board():
            if np.sum(row) == 0:  # there is a 'block'
                blocked = True
        if not blocked:
            opt = self.state.get_all_optimal_moves()
            # take the top 2–3 optimal, plus 30% of the rest at random
            extra = random.sample(
                self.state.get_all_moves()[:-1], k=int(0.3 * len(opt))
            )
            return opt[:-1] + extra + [(GrowAction(), None)]
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
        # moves = state.get_all_moves()
        test = state.get_random_move()
        # print("random move is ", test)

        return test

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
            res_state = state.move(action, res)
            score = 0.0
            if res is None:  # grow action
                # early-game grow bonus, late-game penalty
                ratio = len(res_state.get_all_moves()) / len(
                    state.get_all_moves()
                )  # the increase in moves that we get if we were to grow now
                score += ratio * 0.1
            else:
                # reward long jumps
                dist = abs(res.r - action.coord.r)
                score += dist if dist > 0 else -0.5
                # centering bonus early
                if depth < 15:
                    start_c, end_c = action.coord.c, res.c
                    if abs(start_c - mid_col) > abs(end_c - mid_col):
                        score += 0.1

                coords = state.get_all_pos(res_state.get_current_player())

                rows = [r for (r, _) in coords]
                mean_r = sum(rows) / len(coords)  # mean row of frogs
                spread = sum(
                    (r - mean_r) ** 2 for r in rows
                )  # find variance of frog rows, penalize too much spreading. Leaving froggy behind

                score -= 0.1 * spread

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
            if res is None:  # grow action
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
        # max_depth = 30

        playout_moves = []

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

                if (
                    res is not None and state.get_current_player() == self.root_player
                ):  # moveaction
                    start = action.coord
                    end = res
                    playout_moves.append((start, end))
                    # don't rave growactions

                state = state.move(action, res, in_place=True)  # for performance
                state.toggle_player()

            # state.toggle_player()
            # action, res = self.rollout_policy(state, depth=self.depth + depth // 2)
            # state = state.move(action, res)
            # state.toggle_player()
            depth += 1

        result = 0

        MonteCarloTreeSearchNode.playouts_done += 1

        # incremental average
        MonteCarloTreeSearchNode.avg_playout_depth += (
            depth - MonteCarloTreeSearchNode.avg_playout_depth
        ) / MonteCarloTreeSearchNode.playouts_done

        if state.is_game_over():
            # compute which side actually reached the goal
            winner_piece = None

            frog_count, opp_count = (
                state.frog_border_count[BitBoard.FROG],
                state.frog_border_count[BitBoard.OPPONENT],
            )
            if frog_count == BOARD_N - 2 and opp_count != BOARD_N - 2:
                winner_piece = BitBoard.FROG
            elif opp_count == BOARD_N - 2 and frog_count != BOARD_N - 2:
                winner_piece = BitBoard.OPPONENT
            else:
                result = 0  # draw

            if winner_piece:  # not a draw
                result = 1 if winner_piece == self.root_player else -1

        else:
            eval_score = 1 if state.evaluate_position() > 0 else -1

            if state.get_current_player() == self.root_player:
                result = eval_score
            else:
                result = -eval_score

        return result, playout_moves
        # return 1 if state.evaluate_position() > 0 else -1
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

    def backpropagate(self, result, playout_moves=None):
        self._number_of_visits += 1
        self._results[result] += 1
        # self._total_reward += result

        if playout_moves:
            for a in playout_moves:
                self._rave_visits[a] += 1
                self._rave_results[a] += result

        if self.parent:
            self.parent.backpropagate(-result, playout_moves)

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
                node = node.UCB_rave_choose()
        return node

    def UCB_rave_choose(self):
        best, best_score = None, -float("inf")
        parent_N = self._number_of_visits
        for child in self.children:
            a = child.parent_action  # the (action,res) tuple
            mv, res = a
            # 1) exact statistics
            Q = child._results[+1] - child._results[-1]
            N = child._number_of_visits
            v_exact = Q / (N + 1e-9)

            # 2) rave statistics
            if res is not None:
                R = self._rave_results[(mv.coord, res)]
                Nr = self._rave_visits[(mv.coord, res)]
                v_rave = (R / Nr) if Nr > 0 else 0

                # 3) mixing parameter: the more visits this child has, the less we trust RAVE
                beta = Nr / (N + Nr + 1e-9)
                value_term = (1 - beta) * v_exact + beta * v_rave
            else:
                value_term = 0

            explore_term = self.c * math.sqrt(math.log(parent_N + 1) / (N + 1e-9))

            score = value_term + explore_term
            if score > best_score:
                best, best_score = child, score

        return best

    def UCB_choose(self):
        best, best_score = None, -1e9
        totalN = self.n() + 1
        for child in self.children:
            q = child.q()
            n = child.n()
            exploit = q / (n + 1e-6)
            explore = self.c * math.sqrt(2 * math.log(totalN) / (n + 1e-6))

            # a quick “grow‐worth” estimate:
            action, res = child.parent_action
            if res is None:  # grow action
                # how many new hops does grow give you?
                after = child.state.get_all_moves()
                before = self.state.get_all_moves()
                grow_delta = len(after) - len(before)

                if grow_delta <= 0:
                    grow_bonus = -1.0  # kill pointless grows
                else:
                    grow_bonus = +0.2 * grow_delta
            else:
                grow_bonus = 0

            score = exploit + explore + grow_bonus
            if score > best_score:
                best, best_score = child, score
        return best

    def kUCB_choose(self):
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

    def best_action(self, safety_margin: float = 0.5, decay: float = 0.975):
        # 1) snapshot
        t0 = time.perf_counter()
        rem_clock = self.time_budget - safety_margin
        assert rem_clock > 0, "no clock left!"

        # 2) count how many plies have been played (total, both sides)
        moves_played = self.state.get_ply_count() // 2
        # clamp to [0,150]
        moves_played = min(max(moves_played, 0), 150 // 2)

        sum_geom = decay**moves_played - decay ** (150 // 2)
        weight = (1 - decay) * decay**moves_played / sum_geom

        # 5) allocate exactly that fraction of rem_clock to *this* move
        alloc_time = rem_clock * weight

        # 6) now run MCTS until we hit our hard deadline
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
            reward, playout_moves = leaf.simulate_playout()
            leaf.backpropagate(reward, playout_moves)
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
