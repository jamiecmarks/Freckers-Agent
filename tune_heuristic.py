import json
import random
import subprocess
import sys
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

# --- Hill-climbing tuner as before ---
class HeuristicTuner:
    def __init__(self, initial_weights, perturbation=0.02, min_perturbation=0.005, decay=0.995):
        self.weights = initial_weights.copy()
        self.base_perturbation = perturbation
        self.perturbation = perturbation
        self.min_perturbation = min_perturbation
        self.decay = decay
        self.last_perturb = None
        self.performance_buffer = []
        self.buffer_size = 3
        self.direction_memory = {}
        self.step_count = 0

        self.best_weights = deepcopy(self.weights)
        self.best_avg_score = 0
        self.recent_scores = []

    def propose_change(self):
        if self.last_perturb is not None and len(self.performance_buffer) < self.buffer_size:
            return

        if self.last_perturb and len(self.performance_buffer) == self.buffer_size:
            avg_result = sum(self.performance_buffer) / self.buffer_size
            w, delta = self.last_perturb
            if avg_result <= 0:
                self.weights[w] -= delta
            else:
                self.direction_memory[w] = delta

        self.performance_buffer = []

        # Backup & revert if recent performance is much worse
        if len(self.recent_scores) >= 10:
            recent_avg = sum(self.recent_scores[-10:]) / 10
            if recent_avg < self.best_avg_score - 0.2:
                self.weights = deepcopy(self.best_weights)
                self.direction_memory.clear()
                self.perturbation = self.base_perturbation
                print("[REVERT] Performance dropped too far, reverting to best weights")

        w = random.choice(list(self.weights.keys()))
        prev_delta = self.direction_memory.get(w, 0)
        bias = 0.5 * prev_delta
        delta = ((random.random() * 2 - 1) * self.perturbation) + bias
        self.last_perturb = (w, delta)
        self.weights[w] += delta

        self.step_count += 1
        self.perturbation = max(self.min_perturbation, self.base_perturbation * (self.decay ** self.step_count))

    def evaluate_outcome(self, won: bool):
        score = 1 if won else 0
        self.performance_buffer.append(score)
        self.recent_scores.append(score)
        if len(self.recent_scores) > 100:
            self.recent_scores.pop(0)

        recent_avg = sum(self.recent_scores[-10:]) / 10
        if recent_avg > self.best_avg_score:
            self.best_avg_score = recent_avg
            self.best_weights = deepcopy(self.weights)

    def get_weights(self):
        return deepcopy(self.weights)



def dump_weights(weights, path="weights.json"):
    with open(path, "w") as f:
        json.dump(weights, f)

def main(n_games=500):
    with open("weights.json", "r") as f:
        initial = json.load(f)
    tuner = HeuristicTuner(initial, perturbation=0.02)
    Path("results.txt").write_text("0")

    prev_wins = 0
    last_10 = []
    with tqdm(total=n_games, desc="Running Games", ncols=n_games) as pbar:
        for i in range(1, n_games+1):
            tuner.propose_change()
            dump_weights(tuner.get_weights(), "weights.json")

            # **No --weights flag here!**
            subprocess.run(
                [sys.executable, "-m", "referee", "agent", "agent2"],
            )
            
            with open("results.txt", "r") as f:
                try:
                    current_wins = int(f.read().strip())
                except ValueError:
                    current_wins = 0
            # 3) read the updated total from results.txt
            new_wins = int(Path("results.txt").read_text().strip() or "0")

            # 4) did we win this match?
            if len(last_10) < 10:
                last_10.append(new_wins - prev_wins) 
            else:
                last_10.pop(0)
                last_10.append(new_wins - prev_wins)
            if i%10 == 0:
                print("Average last 10: ", sum(last_10)*10, "% winrate")
            won = (new_wins > prev_wins)
            if won:
                print(f"Game {i} won")
            else:
                print(f"Game {i} lost")

            # 5) feed result into tuner
            tuner.evaluate_outcome(won)

            # 6) advance baseline and update progress
            prev_wins = new_wins
            pbar.set_postfix(wins=f"{current_wins}/{i}")
            pbar.update(1)

    wins = int(Path("results.txt").read_text().strip() or "0")
    print(f"\nFinal Success Rate: {wins / n_games:.2f}%")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv)>1 else 100
    main(n)

# --- Helper to write the current weight vector to disk ---
