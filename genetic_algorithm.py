import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np


def write_weights(path: str, keys: list, weights: np.ndarray):
    d = {k: float(w) for k, w in zip(keys, weights)}
    with open(path, 'w') as f:
        json.dump(d, f)


def evaluate_pair(pair: np.ndarray, keys: list, n_games: int = 10) -> float:
    n = len(keys)
    red_w = pair[:n]
    blue_w = pair[n:]
    write_weights('genetic_weights_red.json', keys, red_w)
    write_weights('genetic_weights_blue.json', keys, blue_w)
    wins = 0
    for _ in range(n_games):
        subprocess.run([sys.executable, '-m', 'referee', 'agent', 'agent2'],
                        stdout = subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            score = float(Path('eval.txt').read_text().strip())
            if score > 0:
                wins += 1
        except Exception:
            pass
    return wins / n_games


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    point = random.randrange(1, len(parent1))
    return np.concatenate([parent1[:point], parent2[point:]])


def mutate(weights: np.ndarray, mutation_rate: float, mutation_scale: float = 0.1) -> np.ndarray:
    for i in range(len(weights)):
        if random.random() < mutation_rate:
            weights[i] += random.gauss(0, mutation_scale)
            weights[i] = min(max(weights[i], 0.0), 1.0)
    return weights


def save_checkpoint(gen, population, scores, path='checkpoint.json'):
    checkpoint = {
        'generation': gen,
        'population': [ind.tolist() for ind in population],
        'scores': scores
    }
    with open(path, 'w') as f:
        json.dump(checkpoint, f)


def load_checkpoint(path='checkpoint.json'):
    if not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    population = [np.array(ind) for ind in data['population']]
    return data['generation'], population, data['scores']


def genetic_algorithm(
    feature_keys: list,
    pop_size: int = 30,
    generations: int = 100,
    games_per_eval: int = 20,
    elite_frac: float = 0.2,
    mutation_rate: float = 0.05,
    checkpoint_path: str = 'checkpoint.json'
):
    n = len(feature_keys)
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint:
        start_gen, population, scores = checkpoint
        print(f"Resuming from generation {start_gen}")
    else:
        start_gen = 1
        population = [np.random.rand(2 * n) for _ in range(pop_size)]
        scores = [0.0] * pop_size

    n_elite = max(1, int(elite_frac * pop_size))

    for gen in range(start_gen, generations + 1):
        print(f"Generation {gen}")
        for i, indiv in enumerate(population):
            scores[i] = evaluate_pair(indiv, feature_keys, games_per_eval+gen//10)
            print(f"  Pair {i}: red win rate = {scores[i]:.2f}")

        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        elites = [ind for (_, ind) in ranked[:n_elite]]
        best_score, best_pair = ranked[0]
        print(f"  Best pair win rate = {best_score:.2f}")

        save_checkpoint(gen + 1, population, scores, checkpoint_path)

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop

    best_pair = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)[0][1]
    red_best = best_pair[:n]
    blue_best = best_pair[n:]
    write_weights('genetic_weights_red.json', feature_keys, red_best)
    write_weights('genetic_weights_blue.json', feature_keys, blue_best)
    print(f"\nOptimal RED weights saved to genetic_weights_red.json")
    print(f"Optimal BLUE weights saved to genetic_weights_blue.json")
    return red_best, blue_best, best_score


if __name__ == '__main__':
    feature_keys = [
        'distance', 'goal_count', 'connectivity',
        'dispersion', 'mobility_diff', 'jump_mobility'
    ]
    pop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    generations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    games_per_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    genetic_algorithm(
        feature_keys,
        pop_size=pop_size,
        generations=generations,
        games_per_eval=games_per_eval
    )

