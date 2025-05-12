import subprocess
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

python_path = sys.executable

# Parameters
n_games_per_setting = 20
depth_values = [ 6, 14, 20, 30, 40]

results = []

    # Set swap depths

with open("results.txt", "w") as f:
    f.write("0")

total_wins = 0
for i in tqdm(range(n_games_per_setting), 
                leave=False, ncols=80):
    subprocess.run([python_path, "-m", "referee", "agent", "agent2"],
                   )

    with open("eval.txt", "r") as f:
        result = float(f.read().strip())
        if result > 0:
            total_wins += 1
            print("Red wins (hybrid)")
        else:
            print("Blue wins (MCTS only)")

win_rate = total_wins / n_games_per_setting
print(f"    vs Blue Depth {blue_depth}: {total_wins}/{n_games_per_setting} = {win_rate:.2%}")




