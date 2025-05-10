import subprocess
import sys
from tqdm import tqdm

# Step 1: Set up Python executable path
python_path = sys.executable

# Step 2: Define the number of games to run
n = 10000  # Change this as needed

# Step 3: Reset results.txt to "0"
with open("results.txt", "w") as f:
    f.write("0")

# Step 4: Run the games with live win stats and a progress bar
with tqdm(total=n, desc="Running Games", ncols=100) as pbar:
    total_wins = 0
    for i in range(1, n + 1):

        subprocess.run(
            [python_path, "-m", "referee", "agentblue", "agentred"],
             stderr=subprocess.DEVNULL
        )

        # Read current win count
        with open("eval.txt", "r") as f:
            output = float(f.read())
            if output > 0:
                total_wins +=1


        # Update progress bar and win stats
        pbar.set_postfix(wins=f"{total_wins}/{i}")
        pbar.update(1)

# Step 5: Final win rate
with open("results.txt", "r") as f:
    total_wins = int(f.read().strip())

success_rate = (total_wins / n) * 100 if n > 0 else 0
print(f"\nFinal Success Rate: {success_rate:.2f}%")
