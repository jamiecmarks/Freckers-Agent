import torch
import torch.nn as nn
import torch.optim as optim
import json
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import random

# --- Simple MLP to predict 4 heuristic weights ---
class WeightNet(nn.Module):
    def __init__(self, seed=None):
        super().__init__()
        if seed:
            torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # Raw output (no activation)
            nn.Sigmoid()  # Sigmoid activation to constrain outputs to [0, 1]
        )

    def forward(self, x):
        return self.model(x)

# --- Convert weights to a JSON file ---
def write_weights(path, weights):
    weights_dict = {
        "centrality": float(weights[0]),
        "double_jumps": float(weights[1]),
        "distance": float(weights[2]),
        "mobility": float(weights[3])
    }
    with open(path, "w") as f:
        json.dump(weights_dict, f)

# --- Training loop ---
def train(n_games=1000, batch_size=5):
    # Setup models & optimizers
    random.seed()  # Set a random seed for model initialization
    model_red = WeightNet(seed=random.randint(0, 10000))  # Random seed
    model_blue = WeightNet(seed=random.randint(0, 10000))  # Random seed
    optimizer_red = optim.Adam(model_red.parameters(), lr=0.01)
    optimizer_blue = optim.Adam(model_blue.parameters(), lr=0.01)

    loss_fn = nn.MSELoss()
    history = []

    # Create a log file for progress updates
    with open("game_output.log", "a") as log_file:
        for i in tqdm(range(1, n_games + 1), desc="Training"):
            model_red.eval()
            model_blue.eval()

            # Add slight noise to encourage exploration
            input_tensor = torch.tensor([[random.uniform(-1.0, 1.0)]], dtype=torch.float32)

            # Create empty lists to store batch results
            batch_scores = []
            batch_weights_red = []
            batch_weights_blue = []

            # Run batch of games
            for _ in range(batch_size):
                # 1. Predict weights
                weights_red = model_red(input_tensor).detach().numpy().flatten()
                weights_blue = model_blue(input_tensor).detach().numpy().flatten()

                # Debugging: Print raw weights before writing them
                log_file.write(f"Raw Weights (Red): {weights_red}\n")
                log_file.write(f"Raw Weights (Blue): {weights_blue}\n")

                # 2. Save weights to file
                write_weights("weights.json", weights_red)
                write_weights("weights2.json", weights_blue)

                # 3. Run a game (subprocess doesn't log to game_output.log now)
                subprocess.run([sys.executable, "-m", "referee", "agent", "agent2"])

                # 4. Get score (positive = red wins)
                try:
                    score = float(Path("eval.txt").read_text().strip())
                except:
                    score = 0.0

                batch_scores.append(score)
                batch_weights_red.append(weights_red)
                batch_weights_blue.append(weights_blue)

            # Calculate average score for this batch
            avg_score = sum(batch_scores) / batch_size
            history.append(avg_score)

            # 5. Train both models
            model_red.train()
            model_blue.train()

            # Process batch results
            for j in range(batch_size):
                weights_red = torch.tensor(batch_weights_red[j], dtype=torch.float32)
                weights_blue = torch.tensor(batch_weights_blue[j], dtype=torch.float32)

                pred_red = model_red(input_tensor)
                pred_blue = model_blue(input_tensor)

                # Create detached versions to use as fixed targets
                target_red = pred_red.detach().clone()
                target_blue = pred_blue.detach().clone()

                # Only reward the winner / punish the loser
                score = batch_scores[j]
                if score > 0:  # Red won
                    target_red += torch.tensor([[abs(score)]])
                    target_blue -= torch.tensor([[abs(score)]])
                elif score < 0:  # Blue won
                    target_red -= torch.tensor([[abs(score)]])
                    target_blue += torch.tensor([[abs(score)]])

                # Loss and update
                loss_red = loss_fn(pred_red, target_red)
                loss_blue = loss_fn(pred_blue, target_blue)

                optimizer_red.zero_grad()
                loss_red.backward()
                optimizer_red.step()

                optimizer_blue.zero_grad()
                loss_blue.backward()
                optimizer_blue.step()

            # Debugging: Print weights after update
            log_file.write(f"Updated Red Weights: {model_red.model[4].weight.data}\n")
            log_file.write(f"Updated Blue Weights: {model_blue.model[4].weight.data}\n")

            # Logging
            if i % 10 == 0:
                avg = sum(history[-10:]) / 10
                log_file.write(f"Game {i}: Last 10 avg score = {avg:.3f}\n")
            if i % 100 == 0:
                log_file.write(f"Red weights: {batch_weights_red[-1]}\n")
                log_file.write(f"Blue weights: {batch_weights_blue[-1]}\n")

        # Save models
        torch.save(model_red.state_dict(), "model_red.pth")
        torch.save(model_blue.state_dict(), "model_blue.pth")
        print("Training complete.")

# Entry point
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    train(n)
