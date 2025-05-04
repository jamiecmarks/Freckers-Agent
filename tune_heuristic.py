import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import sys
from pathlib import Path
import random
import json

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
def train(n_games=1000):
    # Setup models & optimizers
    random.seed()  # Set a random seed for model initialization
    model_red = WeightNet(seed=random.randint(0, 10000))  # Random seed
    model_blue = WeightNet(seed=random.randint(0, 10000))  # Random seed
    optimizer_red = optim.Adam(model_red.parameters(), lr=0.002)
    optimizer_blue = optim.Adam(model_blue.parameters(), lr=0.002)

    loss_fn = nn.MSELoss()
    history = []

    # Create a log file for progress updates
    with open("game_output.log", "a") as log_file:
        for i in range(1, n_games + 1):
            model_red.eval()
            model_blue.eval()

            # Add slight noise to encourage exploration
            input_tensor = torch.tensor([[random.uniform(-1.0, 1.0)]], dtype=torch.float32)

            # Predict initial weights
            weights_red = model_red(input_tensor).detach().numpy().flatten()
            weights_blue = model_blue(input_tensor).detach().numpy().flatten()

            # Log initial weights
            log_file.write(f"Initial Weights (Red): {weights_red}\n")
            log_file.write(f"Initial Weights (Blue): {weights_blue}\n")

            # Run a game
            subprocess.run([sys.executable, "-m", "referee", "agent", "agent2"])

            # Read game outcome (positive = red wins, negative = blue wins)
            try:
                score = float(Path("eval.txt").read_text().strip())
            except:
                score = 0.0

            # Read advantage vectors
            try:
                adv_red = list(map(float, Path("red_advantage.txt").read_text().strip().split()))
                adv_blue = list(map(float, Path("blue_advantage.txt").read_text().strip().split()))
            except:
                adv_red = [0.0] * 4  # Default to zeros if the advantage file isn't available
                adv_blue = [0.0] * 4  # Default to zeros if the advantage file isn't available

            # Log advantages
            log_file.write(f"Red Advantage: {adv_red}\n")
            log_file.write(f"Blue Advantage: {adv_blue}\n")

            # Train both models based on the outcome
            model_red.train()
            model_blue.train()

            # Adjust weights based on advantage vectors
            if score > 0:  # Red wins
                # Reinforce red's heuristics, punish blue's
                target_red = torch.tensor(adv_red, dtype=torch.float32) + abs(score)
                target_blue = torch.tensor(adv_blue, dtype=torch.float32) - abs(score)
            elif score < 0:  # Blue wins
                # Reinforce blue's heuristics, punish red's
                target_red = torch.tensor(adv_red, dtype=torch.float32) - abs(score)
                target_blue = torch.tensor(adv_blue, dtype=torch.float32) + abs(score)
            else:
                target_red = torch.tensor(adv_red, dtype=torch.float32)
                target_blue = torch.tensor(adv_blue, dtype=torch.float32)

            # Loss and update
            loss_red = loss_fn(model_red(input_tensor), target_red)
            loss_blue = loss_fn(model_blue(input_tensor), target_blue)

            optimizer_red.zero_grad()
            loss_red.backward()
            optimizer_red.step()

            optimizer_blue.zero_grad()
            loss_blue.backward()
            optimizer_blue.step()

            # Log updated weights
            log_file.write(f"Updated Red Weights: {model_red.model[4].weight.data}\n")
            log_file.write(f"Updated Blue Weights: {model_blue.model[4].weight.data}\n")
            # Predict initial weights
            weights_red = model_red(input_tensor).detach().numpy().flatten()
            weights_blue = model_blue(input_tensor).detach().numpy().flatten()

            # --- 🔧 WRITE THEM TO FILE ---
            write_weights("weights.json", weights_red)
            write_weights("weights2.json", weights_blue)   

            # Logging every 10 games
            if i % 10 == 0:
                avg_score = sum(history[-10:]) / 10 if history else 0
                log_file.write(f"Game {i}: Last 10 avg score = {avg_score:.3f}\n")

        # Save models
        torch.save(model_red.state_dict(), "model_red.pth")
        torch.save(model_blue.state_dict(), "model_blue.pth")
        print("Training complete.")

# Entry point
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    train(n)
