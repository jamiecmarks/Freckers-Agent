import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import sys
from pathlib import Path
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class WeightNet(nn.Module):
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.params = nn.Parameter(torch.rand(4))

    def forward(self):
        return torch.sigmoid(self.params)

def write_weights(path, weights):
    """Write model weights to a JSON file."""
    weights_dict = {
        "centrality": float(weights[0]),
        "double_jumps": float(weights[1]),
        "distance": float(weights[2]),
        "mobility": float(weights[3])
    }
    with open(path, "w") as f:
        json.dump(weights_dict, f)

class AdvantageTracker:
    def __init__(self, n):
        """Initialize tracker for n features."""
        self.sum = np.zeros(n)
        self.sum_sq = np.zeros(n)
        self.delta_sum = np.zeros(n)
        self.delta_sum_sq = np.zeros(n)
        self.count = 0

    def update(self, adv, delta):
        """Update tracker with advantage and delta values."""
        adv = np.clip(adv, -10, 10)  # Prevent outliers
        delta = np.clip(delta, -10, 10)
        self.sum += adv
        self.sum_sq += adv**2
        self.delta_sum += delta
        self.delta_sum_sq += delta**2
        self.count += 1

    def mean(self):
        """Return mean of tracked advantage values."""
        return self.sum / (self.count + 1e-8)

    def std(self):
        """Return standard deviation of tracked advantage values."""
        m = self.mean()
        return np.sqrt(self.sum_sq / (self.count + 1e-8) - m**2 + 1e-8)

    def delta_mean(self):
        """Return mean of tracked delta values."""
        return self.delta_sum / (self.count + 1e-8)

    def delta_std(self):
        """Return standard deviation of tracked delta values."""
        m = self.delta_mean()
        return np.sqrt(self.delta_sum_sq / (self.count + 1e-8) - m**2 + 1e-8)

def evaluate_baseline(model, n_games=10):
    """Evaluate model against a baseline with random weights."""
    baseline_weights = np.array([0.25, 0.25, 0.25, 0.25])
    wins, losses, draws = 0, 0, 0
    model.eval()
    weights = model().detach().numpy()
    for _ in range(n_games):
        write_weights("weights.json", weights)
        write_weights("weights2.json", baseline_weights)
        subprocess.run([sys.executable, "-m", "referee", "agent", "agent2"], stdout=subprocess.DEVNULL)
        try:
            score = float(Path("eval.txt").read_text().strip())
            if score > 0:
                wins += 1
            elif score < 0:
                losses += 1
            else:
                draws += 1
        except:
            draws += 1
    return wins / n_games, losses / n_games, draws / n_games

def plot_metrics(history, n_games):
    """Plot weights, losses, and win rates."""
    games = [h['game'] for h in history]
    
    # Plot weights
    plt.figure(figsize=(10, 6))
    weights = np.array([h['weights_red'] for h in history])
    for i, label in enumerate(['centrality', 'double_jumps', 'distance', 'mobility']):
        plt.plot(games, weights[:, i], label=label)
    plt.title('Red Weights Evolution')
    plt.xlabel('Game')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('weights.png')
    plt.close()

    # Plot losses
    plt.figure(figsize=(10, 6))
    losses_red = [h['loss_red'] for h in history]
    losses_blue = [h['loss_blue'] for h in history]
    plt.plot(games, losses_red, label='Red Loss')
    plt.plot(games, losses_blue, label='Blue Loss')
    plt.title('Training Loss')
    plt.xlabel('Game')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('losses.png')
    plt.close()

    # Plot win rates (if available)
    baseline_games = [h['game'] for h in history if 'win_rate' in h]
    if baseline_games:
        win_rates = [h['win_rate'] for h in history if 'win_rate' in h]
        plt.figure(figsize=(10, 6))
        plt.plot(baseline_games, win_rates, label='Win Rate vs Baseline')
        plt.title('Win Rate Against Baseline')
        plt.xlabel('Game')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig('win_rates.png')
        plt.close()

def train(n_games=1000):
    """Train two WeightNet models with enhanced logging and visualization."""
    model_red = WeightNet(seed=random.randint(0, 10000))
    model_blue = WeightNet(seed=random.randint(0, 10000))
    optimizer_red = optim.Adam(model_red.parameters(), lr=0.01)
    optimizer_blue = optim.Adam(model_blue.parameters(), lr=0.01)
    scheduler_red = optim.lr_scheduler.StepLR(optimizer_red, step_size=200, gamma=0.5)
    scheduler_blue = optim.lr_scheduler.StepLR(optimizer_blue, step_size=200, gamma=0.5)
    loss_fn = nn.MSELoss()

    tracker_red = AdvantageTracker(4)
    tracker_blue = AdvantageTracker(4)
    history = []
    wins, losses, draws = 0, 0, 0
    score_sma = 0.0  # Simple moving average of score

    with open("game_output.log", "a") as log_file:
        for i in range(1, n_games + 1):
            model_red.eval()
            model_blue.eval()
            weights_red = model_red().detach().numpy()
            weights_blue = model_blue().detach().numpy()

            # Add exploration noise (decays over time)
            noise_scale = 0.05 * (1 - i / n_games)
            if noise_scale > 0:
                weights_red += np.random.normal(0, noise_scale, 4)
                weights_blue += np.random.normal(0, noise_scale, 4)
                weights_red = np.clip(weights_red, 0, 1)
                weights_blue = np.clip(weights_blue, 0, 1)

            write_weights("weights.json", weights_red)
            write_weights("weights2.json", weights_blue)

            subprocess.run([sys.executable, "-m", "referee", "agent", "agent2"], stdout=subprocess.DEVNULL)

            try:
                score = float(Path("eval.txt").read_text().strip())
                score_sma = 0.9 * score_sma + 0.1 * score  # Update SMA
                if score > 0:
                    wins += 1
                elif score < 0:
                    losses += 1
                else:
                    draws += 1
            except Exception as e:
                score = 0.0
                log_file.write(f"Game {i}: Error reading eval.txt: {e}\n")

            F_red = np.loadtxt("red_pv_features.csv", delimiter=",", skiprows=1)
            delta_red = np.sum(np.abs(np.diff(F_red, axis=0)), axis=0)[1:]
            if tracker_red.count > 0:
                delta_red_z = (delta_red - tracker_red.delta_mean()) / (tracker_red.delta_std() + 1e-8)
            else:
                delta_red_z = delta_red / (np.sum(np.abs(delta_red)) + 1e-8)
            adv_red = delta_red_z * score_sma  # Use smoothed score
            tracker_red.update(adv_red, delta_red)

            F_blue = np.loadtxt("blue_pv_features.csv", delimiter=",", skiprows=1)
            delta_blue = np.sum(np.abs(np.diff(F_blue, axis=0)), axis=0)[1:]
            if tracker_blue.count > 0:
                delta_blue_z = (delta_blue - tracker_blue.delta_mean()) / (tracker_blue.delta_std() + 1e-8)
            else:
                delta_blue_z = delta_blue / (np.sum(np.abs(delta_blue)) + 1e-8)
            adv_blue = delta_blue_z * (-score_sma)
            tracker_blue.update(adv_blue, delta_blue)

            model_red.train()
            model_blue.train()
            current_red = model_red().detach()
            current_blue = model_blue().detach()

            # Dynamic step size based on score magnitude
            step_size = 0.05 + 0.15 * min(abs(score_sma), 1.0)
            target_red = torch.clamp(current_red + step_size * torch.tensor(adv_red, dtype=torch.float32), 0, 1)
            target_blue = torch.clamp(current_blue + step_size * torch.tensor(adv_blue, dtype=torch.float32), 0, 1)

            loss_red = loss_fn(model_red(), target_red)
            loss_blue = loss_fn(model_blue(), target_blue)

            optimizer_red.zero_grad()
            loss_red.backward()
            torch.nn.utils.clip_grad_norm_(model_red.parameters(), max_norm=1.0)
            optimizer_red.step()
            scheduler_red.step()

            optimizer_blue.zero_grad()
            loss_blue.backward()
            torch.nn.utils.clip_grad_norm_(model_blue.parameters(), max_norm=1.0)
            optimizer_blue.step()
            scheduler_blue.step()

            # Compute weight change norm
            weight_change_norm = np.linalg.norm(model_red().detach().numpy() - weights_red)

            # Log metrics
            log_entry = {
                'game': i,
                'timestamp': datetime.now().isoformat(),
                'score': score,
                'score_sma': score_sma,
                'loss_red': loss_red.item(),
                'loss_blue': loss_blue.item(),
                'weights_red': weights_red.tolist(),
                'weights_blue': weights_blue.tolist(),
                'adv_red': adv_red.tolist(),
                'adv_blue': adv_blue.tolist(),
                'delta_red_z': delta_red_z.tolist(),
                'weight_change_norm': weight_change_norm
            }
            history.append(log_entry)

            log_file.write(f"Game {i}: Score={score:.2f}, SMAScore={score_sma:.2f}, "
                         f"LossRed={loss_red.item():.4f}, WeightsRed={weights_red.tolist()}, "
                         f"AdvRed={adv_red.tolist()}, WeightChangeNorm={weight_change_norm:.4f}\n")

            if i % 10 == 0:
                win_rate = wins / (wins + losses + draws + 1e-8)
                log_file.write(f"Game {i} Summary\n")
                log_file.write(f"Win/Loss/Draw: {wins}/{losses}/{draws}, WinRate={win_rate:.3f}\n")
                log_file.write(f"Avg Red Advantage: {tracker_red.mean().tolist()}\n")
                log_file.write(f"Red Delta Std: {tracker_red.delta_std().tolist()}\n")

            if i % 100 == 0:
                win_rate, loss_rate, draw_rate = evaluate_baseline(model_red)
                log_entry['win_rate'] = win_rate
                log_entry['loss_rate'] = loss_rate
                log_entry['draw_rate'] = draw_rate
                log_file.write(f"Game {i}: Baseline Eval - WinRate={win_rate:.3f}, "
                             f"LossRate={loss_rate:.3f}, DrawRate={draw_rate:.3f}\n")

            if i % 50 == 0:
                plot_metrics(history, n_games)
                pd.DataFrame(history).to_csv('training_metrics.csv', index=False)

        torch.save(model_red.state_dict(), "model_red.pth")
        torch.save(model_blue.state_dict(), "model_blue.pth")
        plot_metrics(history, n_games)
        pd.DataFrame(history).to_csv('training_metrics.csv', index=False)
        print("Training complete. Plots saved: weights.png, losses.png, win_rates.png")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    train(n)