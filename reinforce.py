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
        # raw parameters; sigmoid will map to [0,1]
        self.params = nn.Parameter(torch.randn(4))

    def forward(self):
        return torch.sigmoid(self.params)

class AdvantageTracker:
    def __init__(self, n):
        self.sum = np.zeros(n)
        self.sum_sq = np.zeros(n)
        self.delta_sum = np.zeros(n)
        self.delta_sum_sq = np.zeros(n)
        self.count = 0

    def update(self, adv, delta):
        adv = np.clip(adv, -10, 10)
        delta = np.clip(delta, -10, 10)
        self.sum += adv
        self.sum_sq += adv**2
        self.delta_sum += delta
        self.delta_sum_sq += delta**2
        self.count += 1

    def mean(self):
        return self.sum / (self.count + 1e-8)

    def std(self):
        m = self.mean()
        return np.sqrt(self.sum_sq/(self.count+1e-8) - m**2 + 1e-8)

    def delta_mean(self):
        return self.delta_sum/(self.count+1e-8)

    def delta_std(self):
        m = self.delta_mean()
        return np.sqrt(self.delta_sum_sq/(self.count+1e-8) - m**2 + 1e-8)

def write_weights(path, weights):
    d = {"centrality": float(weights[0]),
         "double_jumps": float(weights[1]),
         "distance": float(weights[2]),
         "mobility": float(weights[3])}
    with open(path, 'w') as f:
        json.dump(d, f)

def plot_metrics(history):
    games = [h['game'] for h in history]
    labels = ['centrality', 'double_jumps', 'distance', 'mobility']

    # ---------- Weights Evolution ----------
    plt.figure(figsize=(10,6))
    # collect both red and blue weight arrays
    w_red  = np.array([h['weights_red']  for h in history])
    w_blue = np.array([h['weights_blue'] for h in history])
    # plot each feature for both agents
    for i, lab in enumerate(labels):
        plt.plot(games, w_red[:, i],  label=f'red_{lab}')
        plt.plot(games, w_blue[:, i], '--', label=f'blue_{lab}')
    plt.title('Weights Evolution')
    plt.xlabel('Game')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    plt.savefig('weights.png')
    plt.close()

    # ---------- Training Loss ----------
    plt.figure(figsize=(10,6))
    plt.plot(games, [h['loss_red']  for h in history], label='Red Loss')
    plt.plot(games, [h['loss_blue'] for h in history], label='Blue Loss')
    plt.title('Training Loss')
    plt.xlabel('Game')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('losses.png')
    plt.close()

    # ---------- Win Rate vs Baseline ----------
    if any('win_rate' in h for h in history):
        bs = [h['game']     for h in history if 'win_rate' in h]
        wr = [h['win_rate'] for h in history if 'win_rate' in h]
        plt.figure(figsize=(10,6))
        plt.plot(bs, wr, label='Win Rate vs Baseline')
        plt.title('Win Rate Against Baseline')
        plt.xlabel('Game')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig('win_rates.png')
        plt.close()

def evaluate_baseline(model, n_games=10):
    baseline = np.array([0.25]*4)
    w = model().detach().numpy()
    wins=losses=draws=0
    for _ in range(n_games):
        write_weights('weights.json', w)
        write_weights('weights2.json', baseline)
        subprocess.run([sys.executable, '-m', 'referee','agent','agent2'], stdout=subprocess.DEVNULL)
        try:
            score = float(Path('eval.txt').read_text().strip())
            if score>0: wins+=1
            elif score<0: losses+=1
            else: draws+=1
        except: draws+=1
    return wins/n_games, losses/n_games, draws/n_games

def train(n_games=1000):
    model_red  = WeightNet(seed=random.randint(0,10000))
    model_blue = WeightNet(seed=random.randint(0,10000))

    manual_weights_red  = torch.tensor([0.5, 0.5, 0.5, 0.5])
    manual_weights_blue = torch.tensor([0.5, 0.5, 0.5, 0.5])

    with torch.no_grad():
        model_red.params.copy_(torch.logit(manual_weights_red.clamp(1e-6, 1 - 1e-6)))
        model_blue.params.copy_(torch.logit(manual_weights_blue.clamp(1e-6, 1 - 1e-6)))
    opt_red  = optim.Adam(model_red.parameters(),  lr=0.01)
    opt_blue = optim.Adam(model_blue.parameters(), lr=0.01)
    sched_red  = optim.lr_scheduler.StepLR(opt_red,  step_size=200, gamma=0.5)
    sched_blue = optim.lr_scheduler.StepLR(opt_blue, step_size=200, gamma=0.5)

    tracker_red  = AdvantageTracker(4)
    tracker_blue = AdvantageTracker(4)
    history = []
    wins=losses=draws=0
    score_sma = 0.0

    for i in range(1, n_games+1):
        # generate noisy weights for exploration
        w_red  = model_red().detach().numpy()
        w_blue = model_blue().detach().numpy()
        noise = 0.05*(1 - i/n_games)
        if noise>0:
            w_red  = np.clip(w_red  + np.random.randn(4)*noise,  0,1)
            w_blue = np.clip(w_blue + np.random.randn(4)*noise,  0,1)

        write_weights('weights.json',  w_red)
        write_weights('weights2.json', w_blue)
        subprocess.run([sys.executable,'-m','referee','agent','agent2'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            score = float(Path('eval.txt').read_text().strip())
        except:
            score = 0.0
        score_sma = 0.9*score_sma + 0.1*score
        if score>0: wins+=1
        elif score<0: losses+=1
        else: draws+=1

        # compute feature deltas
        F_red  = np.loadtxt('red_pv_features.csv',  delimiter=',', skiprows=1)
        delta_red = np.sum(np.abs(np.diff(F_red,axis=0)),axis=0)[1:]
        z_red = ((delta_red - tracker_red.delta_mean()) / (tracker_red.delta_std()+1e-8)
                 if tracker_red.count>0 else delta_red/ (np.sum(np.abs(delta_red))+1e-8))
        z_red = z_red / (np.sum(w_red) + 1e-8)
        adv_red = z_red * score_sma
        tracker_red.update(adv_red, delta_red)

        F_blue = np.loadtxt('blue_pv_features.csv', delimiter=',', skiprows=1)
        delta_blue = np.sum(np.abs(np.diff(F_blue,axis=0)),axis=0)[1:]
        z_blue = ((delta_blue - tracker_blue.delta_mean()) / (tracker_blue.delta_std()+1e-8)
                  if tracker_blue.count>0 else delta_blue/(np.sum(np.abs(delta_blue))+1e-8))

        z_blue = z_blue / (np.sum(w_blue) + 1e-8)
        adv_blue = z_blue * (score_sma) # Modified to fix inverse issue?
        tracker_blue.update(adv_blue, delta_blue)
            


        winner = "red" if score > 0 else "blue" if score < 0 else "draw"


        print(f"game {i} : winner - {winner}")
        print("RED metrics")
        print(f"advantage vector: centrality = {delta_red[0]*score_sma:>8.4f}, double_jumps = {delta_red[1]*score_sma:>8.4f}, distance = {delta_red[2]*score_sma:>8.4f}, mobility = {delta_red[3]*score_sma:>8.4f}")
        print(f"normalised advantage vector: centrality = {adv_red[0]:>8.4f}, double_jumps = {adv_red[1]:>8.4f}, distance = {adv_red[2]:>8.4f}, mobility = {adv_red[3]:>8.4f}")
        print(f"feature weights: centrality = {w_red[0]:>8.4f}, double_jumps = {w_red[1]:>8.4f}, distance = {w_red[2]:>8.4f}, mobility = {w_red[3]:>8.4f}")
        print()
        # Print requested metrics for blue agent
        print("BLUE metrics")
        print(f"advantage vector: centrality = {delta_blue[0]*score_sma:>8.4f}, double_jumps = {delta_blue[1]*score_sma:>8.4f}, distance = {delta_blue[2]*score_sma:>8.4f}, mobility = {delta_blue[3]*score_sma:>8.4f}")
        print(f"normalised advantage vector: centrality = {adv_blue[0]:>8.4f}, double_jumps = {adv_blue[1]:>8.4f}, distance = {adv_blue[2]:>8.4f}, mobility = {adv_blue[3]:>8.4f}")
        print(f"feature weights: centrality = {w_blue[0]:>8.4f}, double_jumps = {w_blue[1]:>8.4f}, distance = {w_blue[2]:>8.4f}, mobility = {w_blue[3]:>8.4f}")
        print()
        # --- UPDATED LOSS BLOCK ---
        adv_t_red  = torch.tensor(adv_red,  dtype=torch.float32)
        adv_t_blue = torch.tensor(adv_blue, dtype=torch.float32)
        w_r = model_red()
        w_b = model_blue()
        loss_red  = - torch.dot(w_r, adv_t_red)
        loss_blue = - torch.dot(w_b, adv_t_blue)

        opt_red.zero_grad()
        loss_red.backward()
        torch.nn.utils.clip_grad_norm_(model_red.parameters(), max_norm=0.5)
        opt_red.step()
        sched_red.step()

        opt_blue.zero_grad()
        loss_blue.backward()
        torch.nn.utils.clip_grad_norm_(model_blue.parameters(), max_norm=0.5)
        opt_blue.step()
        sched_blue.step()
        # --- END LOSS BLOCK ---

        history.append({
            'game':i,
            'loss_red': loss_red.item(),
            'loss_blue': loss_blue.item(),
            'win_rate': wins/(wins+losses+draws+1e-8) if i%50==0 else None,
            'weights_red': w_red.tolist(),
            'weights_blue': w_blue.tolist()
        })

        if i%10==0:
            plot_metrics(history)
            pd.DataFrame(history).to_csv('training_metrics.csv', index=False)

    torch.save(model_red.state_dict(),'model_red.pth')
    torch.save(model_blue.state_dict(),'model_blue.pth')
    plot_metrics(history)
    pd.DataFrame(history).to_csv('training_metrics.csv', index=False)

if __name__ == "__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 1000
    train(n)
