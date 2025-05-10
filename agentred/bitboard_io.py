import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torch.nn.functional as F



# === Constants ===
BOARD_N = 8
RECORD_STRUCT = struct.Struct("QQQb")  # 3x uint64 for lilly, red, blue bits; 1x int8 for result label
FLAG_FILE = ".bitboard_logged.flag"

# === I/O Functions ===

def save_bitboard_record(filepath, lilly, red, blue, result):
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, "ab") as f:
        f.write(RECORD_STRUCT.pack(int(lilly), int(red), int(blue), int(result)))


def save_game_record(filepath, history):
    win_file = "bitboards_win.bin"
    loss_file = "bitboards_loss.bin"
    if filepath == win_file:
        result = 1
    else:
        result = 0
    for lilly, red, blue, player in history:
        save_bitboard_record(filepath, lilly, red, blue, result)
    with open(FLAG_FILE, "w") as f:
        f.write("logged")


def reset_logging_flag():
    if os.path.exists(FLAG_FILE):
        os.remove(FLAG_FILE)


def load_bitboard_records(filepath):
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "rb") as f:
        chunk = f.read(RECORD_STRUCT.size)
        while chunk:
            lilly, red, blue, label = RECORD_STRUCT.unpack(chunk)
            records.append(((lilly, red, blue), label))
            chunk = f.read(RECORD_STRUCT.size)
    return records

# === Tensor Conversion ===

def bitboard_to_tensor(lilly, red, blue):
    tensor = np.zeros((3, BOARD_N, BOARD_N), dtype=np.float32)
    for i, bits in enumerate((lilly, red, blue)):
        for pos in range(BOARD_N * BOARD_N):
            if np.uint64(np.uint64(bits) >> np.uint64(pos)) & np.uint64(1):
                r, c = divmod(pos, BOARD_N)
                tensor[i, r, c] = 1.0
    return tensor


def evaluate_model(model, records, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X, y = batch_to_tensors(records)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    y_true = np.array(y, dtype=int)
    y_pred = (probs > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)

    print(f"\n=== Full Evaluation ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"AUC-ROC:   {auc:.3f}")




def batch_to_tensors(records):
    X = np.stack([bitboard_to_tensor(*bb) for bb, _ in records])
    y = np.array([label for _, label in records], dtype=np.float32)  # float for BCE loss
    return X, y

# === Dataset & Model ===

class BitboardDataset(Dataset):
    def __init__(self, records):
        X, y = batch_to_tensors(records)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class SmallBitboardNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # 3 input channels, 8 output channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # 8 input channels, 16 output channels
        self.fc1 = nn.Linear(16 * 8 * 8, 64)  # Flatten the 16x8x8 output from conv layers
        self.fc2 = nn.Linear(64, 1)  # Output a single value (win probability)

    def forward(self, x):
        # Apply convolutions and activation functions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the output of conv2 layer (after pooling, if used)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class BitboardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x))  # output is raw logit

# === Training ===

def train_model(epochs=10, batch_size=32, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    records = []
    records += load_bitboard_records("bitboards_win.bin")
    print(len(records))
    
    # Ensures we have a balanced set of states
    records += random.sample(load_bitboard_records("bitboards_loss.bin"), len(records))
    random.shuffle(records)
    split = int(0.8 * len(records))

    train_records = records[:split]
    test_records  = records[split:]
    if not records:
        print("No training data.")
        return None

    dataset = BitboardDataset(train_records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BitboardNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_batch).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch}/{epochs} — Loss: {total_loss/len(dataset):.4f}, Acc: {acc:.3f}")

    return model, test_records

# === Display & Evaluation ===

def print_board(lilly, red, blue):
    symbols = []
    for r in range(BOARD_N):
        row = []
        for c in range(BOARD_N):
            idx = r * BOARD_N + c
            bit = 1 << idx
            if red & bit:
                row.append("R")
            elif blue & bit:
                row.append("B")
            elif lilly & bit:
                row.append("L")
            else:
                row.append(".")
        symbols.append(" ".join(row))
    print("\n".join(symbols))


def test_model(model, records, num_samples=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X, y = batch_to_tensors(records)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()

    print(f"\n=== Random Sample Evaluations ===")
    indices = random.sample(range(len(records)), min(num_samples, len(records)))
    for i in indices:
        (lilly, red, blue), label = records[i]
        pred_prob = probs[i][0]
        print(f"\n--- Example {i+1} ---")
        print(f"True Label: {label}, Predicted Win Prob: {pred_prob:.3f}")
        print_board(lilly, red, blue)

# === Main ===

if __name__ == "__main__":
    reset_logging_flag()
    model, test_records = train_model(epochs=20)
    if model:
        torch.save(model.state_dict(), "bitboard_model.pt")
        test_model(model, test_records, num_samples=5)
        evaluate_model(model, test_records)
    else:
        print("No model trained.")


