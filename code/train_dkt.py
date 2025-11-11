import os
import pickle, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data/processed/dkt_sequences.pkl"
LOG_DIR = BASE_DIR / "outputs/logs"

EMB = 64
HID = 128
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

class KTDataset(Dataset):
    def __init__(self, path):
        self.data = pickle.load(open(path, "rb"))
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        skills = torch.tensor(s["skills"], dtype=torch.long)
        correct = torch.tensor(s["correct"], dtype=torch.float32)
        interactions = skills * 2 + correct.long() + 1
        return interactions, correct

def pad_batch(batch):
    interactions, labels = zip(*batch)
    lens = torch.tensor([len(x) for x in interactions])
    maxlen = lens.max()
    pad_i = torch.stack([
        torch.cat([x, torch.zeros(maxlen - len(x), dtype=torch.long)])
        for x in interactions
    ])
    pad_y = torch.stack([
        torch.cat([y, torch.full((maxlen - len(y),), -1.0)])
        for y in labels
    ])
    return pad_i, pad_y, lens

class DKT(nn.Module):
    def __init__(self, n_interaction):
        super().__init__()
        self.emb = nn.Embedding(n_interaction, EMB, padding_idx=0)
        self.lstm = nn.LSTM(EMB, HID, batch_first=True)
        self.fc = nn.Linear(HID, 1)
    def forward(self, interaction_ids):
        x = self.emb(interaction_ids)
        h,_ = self.lstm(x)
        logits = self.fc(h).squeeze(-1)
        return logits, h

def train():
    ds = KTDataset(DATA)
    n_skill = 1 + max(max(d["skills"]) for d in ds.data)
    n_interaction = 2 * n_skill + 1
    model = DKT(n_interaction).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=pad_batch)
    epoch_losses = []
    for ep in range(EPOCHS):
        model.train()
        total = 0; n = 0
        for s, y, lens in loader:
            s, y = s.to(DEVICE), y.to(DEVICE)
            logits, _ = model(s)
            mask = (y >= 0)
            loss = (bce(logits, torch.clamp(y,0,1))*mask).sum()/mask.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        avg_loss = total/n
        epoch_losses.append(avg_loss)
        print(f"Epoch {ep+1}: loss={avg_loss:.4f}")

    
    model.eval()
    all_h = []
    with torch.no_grad():
        for s, y, lens in DataLoader(ds, batch_size=64, shuffle=False, collate_fn=pad_batch):
            s = s.to(DEVICE)
            _, h = model(s)
            for i, L in enumerate(lens.tolist()):
                all_h.append(h[i,:L,:].cpu())  
    hidden_path = BASE_DIR / "data/processed/dkt_hidden_states.pt"
    torch.save(all_h, hidden_path)
    print(f"Saved hidden states to {hidden_path}")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = LOG_DIR / "dkt_training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,loss\n")
        for idx, loss in enumerate(epoch_losses, start=1):
            f.write(f"{idx},{loss:.6f}\n")
    print(f"Saved loss log to {log_path}")

    if HAS_MPL:
        plt.figure()
        plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("DKT Training Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        curve_path = LOG_DIR / "dkt_training_curve.png"
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved training curve to {curve_path}")
    else:
        print("matplotlib not available, skipped plotting curve.")

if __name__ == "__main__":
    train()
