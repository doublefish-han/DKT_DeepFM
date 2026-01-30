# Diagnostic sweep for DKT hyperparameters:
# - hidden size (HID)
# - sequence cap (truncate to last T steps)
# - gradient clipping (norm)
import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data/processed/dkt_sequences.pkl"
OUT_METRICS = BASE_DIR / "outputs/metrics"
OUT_FIGS = BASE_DIR / "outputs/figures"

DEFAULT_HIDS = [64, 128, 256]
DEFAULT_CAPS = [100, 200, 300]
DEFAULT_CLIPS = [None, 1.0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KTDataset(Dataset):
    def __init__(self, path: Path, seq_cap: Optional[int] = None):
        import pickle

        self.data = pickle.load(open(path, "rb"))
        self.seq_cap = seq_cap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s = self.data[i]
        skills = torch.tensor(s["skills"], dtype=torch.long)
        correct = torch.tensor(s["correct"], dtype=torch.float32)
        if self.seq_cap and len(skills) > self.seq_cap:
            skills = skills[-self.seq_cap :]
            correct = correct[-self.seq_cap :]
        interactions = skills * 2 + correct.long() + 1  # +1 reserves 0 for padding
        return interactions, correct


def pad_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    interactions, labels = zip(*batch)
    lens = torch.tensor([len(x) for x in interactions])
    maxlen = lens.max()
    pad_i = torch.stack(
        [torch.cat([x, torch.zeros(maxlen - len(x), dtype=torch.long)]) for x in interactions]
    )
    pad_y = torch.stack(
        [torch.cat([y, torch.full((maxlen - len(y),), -1.0)]) for y in labels]
    )  # -1 as mask
    return pad_i, pad_y, lens


class DKT(nn.Module):
    def __init__(self, n_interaction: int, emb: int, hid: int):
        super().__init__()
        self.emb = nn.Embedding(n_interaction, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, interaction_ids):
        x = self.emb(interaction_ids)
        h, _ = self.lstm(x)
        logits = self.fc(h).squeeze(-1)
        return logits, h


def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    all_logits = []
    all_labels = []
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for s, y, _ in loader:
            s, y = s.to(DEVICE), y.to(DEVICE)
            logits, _ = model(s)
            mask = y >= 0
            loss = (bce(logits, torch.clamp(y, 0, 1)) * mask).sum() / mask.sum()
            total_loss += loss.item()
            n_batches += 1

            all_logits.append(logits[mask].detach().cpu())
            all_labels.append(y[mask].detach().cpu())
    if not all_logits:
        return math.nan, math.nan, math.nan
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    probs = torch.sigmoid(logits_cat).numpy()
    labels_np = labels_cat.numpy()
    auc = roc_auc_score(labels_np, probs)
    probs_safe = np.clip(probs, 1e-15, 1 - 1e-15)
    ll = log_loss(labels_np, probs_safe)
    return auc, ll, total_loss / max(n_batches, 1)


def run_single(
    hidden_size: int,
    seq_cap: Optional[int],
    grad_clip: Optional[float],
    emb: int,
    epochs: int,
    batch_size: int,
    seed: int,
):
    set_seed(seed)
    ds = KTDataset(DATA, seq_cap=seq_cap)
    n_skill = 1 + max(max(d["skills"]) for d in ds.data)
    n_interaction = 2 * n_skill + 1
    # train/val split (80/20)
    val_len = max(1, int(0.2 * len(ds)))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

    model = DKT(n_interaction, emb=emb, hid=hidden_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    history = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for s, y, _ in train_loader:
            s, y = s.to(DEVICE), y.to(DEVICE)
            logits, _ = model(s)
            mask = y >= 0
            loss = (bce(logits, torch.clamp(y, 0, 1)) * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)
        val_auc, val_ll, val_loss = evaluate(model, val_loader)
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_logloss": val_ll,
            }
        )
        print(
            f"[HID={hidden_size} cap={seq_cap} clip={grad_clip}] "
            f"Epoch {ep+1}: train_loss={train_loss:.4f} val_auc={val_auc:.4f} val_loss={val_loss:.4f}"
        )
    return history


def plot_curve(history: List[dict], save_path: Path, title: str):
    if not HAS_MPL:
        return
    epochs = [h["epoch"] for h in history]
    aucs = [h["val_auc"] for h in history]
    losses = [h["val_loss"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, aucs, marker="o", label="Val AUC")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val AUC")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(epochs, losses, marker="s", color="tab:red", label="Val Loss")
    ax2.set_ylabel("Val Loss")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run DKT sensitivity diagnostics.")
    parser.add_argument("--hids", type=int, nargs="+", default=DEFAULT_HIDS)
    parser.add_argument("--caps", type=int, nargs="+", default=DEFAULT_CAPS)
    parser.add_argument("--clips", type=float, nargs="+", default=[v for v in DEFAULT_CLIPS if v is not None])
    parser.add_argument("--include_no_clip", action="store_true", help="Include no-clip runs in addition to provided clip norms.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--emb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    clips: List[Optional[float]] = []
    if args.include_no_clip or not args.clips:
        clips.append(None)
    clips.extend(args.clips)

    csv_path = OUT_METRICS / "dkt_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["hidden_size", "seq_cap", "grad_clip", "epoch", "train_loss", "val_loss", "val_logloss", "val_auc"]
        )
        for hid in args.hids:
            for cap in args.caps:
                for clip in clips:
                    history = run_single(
                        hidden_size=hid,
                        seq_cap=cap,
                        grad_clip=clip,
                        emb=args.emb,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        seed=args.seed,
                    )
                    for h in history:
                        writer.writerow(
                            [
                                hid,
                                cap,
                                "none" if clip is None else clip,
                                h["epoch"],
                                f"{h['train_loss']:.6f}",
                                f"{h['val_loss']:.6f}",
                                f"{h['val_logloss']:.6f}",
                                f"{h['val_auc']:.6f}",
                            ]
                        )
                    # Plot one figure per setting
                    clip_tag = "noclip" if clip is None else f"clip{clip}"
                    fig_name = f"dkt_val_curve_h{hid}_cap{cap}_{clip_tag}.png"
                    plot_curve(
                        history,
                        OUT_FIGS / fig_name,
                        title=f"DKT Val (H={hid}, cap={cap}, clip={clip_tag})",
                    )
    print(f"Saved sensitivity metrics to {csv_path}")


if __name__ == "__main__":
    main()
