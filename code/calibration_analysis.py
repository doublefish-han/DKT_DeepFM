"""Calibration diagnostics (Reliability diagram, ECE, Brier).

This script is intentionally *post-hoc*: it does NOT train models.
It reads saved prediction probabilities and labels, then outputs:
- reliability diagram(s)
- ECE / Brier (per model)

Expected input CSV columns (defaults):
- y_true: 0/1 ground truth
- y_pred_prob: predicted probability in [0,1]
- model (optional): model name; if present, metrics/plots are produced per model

"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_OUT_DIR = "outputs"
DEFAULT_METRICS_DIR = os.path.join(DEFAULT_OUT_DIR, "metrics")
DEFAULT_FIG_DIR = os.path.join(DEFAULT_OUT_DIR, "figures")


@dataclass
class CalibrationResult:
    ece: float
    brier: float
    n: int


def _ensure_dirs(metrics_dir: str, fig_dir: str) -> None:
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)


def _auto_find_predictions_csvs(metrics_dir: str) -> list[str]:
    """Find likely per-model prediction CSVs under outputs/metrics.

    By default, we prefer per-run files like '*_test_predictions.csv' / '*_val_predictions.csv'
    (including 'logreg_seedXX_test_predictions.csv'), and we exclude summary/CI artifacts.

    Returns a list of file paths (possibly empty) sorted by mtime (newest first).
    """
    patterns = [
        os.path.join(metrics_dir, "*_test_predictions.csv"),
        os.path.join(metrics_dir, "*_val_predictions.csv"),
        os.path.join(metrics_dir, "*pred*.csv"),
        os.path.join(metrics_dir, "*prediction*.csv"),
        os.path.join(metrics_dir, "*preds*.csv"),
    ]

    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    # Deduplicate
    candidates = sorted(set(candidates))

    # Filter out common summary/CI files
    bad_keywords = ["summary", "ci", "significance", "combined_metrics", "calibration_metrics"]
    filtered = [
        p for p in candidates
        if not any(k in os.path.basename(p).lower() for k in bad_keywords)
    ]

    filtered.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return filtered


def _resolve_column(df: pd.DataFrame, preferred: str, aliases: list[str]) -> str:
    """Return an existing column name.

    - If `preferred` exists, use it.
    - Else try aliases in order.
    - Else raise a clear error.
    """
    if preferred in df.columns:
        return preferred
    for a in aliases:
        if a in df.columns:
            return a
    raise ValueError(f"Missing required column '{preferred}'. Available columns={list(df.columns)}")


def _bin_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bin empirical accuracy and average confidence.

    Returns:
      bin_acc: empirical fraction of positives per bin
      bin_conf: mean predicted probability per bin
      bin_count: number of samples per bin

    Bins are equal-width in [0,1].
    """
    y_true = y_true.astype(float)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)

    # Bin index in [0, n_bins-1]
    bin_idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)

    bin_count = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_sum_true = np.bincount(bin_idx, weights=y_true, minlength=n_bins).astype(float)
    bin_sum_prob = np.bincount(bin_idx, weights=y_prob, minlength=n_bins).astype(float)

    # Avoid divide by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_acc = np.where(bin_count > 0, bin_sum_true / bin_count, np.nan)
        bin_conf = np.where(bin_count > 0, bin_sum_prob / bin_count, np.nan)

    return bin_acc, bin_conf, bin_count


def compute_ece_brier(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[CalibrationResult, Dict[str, np.ndarray]]:
    """Compute ECE and Brier, plus bin-wise stats for plotting."""
    y_true = y_true.astype(float)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)

    bin_acc, bin_conf, bin_count = _bin_stats(y_true, y_prob, n_bins=n_bins)
    n = int(len(y_true))

    # ECE: sum_k (n_k / n) * |acc_k - conf_k|
    weights = np.where(bin_count > 0, bin_count / max(n, 1), 0.0)
    diffs = np.where(np.isfinite(bin_acc) & np.isfinite(bin_conf), np.abs(bin_acc - bin_conf), 0.0)
    ece = float(np.sum(weights * diffs))

    # Brier score: mean (p - y)^2
    brier = float(np.mean((y_prob - y_true) ** 2))

    result = CalibrationResult(ece=ece, brier=brier, n=n)
    stats = {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count}
    return result, stats


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    fig_path: str,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram with a small probability histogram."""
    res, stats = compute_ece_brier(y_true, y_prob, n_bins=n_bins)

    bin_acc = stats["bin_acc"]
    bin_conf = stats["bin_conf"]
    bin_count = stats["bin_count"]

    # x positions: use bin_conf (mean probability in each bin); if NaN, fall back to bin centers
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    x = np.where(np.isfinite(bin_conf), bin_conf, bin_centers)
    y = bin_acc

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    FONT_SIZE = 12

    # Top: reliability
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    mask = bin_count > 0
    ax1.plot(x[mask], y[mask], marker="o", linewidth=1)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # light gray grid for readability
    ax1.grid(True, linestyle="--", linewidth=0.6, color="lightgray", alpha=0.8)

    ax1.set_ylabel("Empirical accuracy", fontsize=FONT_SIZE)
    
    ax1.set_title(f"Reliability Diagram: {model_name}", fontsize=FONT_SIZE)

    metrics_text = f"ECE={res.ece:.4f}\nBrier={res.brier:.4f}\nN={res.n}"
    ax1.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "lightgray", "alpha": 0.9},
    )

    # unify tick label font size
    ax1.tick_params(axis="both", labelsize=FONT_SIZE)

    # Bottom: histogram
    ax2.hist(np.clip(y_prob, 0, 1), bins=bin_edges)
    ax2.set_xlim(0, 1)

    # (No grid for histogram, per journal style)

    ax2.set_ylabel("Count", fontsize=FONT_SIZE)
    ax2.set_xlabel("Predicted probability (mean in bin)", fontsize=FONT_SIZE)

    # draw full box around histogram (no gridlines)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # unify tick label font size
    ax2.tick_params(axis="both", labelsize=FONT_SIZE)

    fig.subplots_adjust(hspace=0.18)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc calibration diagnostics (ECE/Brier + reliability diagram).")
    parser.add_argument(
        "--preds_csv",
        type=str,
        default=None,
        help="Path(s) to predictions CSV. Can be a single path, a glob (e.g. 'outputs/metrics/*_test_predictions.csv'), or comma-separated list. If omitted, auto-detect under outputs/metrics.",
    )
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Base outputs directory (default: outputs).")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of equal-width bins in [0,1].")
    parser.add_argument("--y_true_col", type=str, default="label", help="Ground-truth column name.")
    parser.add_argument("--y_prob_col", type=str, default="prediction", help="Predicted probability column name.")
    parser.add_argument("--model_col", type=str, default="model", help="Optional model name column (if missing, will infer from filename).")

    args = parser.parse_args()

    metrics_dir = os.path.join(args.out_dir, "metrics")
    fig_dir = os.path.join(args.out_dir, "figures")
    _ensure_dirs(metrics_dir, fig_dir)

    # Build a list of prediction CSVs to process.
    preds_paths: list[str]
    if args.preds_csv is not None:
        # Allow comma-separated list or glob patterns.
        raw = [p.strip() for p in args.preds_csv.split(",") if p.strip()]
        expanded: list[str] = []
        for p in raw:
            matches = glob.glob(p)
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(p)
        preds_paths = [p for p in expanded if os.path.exists(p)]
    else:
        preds_paths = _auto_find_predictions_csvs(metrics_dir)

    if not preds_paths:
        raise FileNotFoundError(
            "Could not find any predictions CSVs.\n"
            "Expected CSV(s) containing columns like label/y_true and prediction/y_pred_prob.\n\n"
            "Fix options (pick one):\n"
            "1) Provide --preds_csv path explicitly (can be comma-separated or a glob).\n"
            "2) Ensure outputs/metrics contains '*_test_predictions.csv' or similar files.\n"
        )

    rows: list[dict] = []

    # Process each predictions file independently.
    for preds_path in preds_paths:
        df = pd.read_csv(preds_path)

        base_file = os.path.basename(preds_path).lower()
        if "_test_predictions" in base_file:
            split = "test"
        elif "_val_predictions" in base_file:
            split = "val"
        else:
            split = "unknown"

        # Resolve required columns with sensible aliases
        y_true_col = _resolve_column(df, args.y_true_col, ["y_true", "label", "target", "y", "gt"])
        y_prob_col = _resolve_column(df, args.y_prob_col, ["y_pred_prob", "prediction", "pred", "prob", "y_prob", "p"])

        # If model column missing, infer model name from filename
        if args.model_col in df.columns:
            groups = dict(tuple(df.groupby(args.model_col)))
        else:
            base = os.path.basename(preds_path)
            for suf in ["_test_predictions.csv", "_val_predictions.csv", ".csv"]:
                if base.endswith(suf):
                    base = base[: -len(suf)]
                    break
            inferred_model = base
            groups = {inferred_model: df}

        for model_name, g in groups.items():
            y_true = g[y_true_col].to_numpy(dtype=float)
            y_prob = g[y_prob_col].to_numpy(dtype=float)

            res, _ = compute_ece_brier(y_true, y_prob, n_bins=args.n_bins)
            rows.append({
                "model": str(model_name),
                "split": split,
                "n": res.n,
                "ece": res.ece,
                "brier": res.brier,
                "preds_csv": os.path.relpath(preds_path, start=args.out_dir) if os.path.isabs(preds_path) else preds_path,
            })

            safe_name = str(model_name).replace(" ", "_").replace("/", "_")
            fig_path = os.path.join(fig_dir, f"calibration_reliability_{safe_name}_{split}.png")
            plot_reliability_diagram(
                y_true,
                y_prob,
                model_name=f"{model_name} ({split})",
                fig_path=fig_path,
                n_bins=args.n_bins,
            )

        print(f"[OK] Loaded predictions: {preds_path}")

    out_csv = os.path.join(metrics_dir, "calibration_metrics.csv")
    out_df = pd.DataFrame(rows).sort_values(by=["model", "split", "preds_csv"])
    out_df.to_csv(out_csv, index=False)

    print(f"[OK] Wrote calibration metrics: {out_csv}")
    print(f"[OK] Wrote reliability diagrams to: {fig_dir}")


if __name__ == "__main__":
    main()
