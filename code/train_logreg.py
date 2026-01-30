"""Logistic Regression baseline for the same tabular splits used by DeepFM experiments.

Notes:
  - This baseline is intentionally simple: one-hot encode categorical columns and standardize numeric columns.
  - It does NOT use sequence models; it is a tabular sanity baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def _find_split_csv(base: Path, rel: str) -> Path:
    """Try a few common locations for split CSVs."""
    candidates = [
        base / "data" / "processed" / "train_test_split" / rel,
        base / "data" / "processed" / rel,
        base / "outputs" / "data" / rel,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {rel}. Tried: " + ", ".join(str(c) for c in candidates)
    )


def _pick_label_column(df: pd.DataFrame) -> str:
    for c in ["label", "correct", "y", "y_true", "target"]:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found. Columns={list(df.columns)}")


def _pick_rowid_column(df: pd.DataFrame) -> str:
    for c in ["row_id", "id", "index"]:
        if c in df.columns:
            return c
    # If none exists, create a stable row_id
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return "row_id"


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Simple grid search; fast and deterministic
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "outputs"
    metric_dir = out_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    train_path = _find_split_csv(base_dir, "train.csv")
    test_path = _find_split_csv(base_dir, "test.csv")

    # `val.csv` may not exist in this repo; DeepFM creates val by splitting train.
    # For LogReg, if val.csv is missing, we create a deterministic train/val split.
    try:
        val_path = _find_split_csv(base_dir, "val.csv")
    except FileNotFoundError:
        val_path = None

    train_df_full = pd.read_csv(train_path)
    label_col = _pick_label_column(train_df_full)

    if val_path is not None:
        train_df = train_df_full
        val_df = pd.read_csv(val_path)
    else:
        y_full = train_df_full[label_col].astype(int)
        train_df, val_df = train_test_split(
            train_df_full,
            test_size=0.1,
            random_state=2024,
            stratify=y_full,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    test_df = pd.read_csv(test_path)

    # Ensure row_id exists for paired bootstrap alignment
    train_rowid = _pick_rowid_column(train_df)
    val_rowid = _pick_rowid_column(val_df)
    test_rowid = _pick_rowid_column(test_df)

    # Feature columns: drop label; keep row_id as an identifier (not a feature)
    drop_cols = {label_col, train_rowid}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Infer categorical vs numeric
    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    # Also treat typical ID columns as categorical if they are integer-coded
    for c in ["user_id", "problem_id", "skill_id", "item_id", "question_id"]:
        if c in feature_cols and c not in cat_cols:
            # Many datasets store these as int; we want one-hot for LogReg
            cat_cols.append(c)

    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scaler", StandardScaler(with_mean=False))]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        n_jobs=-1,
        random_state=args.seed,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train = train_df[feature_cols]
    y_train = train_df[label_col].astype(int).to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df[label_col].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df[label_col].astype(int).to_numpy()

    pipe.fit(X_train, y_train)

    val_prob = pipe.predict_proba(X_val)[:, 1]
    thr = _best_f1_threshold(y_val, val_prob)

    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= thr).astype(int)

    metrics = {
        "AUC": float(roc_auc_score(y_test, test_prob)),
        "PR_AUC": float(average_precision_score(y_test, test_prob)),
        "LogLoss": float(log_loss(y_test, np.clip(test_prob, 1e-15, 1 - 1e-15))),
        "F1": float(f1_score(y_test, test_pred)),
        "F1_threshold": float(thr),
    }

    # Save predictions for paired bootstrap (row-aligned)
    preds = pd.DataFrame(
        {
            "row_id": test_df[test_rowid].astype(int),
            "label": y_test.astype(int),
            "prediction": test_prob.astype(float),
            "seed": int(args.seed),
        }
    )
    preds_path = metric_dir / f"logreg_seed{args.seed}_test_predictions.csv"
    preds.to_csv(preds_path, index=False)

    metrics_path = metric_dir / f"logreg_seed{args.seed}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[LogReg] seed={args.seed} metrics={metrics}")
    print(f"[LogReg] saved predictions to {preds_path}")


if __name__ == "__main__":
    main()
