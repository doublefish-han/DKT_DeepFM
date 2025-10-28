"""
Shared utilities for training DeepFM variants with or without DKT mastery features.
Provides data preparation, training with early stopping, multi-seed orchestration,
metric aggregation, statistical testing, and figure generation helpers.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "train_test_split"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

for _path in (OUTPUT_DIR, FIG_DIR, METRIC_DIR, CHECKPOINT_DIR, LOG_DIR):
    _path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    model_name: Literal["deepfm", "deepfm_dkt"]
    mastery_variant: Literal["auto", "none", "full", "zero", "shuffle"] = "auto"
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 2048
    val_ratio: float = 0.1
    val_random_state: int = 2024
    seeds: Sequence[int] = (42, 52, 62, 72, 82)
    device: Optional[str] = None
    n_bootstrap: int = 10000
    bootstrap_seed: int = 2025

    def __post_init__(self) -> None:
        if self.mastery_variant == "auto":
            self.mastery_variant = "none" if self.model_name == "deepfm" else "full"

    @property
    def uses_mastery(self) -> bool:
        return self.mastery_variant in {"full", "zero", "shuffle"}

    @property
    def run_tag(self) -> str:
        if self.mastery_variant in {"none", "full"}:
            return self.model_name
        return f"{self.model_name}_{self.mastery_variant}"

    @property
    def display_name(self) -> str:
        mapping = {
            ("deepfm", "none"): "DeepFM",
            ("deepfm_dkt", "full"): "DeepFM+DKT",
            ("deepfm_dkt", "zero"): "Zero-Mastery",
            ("deepfm_dkt", "shuffle"): "Shuffled-Mastery",
        }
        return mapping.get((self.model_name, self.mastery_variant), self.run_tag)

    @property
    def max_epochs_label(self) -> str:
        return f"{self.max_epochs}_epochs"


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    sparse_cols: List[str]
    dense_cols: List[str]
    target: str = "correct"


@dataclass
class SeedResult:
    seed: int
    metrics: Dict[str, float]
    history: pd.DataFrame
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    checkpoint_path: Path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(config: ExperimentConfig, seed: int) -> DatasetBundle:
    if config.uses_mastery:
        train_path = DATA_DIR / "train_with_mastery.csv"
        test_path = DATA_DIR / "test_with_mastery.csv"
    else:
        train_path = DATA_DIR / "train.csv"
        test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected data splits not found for {'fusion' if config.uses_mastery else 'baseline'} model.\n"
            f"Missing: {train_path} or {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.reset_index(drop=True).assign(row_id=lambda df: np.arange(len(df)))
    test_df = test_df.reset_index(drop=True).assign(row_id=lambda df: np.arange(len(df)))

    sparse_cols = [c for c in ["user_id", "problem_id", "skill_id"] if c in train_df.columns]
    dense_candidates = ["weekday", "hour", "opportunity", "duration"]
    dense_cols = [c for c in dense_candidates if c in train_df.columns]

    mastery_cols: List[str] = []
    if config.uses_mastery:
        mastery_cols = [c for c in train_df.columns if c.startswith("mastery_")]
        dense_cols.extend(mastery_cols)
        if config.mastery_variant == "zero":
            for df in (train_df, test_df):
                if mastery_cols:
                    df.loc[:, mastery_cols] = 0.0
        elif config.mastery_variant == "shuffle":
            if mastery_cols:
                rng_train = np.random.default_rng(seed)
                rng_test = np.random.default_rng(seed + 13)
                for df, rng in ((train_df, rng_train), (test_df, rng_test)):
                    if len(df):
                        perm = rng.permutation(len(df))
                        df.loc[:, mastery_cols] = df.iloc[perm][mastery_cols].to_numpy()

    if not dense_cols:
        raise ValueError("No dense feature columns detected; verify preprocessing output.")

    target = "correct"
    if target not in train_df.columns:
        raise KeyError(f"Target column '{target}' missing from training data.")

    stratify_col = train_df[target] if train_df[target].nunique() > 1 else None
    train_core, val_core = train_test_split(
        train_df,
        test_size=config.val_ratio,
        random_state=config.val_random_state,
        stratify=stratify_col,
    )

    return DatasetBundle(
        train_df=train_core.reset_index(drop=True),
        val_df=val_core.reset_index(drop=True),
        test_df=test_df,
        sparse_cols=sparse_cols,
        dense_cols=dense_cols,
        target=target,
    )


def build_feature_columns(bundle: DatasetBundle) -> Tuple[List[SparseFeat], List[DenseFeat], List[str]]:
    merged = pd.concat([bundle.train_df, bundle.val_df, bundle.test_df], axis=0, ignore_index=True)
    linear_cols: List[SparseFeat | DenseFeat] = []
    dnn_cols: List[SparseFeat | DenseFeat] = []

    for col in bundle.sparse_cols:
        vocab_size = int(merged[col].max()) + 1
        linear_cols.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=8))
        dnn_cols.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=8))

    for col in bundle.dense_cols:
        linear_cols.append(DenseFeat(col, 1))
        dnn_cols.append(DenseFeat(col, 1))

    feature_names = get_feature_names(linear_cols + dnn_cols)
    return linear_cols, dnn_cols, feature_names


def to_model_input(df: pd.DataFrame, feature_names: Iterable[str]) -> Dict[str, np.ndarray]:
    return {name: df[name].values for name in feature_names}


def clip_probs(y_pred: ArrayLike, eps: float = 1e-7) -> np.ndarray:
    arr = np.asarray(y_pred, dtype=np.float64)
    return np.clip(arr, eps, 1.0 - eps)


def parse_history(history_obj) -> pd.DataFrame:
    if history_obj is None:
        return pd.DataFrame()
    history_dict = None
    if hasattr(history_obj, "history"):
        history_dict = history_obj.history
    elif isinstance(history_obj, dict):
        history_dict = history_obj
    if not history_dict:
        return pd.DataFrame()
    df = pd.DataFrame(history_dict)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    return df


def train_single_seed(config: ExperimentConfig, seed: int) -> SeedResult:
    set_global_seed(seed)
    bundle = load_dataset(config, seed)
    linear_cols, dnn_cols, feature_names = build_feature_columns(bundle)

    train_input = to_model_input(bundle.train_df, feature_names)
    val_input = to_model_input(bundle.val_df, feature_names)
    test_input = to_model_input(bundle.test_df, feature_names)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task="binary",
        dnn_hidden_units=(256, 128, 64),
        dnn_dropout=0.2,
        l2_reg_embedding=1e-5,
        device=device,
        seed=seed,
    )
    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    ckpt_path = CHECKPOINT_DIR / f"{config.run_tag}_seed{seed}.pt"
    mc_cb = ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=True,
    )

    history = model.fit(
        x=train_input,
        y=bundle.train_df[bundle.target].values,
        batch_size=config.batch_size,
        epochs=config.max_epochs,
        verbose=2,
        shuffle=True,
        validation_data=(val_input, bundle.val_df[bundle.target].values),
        callbacks=[mc_cb],
    )

    if ckpt_path.exists():
        try:
            checkpoint_obj = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            checkpoint_obj = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(checkpoint_obj, DeepFM):
            model = checkpoint_obj.to(device)
        elif isinstance(checkpoint_obj, dict):
            state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
            model.load_state_dict(state_dict)

    val_pred = clip_probs(model.predict(val_input).squeeze())
    test_pred = clip_probs(model.predict(test_input).squeeze())

    val_true = bundle.val_df[bundle.target].values
    test_true = bundle.test_df[bundle.target].values

    metrics = {
        "seed": seed,
        "AUC": float(roc_auc_score(test_true, test_pred)),
        "PR_AUC": float(average_precision_score(test_true, test_pred)),
        "LogLoss": float(log_loss(test_true, test_pred)),
        "F1": float(f1_score(test_true, (test_pred >= 0.5).astype(int))),
    }

    history_df = parse_history(history)
    history_df["seed"] = seed
    history_path = LOG_DIR / f"{config.run_tag}_seed{seed}_history.csv"
    history_df.to_csv(history_path, index=False)

    val_pred_df = pd.DataFrame(
        {
            "row_id": bundle.val_df["row_id"].values,
            "label": val_true,
            "prediction": val_pred,
            "seed": seed,
        }
    )
    val_pred_path = METRIC_DIR / f"{config.run_tag}_seed{seed}_val_predictions.csv"
    val_pred_df.to_csv(val_pred_path, index=False)

    test_pred_df = pd.DataFrame(
        {
            "row_id": bundle.test_df["row_id"].values,
            "label": test_true,
            "prediction": test_pred,
            "seed": seed,
        }
    )
    test_pred_path = METRIC_DIR / f"{config.run_tag}_seed{seed}_test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)

    metrics_path = METRIC_DIR / f"{config.run_tag}_seed{seed}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return SeedResult(
        seed=seed,
        metrics=metrics,
        history=history_df,
        val_predictions=val_pred_df,
        test_predictions=test_pred_df,
        checkpoint_path=ckpt_path,
    )


def run_multi_seed(config: ExperimentConfig) -> List[SeedResult]:
    results: List[SeedResult] = []
    for seed in config.seeds:
        result = train_single_seed(config, seed)
        results.append(result)
    return results


def aggregate_metrics(results: Sequence[SeedResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.DataFrame([res.metrics for res in results])
    summary_rows = []
    ci_rows = []
    for metric in ["AUC", "PR_AUC", "LogLoss", "F1"]:
        values = metrics_df[metric].values
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        ci = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "mean_std": f"{mean:.4f} ± {std:.4f}",
            }
        )
        ci_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "ci_low": mean - ci,
                "ci_high": mean + ci,
                "half_width": ci,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    ci_df = pd.DataFrame(ci_rows)
    return summary_df, ci_df


def bootstrap_significance(
    baseline_values: Sequence[float],
    variant_values: Sequence[float],
    *,
    greater_is_better: bool,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, float]:
    if len(baseline_values) != len(variant_values):
        raise ValueError("Baseline and variant metric lists must have equal length for paired bootstrap.")
    baseline_arr = np.asarray(baseline_values, dtype=np.float64)
    variant_arr = np.asarray(variant_values, dtype=np.float64)
    if greater_is_better:
        diffs = variant_arr - baseline_arr
    else:
        diffs = baseline_arr - variant_arr

    rng = np.random.default_rng(random_state)
    boot_samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sampled = rng.choice(diffs, size=diffs.shape[0], replace=True)
        boot_samples[i] = sampled.mean()

    ci_low, ci_high = np.percentile(boot_samples, [2.5, 97.5])
    observed = diffs.mean()
    if observed >= 0:
        p_value = float(np.mean(boot_samples <= 0))
    else:
        p_value = float(np.mean(boot_samples >= 0))

    return {
        "mean_diff": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }


def ensemble_predictions(results: Sequence[SeedResult]) -> pd.DataFrame:
    merged = None
    for res in results:
        df = res.test_predictions[["row_id", "label", "prediction"]].copy()
        df = df.rename(columns={"prediction": f"pred_seed_{res.seed}"})
        merged = df if merged is None else merged.merge(df, on=["row_id", "label"], how="inner")
    return merged


def compute_ensemble_mean(merged_predictions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_cols = [c for c in merged_predictions.columns if c.startswith("pred_seed_")]
    probs = merged_predictions[pred_cols].mean(axis=1).values
    labels = merged_predictions["label"].values
    row_ids = merged_predictions["row_id"].values
    return row_ids, labels, probs


def plot_loss_curves(
    model_name: str,
    results: Sequence[SeedResult],
    save_path: Optional[Path] = None,
) -> Path:
    if save_path is None:
        sanitized = model_name.lower().replace("+", "_plus_").replace(" ", "_")
        save_path = FIG_DIR / f"{sanitized}_loss_curve.png"

    candidate: Optional[Tuple[str, str, str]] = None
    for res in results:
        columns = set(res.history.columns)
        if {"epoch", "loss", "val_loss"} <= columns:
            candidate = ("loss", "val_loss", "Loss")
            break
    if candidate is None:
        for res in results:
            columns = set(res.history.columns)
            if {"epoch", "auc", "val_auc"} <= columns:
                candidate = ("auc", "val_auc", "AUC")
                break

    if candidate is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return save_path

    train_col, val_col, ylabel = candidate
    histories: List[pd.DataFrame] = []
    for res in results:
        if {"epoch", train_col, val_col} <= set(res.history.columns):
            df = res.history[["epoch", train_col, val_col]].copy()
            df = df.rename(columns={train_col: "train_metric", val_col: "val_metric"})
            histories.append(df)

    if not histories:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        return save_path

    merged = pd.concat(histories, axis=0, ignore_index=True)
    grouped = merged.groupby("epoch", as_index=False).agg({"train_metric": "mean", "val_metric": "mean"})

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        grouped["epoch"],
        grouped["train_metric"],
        label=f"Train {ylabel}",
        color="tab:blue",
        linewidth=2,
    )
    ax.plot(
        grouped["epoch"],
        grouped["val_metric"],
        label=f"Validation {ylabel}",
        color="tab:orange",
        linewidth=2,
        linestyle="--",
    )

    for res_df in histories:
        ax.plot(res_df["epoch"], res_df["train_metric"], color="tab:blue", alpha=0.15, linewidth=1)
        ax.plot(res_df["epoch"], res_df["val_metric"], color="tab:orange", alpha=0.15, linewidth=1, linestyle="--")

    ax.set_title(f"{model_name} train vs validation {ylabel}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_roc_pr_curves(
    baseline_preds: pd.DataFrame,
    fusion_preds: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    if save_path is None:
        save_path = FIG_DIR / "roc_pr_curves.png"

    def _get_mean_preds(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        _, labels, probs = compute_ensemble_mean(df)
        return labels, probs

    labels_b, probs_b = _get_mean_preds(baseline_preds)
    labels_f, probs_f = _get_mean_preds(fusion_preds)

    if not np.array_equal(labels_b, labels_f):
        raise ValueError("Baseline and fusion predictions have misaligned labels.")

    fpr_b, tpr_b, _ = roc_curve(labels_b, probs_b)
    fpr_f, tpr_f, _ = roc_curve(labels_f, probs_f)
    precision_b, recall_b, _ = precision_recall_curve(labels_b, probs_b)
    precision_f, recall_f, _ = precision_recall_curve(labels_f, probs_f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(fpr_b, tpr_b, label=f"DeepFM (AUC={roc_auc_score(labels_b, probs_b):.3f})")
    axes[0].plot(fpr_f, tpr_f, label=f"DeepFM+DKT (AUC={roc_auc_score(labels_f, probs_f):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right")

    axes[1].plot(recall_b, precision_b, label=f"DeepFM (PR-AUC={average_precision_score(labels_b, probs_b):.3f})")
    axes[1].plot(
        recall_f,
        precision_f,
        label=f"DeepFM+DKT (PR-AUC={average_precision_score(labels_f, probs_f):.3f})",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    merged_predictions: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
) -> Path:
    if save_path is None:
        sanitized = title.lower().replace("+", "_plus_").replace(" ", "_")
        save_path = FIG_DIR / f"{sanitized}_confusion_matrix.png"

    _, labels, probs = compute_ensemble_mean(merged_predictions)
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def save_metric_tables(
    config: ExperimentConfig,
    summary_df: pd.DataFrame,
    ci_df: pd.DataFrame,
) -> None:
    csv_path = METRIC_DIR / f"{config.run_tag}_metrics_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    md_lines = [f"### {config.display_name}", "| Metric | Mean ± Std |", "| --- | --- |"]
    for _, row in summary_df.iterrows():
        md_lines.append(f"| {row['metric']} | {row['mean_std']} |")
    md_path = METRIC_DIR / f"{config.run_tag}_metrics_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    ci_path = METRIC_DIR / f"{config.run_tag}_metrics_ci.csv"
    ci_df.to_csv(ci_path, index=False)


def save_significance_report(report: Dict[str, Dict[str, float]]) -> Path:
    path = METRIC_DIR / "significance_report.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    return path
