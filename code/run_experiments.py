"""
Run multi-seed training for DeepFM baseline and DeepFM+DKT fusion models,
aggregate metrics, compute confidence intervals and significance tests,
and export figures (loss curves, ROC/PR, confusion matrices).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import json
import subprocess
import numpy as np
import argparse

from .deepfm_experiment import (
    ExperimentConfig,
    METRIC_DIR,
    aggregate_metrics,
    bootstrap_significance,
    ensemble_predictions,
    paired_bootstrap_from_predictions,
    plot_confusion_matrix,
    plot_loss_curves,
    plot_roc_pr_curves,
    run_multi_seed,
    save_metric_tables,
    save_significance_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only rebuild combined summary/CI and significance_report.json from existing per-model artifacts in outputs/metrics. No training will be run.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=None,
        help="Override the number of bootstrap resamples used for paired/seed bootstrap significance tests (e.g., 2000 for faster runs). If not set, uses ExperimentConfig.n_bootstrap.",
    )
    args = parser.parse_args()

    def _effective_n_bootstrap(default_n: int) -> int:
        if args.n_bootstrap is None:
            return default_n
        if args.n_bootstrap <= 0:
            raise ValueError("--n_bootstrap must be a positive integer")
        return int(args.n_bootstrap)

    configs = [
        ExperimentConfig(model_name="deepfm"),
        ExperimentConfig(model_name="deepfm_dkt"),
        ExperimentConfig(model_name="deepfm_dkt", mastery_variant="zero"),
        ExperimentConfig(model_name="deepfm_dkt", mastery_variant="shuffle"),
        ExperimentConfig(model_name="deepfm_dkt", mastery_variant="leaky"),
        ExperimentConfig(model_name="deepfm_dkt", mastery_location="fm_only"),
        ExperimentConfig(model_name="deepfm_dkt", mastery_location="dnn_only"),
    ]

    # Mapping from run_tag -> display_name for combined tables
    run_tag_to_name = {c.run_tag: c.display_name for c in configs}

    results_map = {}
    preds_map = {}
    loss_figs = {}
    cm_figs = {}
    combined_summary_frames = []
    combined_ci_frames = []

    metric_dir = METRIC_DIR

    def _extract_seed_from_path(p: Path) -> int:
        # expects ..._seed{N}_...
        stem = p.stem
        if "_seed" not in stem:
            return -1
        try:
            return int(stem.split("_seed", 1)[1].split("_", 1)[0])
        except Exception:
            return -1

    def _load_ensemble_preds(prefix: str) -> pd.DataFrame:
        files = sorted(metric_dir.glob(f"{prefix}_seed*_test_predictions.csv"), key=_extract_seed_from_path)
        if not files:
            raise FileNotFoundError(f"No prediction files found for prefix='{prefix}' under {metric_dir}")
        dfs = [pd.read_csv(p) for p in files]
        all_df = pd.concat(dfs, ignore_index=True)
        # Expect columns: row_id,label,prediction,seed (seed may be missing; that is fine)
        return (
            all_df.groupby(["row_id", "label"], as_index=False)["prediction"].mean()
            .assign(seed=0)
            .sort_values("row_id")
            .reset_index(drop=True)
        )

    def _sanitize_preds(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Drop NaN/inf in label or prediction to keep plotting/bootstrap robust."""
        if df is None or df.empty:
            return df
        out = df.copy()
        # Some artifacts may use different column names; we only sanitize if present.
        cols = [c for c in ["label", "prediction"] if c in out.columns]
        if not cols:
            return out
        before = len(out)
        mask = np.ones(before, dtype=bool)
        for c in cols:
            mask &= np.isfinite(out[c].to_numpy(dtype=float, copy=False))
        out = out.loc[mask].reset_index(drop=True)
        dropped = before - len(out)
        if dropped > 0:
            print(f"[WARN] Dropped {dropped}/{before} non-finite rows from predictions: {name}")
        return out

    # If aggregate_only: do NOT train; rebuild combined tables and significance report from existing artifacts.
    if args.aggregate_only:
        # 1) Combined summary/CI from per-model summary/ci files
        for cfg in configs:
            summary_path = metric_dir / f"{cfg.run_tag}_metrics_summary.csv"
            ci_path = metric_dir / f"{cfg.run_tag}_metrics_ci.csv"
            if summary_path.exists():
                s = pd.read_csv(summary_path)
                combined_summary_frames.append(s.assign(model=cfg.display_name, run=cfg.run_tag))
            else:
                print(f"[WARN] Missing summary file: {summary_path}")
            if ci_path.exists():
                c = pd.read_csv(ci_path)
                combined_ci_frames.append(c.assign(model=cfg.display_name, run=cfg.run_tag))
            else:
                print(f"[WARN] Missing CI file: {ci_path}")

        # 2) LogReg summary/CI from per-seed metrics json
        logreg_run_tag = "logreg"
        logreg_display_name = "LogReg"

        # Infer seeds from existing logreg prediction files
        logreg_pred_files = sorted(metric_dir.glob("logreg_seed*_test_predictions.csv"), key=_extract_seed_from_path)
        logreg_seeds = [
            _extract_seed_from_path(p)
            for p in logreg_pred_files
            if _extract_seed_from_path(p) != -1
        ]
        logreg_seeds = sorted(list(dict.fromkeys(logreg_seeds)))
        if not logreg_seeds:
            raise FileNotFoundError(f"No logreg_seed*_test_predictions.csv found under {metric_dir}")

        seed_metrics = []
        for sd in logreg_seeds:
            mp = metric_dir / f"logreg_seed{sd}_metrics.json"
            if not mp.exists():
                raise FileNotFoundError(f"Missing LogReg metrics file: {mp}")
            with open(mp, "r", encoding="utf-8") as f:
                seed_metrics.append(json.load(f))

        def _summarize_metric(name: str) -> tuple[float, float]:
            vals = np.array([m[name] for m in seed_metrics], dtype=float)
            return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        rows = []
        for mname in ["AUC", "PR_AUC", "LogLoss", "F1"]:
            mean_v, std_v = _summarize_metric(mname)
            rows.append({"metric": mname, "mean": mean_v, "std": std_v, "n_seeds": len(logreg_seeds)})

        logreg_summary_df = pd.DataFrame(rows)
        logreg_ci_df = pd.DataFrame(
            [{"metric": r["metric"], "ci_low": np.nan, "ci_high": np.nan, "n_bootstrap": np.nan} for r in rows]
        )
        combined_summary_frames.append(logreg_summary_df.assign(model=logreg_display_name, run=logreg_run_tag))
        combined_ci_frames.append(logreg_ci_df.assign(model=logreg_display_name, run=logreg_run_tag))

        # 3) Write combined tables
        combined_summary = pd.concat(combined_summary_frames, ignore_index=True)
        combined_summary_path = METRIC_DIR / "combined_metrics_summary.csv"
        combined_summary.to_csv(combined_summary_path, index=False)

        combined_ci = pd.concat(combined_ci_frames, ignore_index=True)
        combined_ci_path = METRIC_DIR / "combined_metrics_ci.csv"
        combined_ci.to_csv(combined_ci_path, index=False)

        # 4) Build ensemble predictions for paired bootstrap
        preds_map["deepfm"] = _sanitize_preds(_load_ensemble_preds("deepfm"), "deepfm")
        preds_map["deepfm_dkt"] = _sanitize_preds(_load_ensemble_preds("deepfm_dkt"), "deepfm_dkt")
        preds_map["deepfm_dkt_zero"] = _sanitize_preds(_load_ensemble_preds("deepfm_dkt_zero"), "deepfm_dkt_zero")
        preds_map["deepfm_dkt_shuffle"] = _sanitize_preds(_load_ensemble_preds("deepfm_dkt_shuffle"), "deepfm_dkt_shuffle")
        preds_map["deepfm_dkt_leaky"] = _sanitize_preds(_load_ensemble_preds("deepfm_dkt_leaky"), "deepfm_dkt_leaky")
        preds_map["deepfm_dkt_fm_only"] = _sanitize_preds(_load_ensemble_preds("deepfm_dkt_fm_only"), "deepfm_dkt_fm_only")
        preds_map["deepfm_dkt_dnn_only"] = _sanitize_preds(_load_ensemble_preds("deepfm_dkt_dnn_only"), "deepfm_dkt_dnn_only")
        preds_map["logreg"] = _sanitize_preds(_load_ensemble_preds("logreg"), "logreg")

        baseline_preds = preds_map["deepfm"]
        fusion_preds = preds_map["deepfm_dkt"]

        # Reuse bootstrap settings from the deepfm_dkt config
        fusion_config = next(cfg for cfg in configs if cfg.run_tag == "deepfm_dkt")
        n_boot = _effective_n_bootstrap(fusion_config.n_bootstrap)
        if args.n_bootstrap is not None:
            print(f"[INFO] Overriding n_bootstrap: {fusion_config.n_bootstrap} -> {n_boot}")

        # ROC/PR (DeepFM vs DeepFM+DKT)
        try:
            roc_pr_fig = plot_roc_pr_curves(baseline_preds, fusion_preds)
        except ValueError as e:
            print(f"[WARN] Skipping ROC/PR plot due to: {e}")
            roc_pr_fig = None

        significance_report = {
            "paired_bootstrap/DeepFM_vs_DeepFM+DKT/AUC": paired_bootstrap_from_predictions(
                baseline_preds,
                fusion_preds,
                metric="AUC",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed,
            ),
            "paired_bootstrap/DeepFM_vs_DeepFM+DKT/LogLoss": paired_bootstrap_from_predictions(
                baseline_preds,
                fusion_preds,
                metric="LogLoss",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 1,
            ),
            "paired_bootstrap/DeepFM_vs_DeepFM+DKT/PR_AUC": paired_bootstrap_from_predictions(
                baseline_preds,
                fusion_preds,
                metric="PR_AUC",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 2,
            ),
            "paired_bootstrap/DeepFM_vs_DeepFM+DKT/F1": paired_bootstrap_from_predictions(
                baseline_preds,
                fusion_preds,
                metric="F1",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 3,
            ),
            "paired_bootstrap/LogReg_vs_DeepFM/AUC": paired_bootstrap_from_predictions(
                preds_map["logreg"],
                baseline_preds,
                metric="AUC",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 100,
            ),
            "paired_bootstrap/LogReg_vs_DeepFM/LogLoss": paired_bootstrap_from_predictions(
                preds_map["logreg"],
                baseline_preds,
                metric="LogLoss",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 101,
            ),
            "paired_bootstrap/LogReg_vs_DeepFM+DKT/AUC": paired_bootstrap_from_predictions(
                preds_map["logreg"],
                fusion_preds,
                metric="AUC",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 102,
            ),
            "paired_bootstrap/LogReg_vs_DeepFM+DKT/LogLoss": paired_bootstrap_from_predictions(
                preds_map["logreg"],
                fusion_preds,
                metric="LogLoss",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 103,
            ),
            "paired_bootstrap/DeepFM+DKT_vs_FM-only/AUC": paired_bootstrap_from_predictions(
                fusion_preds,
                preds_map["deepfm_dkt_fm_only"],
                metric="AUC",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 200,
            ),
            "paired_bootstrap/DeepFM+DKT_vs_FM-only/LogLoss": paired_bootstrap_from_predictions(
                fusion_preds,
                preds_map["deepfm_dkt_fm_only"],
                metric="LogLoss",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 201,
            ),
            "paired_bootstrap/DeepFM+DKT_vs_DNN-only/AUC": paired_bootstrap_from_predictions(
                fusion_preds,
                preds_map["deepfm_dkt_dnn_only"],
                metric="AUC",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 210,
            ),
            "paired_bootstrap/DeepFM+DKT_vs_DNN-only/LogLoss": paired_bootstrap_from_predictions(
                fusion_preds,
                preds_map["deepfm_dkt_dnn_only"],
                metric="LogLoss",
                n_bootstrap=n_boot,
                random_state=fusion_config.bootstrap_seed + 211,
            ),
        }

        # Leakage audit: leaky h_t vs non-leaky h_{t-1}
        significance_report["paired_bootstrap/Leaky_h_t_vs_NonLeaky_h_t-1/AUC"] = paired_bootstrap_from_predictions(
            preds_map["deepfm_dkt_leaky"],
            fusion_preds,
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 10,
        )
        significance_report["paired_bootstrap/Leaky_h_t_vs_NonLeaky_h_t-1/LogLoss"] = paired_bootstrap_from_predictions(
            preds_map["deepfm_dkt_leaky"],
            fusion_preds,
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 11,
        )

        # Info ablations (paired)
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Zero/AUC"] = paired_bootstrap_from_predictions(
            fusion_preds,
            preds_map["deepfm_dkt_zero"],
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 20,
        )
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Zero/LogLoss"] = paired_bootstrap_from_predictions(
            fusion_preds,
            preds_map["deepfm_dkt_zero"],
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 21,
        )
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Shuffled/AUC"] = paired_bootstrap_from_predictions(
            fusion_preds,
            preds_map["deepfm_dkt_shuffle"],
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 30,
        )
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Shuffled/LogLoss"] = paired_bootstrap_from_predictions(
            fusion_preds,
            preds_map["deepfm_dkt_shuffle"],
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 31,
        )

        sig_path = save_significance_report(significance_report)

        print("Artifacts generated (aggregate_only):")
        print(f"- ROC/PR figure (DeepFM vs DeepFM+DKT): {roc_pr_fig}")
        print(f"- Combined metric summary: {combined_summary_path}")
        print(f"- Combined metric CI: {combined_ci_path}")
        print(f"- Significance report: {sig_path}")
        return

    # --- Full pipeline: train + aggregate + significance ---
    for config in configs:
        print(f"=== Running configuration: {config.display_name} ({config.run_tag}) ===")
        try:
            results = run_multi_seed(config)
        except Exception as exc:
            print(f"[ERROR] Training failed for {config.display_name}: {exc}")
            continue

        results_map[config.run_tag] = results

        summary_df, ci_df = aggregate_metrics(results)
        save_metric_tables(config, summary_df, ci_df)

        combined_summary_frames.append(summary_df.assign(model=config.display_name, run=config.run_tag))
        combined_ci_frames.append(ci_df.assign(model=config.display_name, run=config.run_tag))

        try:
            loss_figs[config.run_tag] = plot_loss_curves(config.display_name, results)
            preds_df = _sanitize_preds(ensemble_predictions(results), config.run_tag)
            preds_map[config.run_tag] = preds_df
            cm_figs[config.run_tag] = plot_confusion_matrix(preds_df, config.display_name)
        except Exception as exc:
            print(f"[WARN] Plotting failed for {config.display_name}: {exc}")

    # --- Simple tabular baseline: Logistic Regression ---
    logreg_run_tag = "logreg"
    logreg_display_name = "LogReg"

    # Reuse seeds from the fusion config (same seeds -> easier paired bootstrap)
    # Note: fusion_config is defined later, so we derive seeds from any deepfm_dkt config present.
    dkt_cfg = next((c for c in configs if c.run_tag == "deepfm_dkt"), None)
    logreg_seeds = list(dkt_cfg.seeds) if dkt_cfg is not None else [42]

    logreg_seed_preds = []
    for sd in logreg_seeds:
        pred_path = metric_dir / f"logreg_seed{sd}_test_predictions.csv"
        metrics_path = metric_dir / f"logreg_seed{sd}_metrics.json"

        if not pred_path.exists() or not metrics_path.exists():
            # Run the baseline if artifacts are missing
            cmd = ["python", "-m", "src.train_logreg", "--seed", str(sd)]
            print(f"=== Running LogReg baseline: seed={sd} ===")
            subprocess.run(cmd, check=True)

        dfp = pd.read_csv(pred_path)
        # Expect columns: row_id,label,prediction,seed
        logreg_seed_preds.append(dfp)

    # Ensemble across seeds by averaging probabilities per row_id
    logreg_all = pd.concat(logreg_seed_preds, ignore_index=True)
    logreg_preds = (
        logreg_all.groupby(["row_id", "label"], as_index=False)["prediction"].mean()
        .assign(seed=0)
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    preds_map[logreg_run_tag] = _sanitize_preds(logreg_preds, "logreg")

    # Also build a seed-level results-like summary for combined tables
    # (mean over seeds; std over seeds; CI left blank here)
    seed_metrics = []
    for sd in logreg_seeds:
        mp = metric_dir / f"logreg_seed{sd}_metrics.json"
        with open(mp, "r", encoding="utf-8") as f:
            seed_metrics.append(json.load(f))

    def _summarize_metric(name: str) -> tuple[float, float]:
        vals = np.array([m[name] for m in seed_metrics], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    rows = []
    for mname in ["AUC", "PR_AUC", "LogLoss", "F1"]:
        mean_v, std_v = _summarize_metric(mname)
        rows.append({"metric": mname, "mean": mean_v, "std": std_v, "n_seeds": len(logreg_seeds)})

    logreg_summary_df = pd.DataFrame(rows)
    # CI table placeholder (kept consistent with combined export)
    logreg_ci_df = pd.DataFrame(
        [{"metric": r["metric"], "ci_low": np.nan, "ci_high": np.nan, "n_bootstrap": np.nan} for r in rows]
    )

    combined_summary_frames.append(logreg_summary_df.assign(model=logreg_display_name, run=logreg_run_tag))
    combined_ci_frames.append(logreg_ci_df.assign(model=logreg_display_name, run=logreg_run_tag))

    baseline_config = next(cfg for cfg in configs if cfg.run_tag == "deepfm")
    fusion_config = next(cfg for cfg in configs if cfg.run_tag == "deepfm_dkt")
    n_boot = _effective_n_bootstrap(fusion_config.n_bootstrap)
    if args.n_bootstrap is not None:
        print(f"[INFO] Overriding n_bootstrap: {fusion_config.n_bootstrap} -> {n_boot}")

    if "deepfm" not in results_map or "deepfm_dkt" not in results_map:
        print("[WARN] Missing baseline or fusion results; skipping ROC/PR and significance analysis.")
        print(f"Available models: {list(results_map.keys())}")
        return

    baseline_results = results_map["deepfm"]
    fusion_results = results_map["deepfm_dkt"]
    zero_results = results_map.get("deepfm_dkt_zero")
    shuffle_results = results_map.get("deepfm_dkt_shuffle")
    leaky_results = results_map.get("deepfm_dkt_leaky")

    baseline_preds = preds_map.get("deepfm")
    if baseline_preds is None:
        baseline_preds = ensemble_predictions(baseline_results)
        preds_map["deepfm"] = baseline_preds

    fusion_preds = preds_map.get("deepfm_dkt")
    if fusion_preds is None:
        fusion_preds = ensemble_predictions(fusion_results)
        preds_map["deepfm_dkt"] = fusion_preds

    leaky_preds = preds_map.get("deepfm_dkt_leaky")
    if leaky_results is not None and leaky_preds is None:
        leaky_preds = ensemble_predictions(leaky_results)
        preds_map["deepfm_dkt_leaky"] = leaky_preds

    try:
        roc_pr_fig = plot_roc_pr_curves(baseline_preds, fusion_preds)
    except ValueError as e:
        print(f"[WARN] Skipping ROC/PR plot due to: {e}")
        roc_pr_fig = None

    combined_summary = pd.concat(combined_summary_frames, ignore_index=True)
    combined_summary_path = METRIC_DIR / "combined_metrics_summary.csv"
    combined_summary.to_csv(combined_summary_path, index=False)

    combined_ci = pd.concat(combined_ci_frames, ignore_index=True)
    combined_ci_path = METRIC_DIR / "combined_metrics_ci.csv"
    combined_ci.to_csv(combined_ci_path, index=False)

    significance_report = {
        # Preferred: paired bootstrap over per-sample predictions (ensemble mean)
        "paired_bootstrap/DeepFM_vs_DeepFM+DKT/AUC": paired_bootstrap_from_predictions(
            baseline_preds,
            fusion_preds,
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed,
        ),
        "paired_bootstrap/DeepFM_vs_DeepFM+DKT/LogLoss": paired_bootstrap_from_predictions(
            baseline_preds,
            fusion_preds,
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 1,
        ),
        "paired_bootstrap/DeepFM_vs_DeepFM+DKT/PR_AUC": paired_bootstrap_from_predictions(
            baseline_preds,
            fusion_preds,
            metric="PR_AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 2,
        ),
        "paired_bootstrap/DeepFM_vs_DeepFM+DKT/F1": paired_bootstrap_from_predictions(
            baseline_preds,
            fusion_preds,
            metric="F1",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 3,
        ),
        # Kept for reference: seed-level bootstrap over per-seed metrics
        "seed_bootstrap/DeepFM_vs_DeepFM+DKT/AUC": bootstrap_significance(
            baseline_values=[res.metrics["AUC"] for res in baseline_results],
            variant_values=[res.metrics["AUC"] for res in fusion_results],
            greater_is_better=True,
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed,
        ),
        "seed_bootstrap/DeepFM_vs_DeepFM+DKT/LogLoss": bootstrap_significance(
            baseline_values=[res.metrics["LogLoss"] for res in baseline_results],
            variant_values=[res.metrics["LogLoss"] for res in fusion_results],
            greater_is_better=False,
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 1,
        ),
        "paired_bootstrap/LogReg_vs_DeepFM/AUC": paired_bootstrap_from_predictions(
            preds_map["logreg"],
            baseline_preds,
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 100,
        ),
        "paired_bootstrap/LogReg_vs_DeepFM/LogLoss": paired_bootstrap_from_predictions(
            preds_map["logreg"],
            baseline_preds,
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 101,
        ),
        "paired_bootstrap/LogReg_vs_DeepFM+DKT/AUC": paired_bootstrap_from_predictions(
            preds_map["logreg"],
            fusion_preds,
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 102,
        ),
        "paired_bootstrap/LogReg_vs_DeepFM+DKT/LogLoss": paired_bootstrap_from_predictions(
            preds_map["logreg"],
            fusion_preds,
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 103,
        ),
    }

    # Leakage audit: leaky h_t vs non-leaky h_{t-1}
    if leaky_preds is not None:
        significance_report["paired_bootstrap/Leaky_h_t_vs_NonLeaky_h_t-1/AUC"] = paired_bootstrap_from_predictions(
            leaky_preds,
            fusion_preds,
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 10,
        )
        significance_report["paired_bootstrap/Leaky_h_t_vs_NonLeaky_h_t-1/LogLoss"] = paired_bootstrap_from_predictions(
            leaky_preds,
            fusion_preds,
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 11,
        )

    if zero_results is not None:
        significance_report["DeepFM+DKT_vs_Zero_AUC"] = bootstrap_significance(
            baseline_values=[res.metrics["AUC"] for res in fusion_results],
            variant_values=[res.metrics["AUC"] for res in zero_results],
            greater_is_better=True,
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 2,
        )
        significance_report["DeepFM+DKT_vs_Zero_LogLoss"] = bootstrap_significance(
            baseline_values=[res.metrics["LogLoss"] for res in fusion_results],
            variant_values=[res.metrics["LogLoss"] for res in zero_results],
            greater_is_better=False,
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 3,
        )
        zero_preds = preds_map.get("deepfm_dkt_zero")
        if zero_preds is None:
            zero_preds = ensemble_predictions(zero_results)
            preds_map["deepfm_dkt_zero"] = zero_preds
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Zero/AUC"] = paired_bootstrap_from_predictions(
            fusion_preds,
            zero_preds,
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 20,
        )
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Zero/LogLoss"] = paired_bootstrap_from_predictions(
            fusion_preds,
            zero_preds,
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 21,
        )

    if shuffle_results is not None:
        significance_report["DeepFM+DKT_vs_Shuffled_AUC"] = bootstrap_significance(
            baseline_values=[res.metrics["AUC"] for res in fusion_results],
            variant_values=[res.metrics["AUC"] for res in shuffle_results],
            greater_is_better=True,
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 4,
        )
        significance_report["DeepFM+DKT_vs_Shuffled_LogLoss"] = bootstrap_significance(
            baseline_values=[res.metrics["LogLoss"] for res in fusion_results],
            variant_values=[res.metrics["LogLoss"] for res in shuffle_results],
            greater_is_better=False,
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 5,
        )
        shuffle_preds = preds_map.get("deepfm_dkt_shuffle")
        if shuffle_preds is None:
            shuffle_preds = ensemble_predictions(shuffle_results)
            preds_map["deepfm_dkt_shuffle"] = shuffle_preds
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Shuffled/AUC"] = paired_bootstrap_from_predictions(
            fusion_preds,
            shuffle_preds,
            metric="AUC",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 30,
        )
        significance_report["paired_bootstrap/DeepFM+DKT_vs_Shuffled/LogLoss"] = paired_bootstrap_from_predictions(
            fusion_preds,
            shuffle_preds,
            metric="LogLoss",
            n_bootstrap=n_boot,
            random_state=fusion_config.bootstrap_seed + 31,
        )

    sig_path = save_significance_report(significance_report)

    print("Artifacts generated:")
    for config in configs:
        loss_path = loss_figs.get(config.run_tag)
        cm_path = cm_figs.get(config.run_tag)
        print(
            f"- {config.display_name}: "
            f"loss curve={loss_path if loss_path else 'N/A'}, "
            f"confusion matrix={cm_path if cm_path else 'N/A'}"
        )
    print(f"- {logreg_display_name}: metrics/preds in {METRIC_DIR}")
    print(f"- ROC/PR figure (DeepFM vs DeepFM+DKT): {roc_pr_fig}")
    print(f"- Combined metric summary: {combined_summary_path}")
    print(f"- Combined metric CI: {combined_ci_path}")
    print(f"- Significance report: {sig_path}")


if __name__ == "__main__":
    main()
