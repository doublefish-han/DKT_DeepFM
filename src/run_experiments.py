"""
Run multi-seed training for DeepFM baseline and DeepFM+DKT fusion models,
aggregate metrics, compute confidence intervals and significance tests,
and export figures (loss curves, ROC/PR, confusion matrices).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from deepfm_experiment import (
    ExperimentConfig,
    METRIC_DIR,
    aggregate_metrics,
    bootstrap_significance,
    ensemble_predictions,
    plot_confusion_matrix,
    plot_loss_curves,
    plot_roc_pr_curves,
    run_multi_seed,
    save_metric_tables,
    save_significance_report,
)


def main() -> None:
    configs = [
        ExperimentConfig(model_name="deepfm"),
        ExperimentConfig(model_name="deepfm_dkt"),
        ExperimentConfig(model_name="deepfm_dkt", mastery_variant="zero"),
        ExperimentConfig(model_name="deepfm_dkt", mastery_variant="shuffle"),
    ]

    results_map = {}
    preds_map = {}
    loss_figs = {}
    cm_figs = {}
    combined_summary_frames = []
    combined_ci_frames = []

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
            preds_df = ensemble_predictions(results)
            preds_map[config.run_tag] = preds_df
            cm_figs[config.run_tag] = plot_confusion_matrix(preds_df, config.display_name)
        except Exception as exc:
            print(f"[WARN] Plotting failed for {config.display_name}: {exc}")

    baseline_config = next(cfg for cfg in configs if cfg.run_tag == "deepfm")
    fusion_config = next(cfg for cfg in configs if cfg.run_tag == "deepfm_dkt")

    if "deepfm" not in results_map or "deepfm_dkt" not in results_map:
        print("[WARN] Missing baseline or fusion results; skipping ROC/PR and significance analysis.")
        print(f"Available models: {list(results_map.keys())}")
        return

    baseline_results = results_map["deepfm"]
    fusion_results = results_map["deepfm_dkt"]
    zero_results = results_map.get("deepfm_dkt_zero")
    shuffle_results = results_map.get("deepfm_dkt_shuffle")

    baseline_preds = preds_map.get("deepfm")
    if baseline_preds is None:
        baseline_preds = ensemble_predictions(baseline_results)
        preds_map["deepfm"] = baseline_preds

    fusion_preds = preds_map.get("deepfm_dkt")
    if fusion_preds is None:
        fusion_preds = ensemble_predictions(fusion_results)
        preds_map["deepfm_dkt"] = fusion_preds
    roc_pr_fig = plot_roc_pr_curves(baseline_preds, fusion_preds)

    combined_summary = pd.concat(combined_summary_frames, ignore_index=True)
    combined_summary_path = METRIC_DIR / "combined_metrics_summary.csv"
    combined_summary.to_csv(combined_summary_path, index=False)

    combined_ci = pd.concat(combined_ci_frames, ignore_index=True)
    combined_ci_path = METRIC_DIR / "combined_metrics_ci.csv"
    combined_ci.to_csv(combined_ci_path, index=False)

    significance_report = {
        "DeepFM_vs_DeepFM+DKT": bootstrap_significance(
            baseline_values=[res.metrics["AUC"] for res in baseline_results],
            variant_values=[res.metrics["AUC"] for res in fusion_results],
            greater_is_better=True,
            n_bootstrap=fusion_config.n_bootstrap,
            random_state=fusion_config.bootstrap_seed,
        ),
        "DeepFM_vs_DeepFM+DKT_LogLoss": bootstrap_significance(
            baseline_values=[res.metrics["LogLoss"] for res in baseline_results],
            variant_values=[res.metrics["LogLoss"] for res in fusion_results],
            greater_is_better=False,
            n_bootstrap=fusion_config.n_bootstrap,
            random_state=fusion_config.bootstrap_seed + 1,
        ),
    }

    if zero_results is not None:
        significance_report["DeepFM+DKT_vs_Zero_AUC"] = bootstrap_significance(
            baseline_values=[res.metrics["AUC"] for res in fusion_results],
            variant_values=[res.metrics["AUC"] for res in zero_results],
            greater_is_better=True,
            n_bootstrap=fusion_config.n_bootstrap,
            random_state=fusion_config.bootstrap_seed + 2,
        )
        significance_report["DeepFM+DKT_vs_Zero_LogLoss"] = bootstrap_significance(
            baseline_values=[res.metrics["LogLoss"] for res in fusion_results],
            variant_values=[res.metrics["LogLoss"] for res in zero_results],
            greater_is_better=False,
            n_bootstrap=fusion_config.n_bootstrap,
            random_state=fusion_config.bootstrap_seed + 3,
        )

    if shuffle_results is not None:
        significance_report["DeepFM+DKT_vs_Shuffled_AUC"] = bootstrap_significance(
            baseline_values=[res.metrics["AUC"] for res in fusion_results],
            variant_values=[res.metrics["AUC"] for res in shuffle_results],
            greater_is_better=True,
            n_bootstrap=fusion_config.n_bootstrap,
            random_state=fusion_config.bootstrap_seed + 4,
        )
        significance_report["DeepFM+DKT_vs_Shuffled_LogLoss"] = bootstrap_significance(
            baseline_values=[res.metrics["LogLoss"] for res in fusion_results],
            variant_values=[res.metrics["LogLoss"] for res in shuffle_results],
            greater_is_better=False,
            n_bootstrap=fusion_config.n_bootstrap,
            random_state=fusion_config.bootstrap_seed + 5,
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
    print(f"- ROC/PR figure (DeepFM vs DeepFM+DKT): {roc_pr_fig}")
    print(f"- Combined metric summary: {combined_summary_path}")
    print(f"- Combined metric CI: {combined_ci_path}")
    print(f"- Significance report: {sig_path}")


if __name__ == "__main__":
    main()
