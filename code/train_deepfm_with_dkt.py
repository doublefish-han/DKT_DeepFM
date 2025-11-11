"""
Convenience entry point to train the DeepFM+DKT fusion model for a single seed.
For multi-seed experiments and aggregated reporting, use run_experiments.py.
"""
from __future__ import annotations

from deepfm_experiment import ExperimentConfig, aggregate_metrics, save_metric_tables, train_single_seed


def run(seed: int = 42) -> None:
    config = ExperimentConfig(model_name="deepfm_dkt", seeds=(seed,))
    result = train_single_seed(config, seed)
    summary_df, ci_df = aggregate_metrics([result])
    save_metric_tables(config, summary_df, ci_df)


if __name__ == "__main__":
    run()
