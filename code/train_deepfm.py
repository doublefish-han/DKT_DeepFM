"""
Single-seed training entry point for the DeepFM baseline.
For multi-seed experiments, CI calculation, and plotting refer to run_experiments.py.
"""
from __future__ import annotations

from deepfm_experiment import ExperimentConfig, aggregate_metrics, save_metric_tables, train_single_seed


def run(seed: int = 42) -> None:
    config = ExperimentConfig(model_name="deepfm", seeds=(seed,))
    result = train_single_seed(config, seed)
    summary_df, ci_df = aggregate_metrics([result])
    save_metric_tables(config, summary_df, ci_df)


if __name__ == "__main__":
    run()
