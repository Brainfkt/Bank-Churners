from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load import build_audit_tables, load_raw_dataset, prepare_base_dataset
from src.modeling.explain import export_model_diagnostics
from src.modeling.train import compare_unknown_strategies, train_full_benchmark
from src.segmentation.clustering import run_segmentation
from src.utils.config import PATHS
from src.utils.io import save_frame, save_json
from src.visualization.plots import plot_eda_suite, plot_model_curves


def run_project_pipeline() -> dict[str, object]:
    _ensure_output_folders()

    raw_df = load_raw_dataset()
    base_df = prepare_base_dataset(raw_df)

    audit_summary, missing_table, numeric_profile, categorical_profile = build_audit_tables(base_df)
    save_json(audit_summary.__dict__, PATHS.output_metrics / "audit_summary.json")
    save_frame(missing_table, PATHS.output_metrics / "missingness_profile.csv")
    save_frame(numeric_profile, PATHS.output_metrics / "numeric_profile.csv")
    save_frame(categorical_profile, PATHS.output_metrics / "categorical_profile.csv")
    save_frame(base_df, PATHS.data_processed / "bank_churners_base.csv")

    plot_eda_suite(base_df)

    unknown_strategy_df = compare_unknown_strategies(base_df)
    selected_strategy = str(unknown_strategy_df["selected_strategy"].iloc[0])

    benchmark_results = train_full_benchmark(base_df, unknown_strategy=selected_strategy)
    validation_predictions = (benchmark_results["validation_probabilities"] >= benchmark_results["threshold"]).astype(int)
    test_predictions = (benchmark_results["test_probabilities"] >= benchmark_results["threshold"]).astype(int)
    plot_model_curves(
        benchmark_results["bundle"].y_val,
        benchmark_results["validation_probabilities"],
        validation_predictions,
        "Validation",
    )
    plot_model_curves(
        benchmark_results["bundle"].y_test,
        benchmark_results["test_probabilities"],
        test_predictions,
        "Test",
    )

    diagnostics = export_model_diagnostics(
        benchmark_results["best_model_validation"],
        benchmark_results["bundle"],
        benchmark_results["validation_probabilities"],
        benchmark_results["test_probabilities"],
        benchmark_results["threshold"],
    )

    segmentation_results = run_segmentation(base_df, benchmark_results["scores"])
    scored_segments = benchmark_results["scores"].merge(
        segmentation_results["customer_segments"][["CLIENTNUM", "cluster_label"]],
        on="CLIENTNUM",
        how="left",
    )
    save_frame(scored_segments, PATHS.output_predictions / "customer_risk_scores_with_segments.csv")

    return {
        "audit_summary": audit_summary,
        "unknown_strategy_df": unknown_strategy_df,
        "benchmark_results": benchmark_results,
        "diagnostics": diagnostics,
        "segmentation_results": segmentation_results,
    }


def _ensure_output_folders() -> None:
    folders = [
        PATHS.data_processed,
        PATHS.figures,
        PATHS.report_markdown,
        PATHS.models,
        PATHS.output_metrics,
        PATHS.output_predictions,
        PATHS.output_segmentation,
        PATHS.output_dashboard,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    run_project_pipeline()
