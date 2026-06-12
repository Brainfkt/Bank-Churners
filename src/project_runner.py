from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd

from src.data.load import build_audit_tables, load_raw_dataset, prepare_base_dataset
from src.modeling.evaluate import build_slice_metrics
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
    slice_metrics = _export_slice_metrics(base_df, diagnostics, scored_segments, benchmark_results["threshold"])
    run_manifest = _build_run_manifest(
        base_df=base_df,
        selected_strategy=selected_strategy,
        benchmark_results=benchmark_results,
        segmentation_results=segmentation_results,
        slice_metrics=slice_metrics,
    )
    save_json(run_manifest, PATHS.output_metrics / "run_manifest.json")

    return {
        "audit_summary": audit_summary,
        "unknown_strategy_df": unknown_strategy_df,
        "benchmark_results": benchmark_results,
        "diagnostics": diagnostics,
        "segmentation_results": segmentation_results,
        "slice_metrics": slice_metrics,
        "run_manifest": run_manifest,
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


def _export_slice_metrics(
    base_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    scored_segments: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    slice_frame = (
        test_predictions.merge(
            base_df[
                [
                    "CLIENTNUM",
                    "Gender",
                    "Income_Category",
                    "Card_Category",
                    "Customer_Age",
                ]
            ],
            on="CLIENTNUM",
            how="left",
        )
        .merge(scored_segments[["CLIENTNUM", "cluster_label"]], on="CLIENTNUM", how="left")
        .copy()
    )
    slice_frame["age_band"] = pd.cut(
        slice_frame["Customer_Age"],
        bins=[25, 39, 49, 59, 80],
        labels=["26 à 39 ans", "40 à 49 ans", "50 à 59 ans", "60 ans et plus"],
        include_lowest=True,
    ).astype(str)
    slice_frame["risk_band"] = pd.cut(
        slice_frame["churn_probability"],
        bins=[-0.001, 0.15, threshold, 0.50, 1.0],
        labels=["Faible", "Moyen", "Élevé", "Très élevé"],
        include_lowest=True,
    ).astype(str)
    slice_metrics = build_slice_metrics(
        slice_frame,
        dimensions=[
            "Gender",
            "Income_Category",
            "Card_Category",
            "age_band",
            "cluster_label",
            "risk_band",
        ],
    )
    save_frame(slice_metrics, PATHS.output_metrics / "slice_metrics.csv")
    return slice_metrics


def _build_run_manifest(
    base_df: pd.DataFrame,
    selected_strategy: str,
    benchmark_results: dict[str, object],
    segmentation_results: dict[str, object],
    slice_metrics: pd.DataFrame,
) -> dict[str, object]:
    tracked_artifacts = [
        PATHS.output_metrics / "model_selection_summary.json",
        PATHS.output_metrics / "threshold_sensitivity.csv",
        PATHS.output_metrics / "calibration_table.csv",
        PATHS.output_metrics / "slice_metrics.csv",
        PATHS.output_metrics / "model_stability.json",
        PATHS.output_predictions / "customer_risk_scores_with_segments.csv",
        PATHS.output_predictions / "test_set_predictions.csv",
        PATHS.output_segmentation / "segmentation_summary.json",
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_output(["git", "rev-parse", "HEAD"]),
            "branch": _git_output(["git", "branch", "--show-current"]),
            "dirty": bool(_git_output(["git", "status", "--short"])),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "dataset": {
            "rows": int(base_df.shape[0]),
            "columns": int(base_df.shape[1]),
            "churn_rate": float(base_df["churn_flag"].mean()),
        },
        "model": {
            "selected_model": str(benchmark_results["benchmark"].iloc[0]["candidate_name"]),
            "selected_unknown_strategy": selected_strategy,
            "recommended_threshold": float(benchmark_results["threshold"]),
            "test_pr_auc": float(benchmark_results["test_summary"].pr_auc),
            "test_recall": float(benchmark_results["test_summary"].recall),
            "test_precision": float(benchmark_results["test_summary"].precision),
        },
        "segmentation": {
            "mode": segmentation_results["mode"],
            "best_silhouette": float(segmentation_results["best_silhouette"]),
        },
        "monitoring": {
            "slice_metric_rows": int(len(slice_metrics)),
        },
        "artifacts": [
            {
                "path": str(path.relative_to(PATHS.root)),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
            for path in tracked_artifacts
        ],
    }


def _git_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=PATHS.root, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout.strip()


if __name__ == "__main__":
    run_project_pipeline()
