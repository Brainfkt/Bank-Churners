from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationSummary:
    roc_auc: float
    pr_auc: float
    recall: float
    precision: float
    f1: float
    f2: float
    brier: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int


def summarize_predictions(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> ClassificationSummary:
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return ClassificationSummary(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        recall=float(recall_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        f2=float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        brier=float(brier_score_loss(y_true, probabilities)),
        threshold=float(threshold),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    min_precision: float = 0.30,
) -> dict[str, float]:
    candidate_thresholds = np.round(np.linspace(0.05, 0.95, 181), 3)
    records = []
    for threshold in candidate_thresholds:
        metrics = summarize_predictions(y_true, probabilities, threshold)
        records.append(
            {
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "f2": metrics.f2,
            }
        )
    frame = pd.DataFrame(records).sort_values(["f2", "recall", "precision"], ascending=False).reset_index(drop=True)
    eligible = frame[frame["precision"] >= min_precision]
    chosen = eligible.iloc[0] if not eligible.empty else frame.iloc[0]
    return chosen.to_dict()


def build_threshold_sensitivity(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 181), 3)

    records = []
    total = len(probabilities)
    for threshold in thresholds:
        metrics = summarize_predictions(y_true, probabilities, float(threshold))
        targeted = metrics.tp + metrics.fp
        records.append(
            {
                "threshold": float(threshold),
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "f2": metrics.f2,
                "targeted_customers": int(targeted),
                "targeted_rate": float(targeted / total) if total else 0.0,
                "true_positives": metrics.tp,
                "false_positives": metrics.fp,
                "false_negatives": metrics.fn,
                "true_negatives": metrics.tn,
            }
        )
    return pd.DataFrame(records)


def build_calibration_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"actual_churn": np.asarray(y_true), "probability": probabilities})
    frame["score_bin"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    base_rate = float(frame["actual_churn"].mean()) if not frame.empty else 0.0
    calibration = (
        frame.groupby("score_bin", observed=False)
        .agg(
            customers=("actual_churn", "count"),
            observed_churn_rate=("actual_churn", "mean"),
            mean_predicted_score=("probability", "mean"),
            min_score=("probability", "min"),
            max_score=("probability", "max"),
        )
        .reset_index()
    )
    calibration = calibration[calibration["customers"] > 0].copy()
    calibration["score_bin"] = calibration["score_bin"].astype(str)
    calibration["lift_vs_base_rate"] = np.where(
        base_rate > 0,
        calibration["observed_churn_rate"] / base_rate,
        np.nan,
    )
    return calibration.reset_index(drop=True)


def build_model_stability_report(
    validation_summary: ClassificationSummary,
    test_summary: ClassificationSummary,
) -> dict[str, object]:
    checks = {
        "pr_auc_gap": test_summary.pr_auc - validation_summary.pr_auc,
        "recall_gap": test_summary.recall - validation_summary.recall,
        "precision_gap": test_summary.precision - validation_summary.precision,
        "brier_gap": test_summary.brier - validation_summary.brier,
    }
    material_drop = checks["pr_auc_gap"] < -0.03 or checks["recall_gap"] < -0.05 or checks["precision_gap"] < -0.08
    if material_drop:
        interpretation = (
            "Le test montre une dégradation matérielle par rapport à la validation. "
            "Le modèle reste utilisable comme aide à la priorisation, mais la robustesse doit être relue avant tout usage plus productif."
        )
    else:
        interpretation = (
            "Les écarts validation/test restent contenus. Cela renforce la lecture du modèle comme outil de scoring, "
            "sans supprimer les limites liées au caractère statique du dataset."
        )
    return {
        "validation": validation_summary.__dict__,
        "test": test_summary.__dict__,
        "gaps_test_minus_validation": {key: float(value) for key, value in checks.items()},
        "material_drop_detected": bool(material_drop),
        "interpretation": interpretation,
    }


def build_slice_metrics(
    frame: pd.DataFrame,
    dimensions: list[str],
    y_true_column: str = "actual_churn",
    probability_column: str = "churn_probability",
    prediction_column: str = "predicted_label_recommended",
    min_cases: int = 30,
) -> pd.DataFrame:
    rows = []
    for dimension in dimensions:
        if dimension not in frame.columns:
            continue
        for value, subset in frame.groupby(dimension, dropna=False):
            y_true = subset[y_true_column].astype(int)
            y_score = subset[probability_column].astype(float)
            y_pred = subset[prediction_column].astype(int)
            has_both_classes = y_true.nunique() == 2
            tn = fp = fn = tp = 0
            if has_both_classes:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            else:
                labels = [0, 1]
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

            rows.append(
                {
                    "dimension": dimension,
                    "value": "Non renseigné" if pd.isna(value) else str(value),
                    "n": int(len(subset)),
                    "churn_rate": float(y_true.mean()) if len(subset) else 0.0,
                    "mean_score": float(y_score.mean()) if len(subset) else 0.0,
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "pr_auc": float(average_precision_score(y_true, y_score)) if has_both_classes and len(subset) >= min_cases else np.nan,
                    "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes and len(subset) >= min_cases else np.nan,
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_negatives": int(tn),
                    "is_interpretable": bool(has_both_classes and len(subset) >= min_cases),
                }
            )
    return pd.DataFrame(rows)


def summary_to_dict(prefix: str, summary: ClassificationSummary) -> dict[str, float]:
    payload = summary.__dict__.copy()
    return {f"{prefix}_{key}": value for key, value in payload.items()}
