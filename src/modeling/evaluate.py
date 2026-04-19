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


def summary_to_dict(prefix: str, summary: ClassificationSummary) -> dict[str, float]:
    payload = summary.__dict__.copy()
    return {f"{prefix}_{key}": value for key, value in payload.items()}
