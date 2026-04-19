from __future__ import annotations

import os

from src.utils.config import PATHS

os.environ.setdefault("MPLCONFIGDIR", str(PATHS.root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

from src.features.preprocessing import get_feature_names_from_preprocessor
from src.utils.io import save_frame, save_json


def export_model_diagnostics(best_model, bundle, validation_probabilities, test_probabilities, threshold: float) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "CLIENTNUM": bundle.ids_test,
            "actual_churn": bundle.y_test,
            "churn_probability": test_probabilities,
            "predicted_label_recommended": (test_probabilities >= threshold).astype(int),
        }
    )
    predictions["prediction_outcome"] = np.select(
        [
            (predictions["actual_churn"] == 1) & (predictions["predicted_label_recommended"] == 1),
            (predictions["actual_churn"] == 0) & (predictions["predicted_label_recommended"] == 1),
            (predictions["actual_churn"] == 1) & (predictions["predicted_label_recommended"] == 0),
        ],
        [
            "true_positive",
            "false_positive",
            "false_negative",
        ],
        default="true_negative",
    )
    save_frame(predictions, PATHS.output_predictions / "test_set_predictions.csv")
    _export_feature_importance(best_model, bundle.X_train)
    _export_shap_outputs(best_model, bundle.X_test, predictions)
    return predictions


def _resolve_pipeline_parts(best_model):
    if isinstance(best_model, CalibratedClassifierCV):
        base_estimator = best_model.calibrated_classifiers_[0].estimator
    else:
        base_estimator = best_model
    preprocessor = base_estimator.named_steps["preprocessor"]
    classifier = base_estimator.named_steps["classifier"]
    return preprocessor, classifier


def _export_feature_importance(best_model, X_train: pd.DataFrame) -> None:
    preprocessor, classifier = _resolve_pipeline_parts(best_model)
    transformed = preprocessor.transform(X_train)
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    else:
        importance = np.abs(classifier.coef_[0])

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    save_frame(importance_df, PATHS.output_metrics / "feature_importance_top20.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#2F5D62")
    ax.set_title("Top 20 predictive features")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(PATHS.figures / "feature_importance_top20.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _export_shap_outputs(best_model, X_test: pd.DataFrame, test_predictions: pd.DataFrame) -> None:
    preprocessor, classifier = _resolve_pipeline_parts(best_model)
    sample = X_test.copy().iloc[:400].reset_index(drop=True)
    sample_predictions = test_predictions.reset_index(drop=True).iloc[: len(sample)].copy()
    transformed = preprocessor.transform(sample)
    feature_names = get_feature_names_from_preprocessor(preprocessor)
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    try:
        explainer = shap.Explainer(classifier, transformed_df)
        shap_values = explainer(transformed_df)
    except Exception:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(transformed_df)

    plt.figure()
    shap.summary_plot(shap_values, transformed_df, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(PATHS.figures / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(PATHS.figures / "shap_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    explanation_cases = {}
    for outcome_name in ["true_positive", "false_positive", "false_negative"]:
        subset = sample_predictions.loc[sample_predictions["prediction_outcome"] == outcome_name]
        if subset.empty:
            continue
        customer_id = subset.sort_values("churn_probability", ascending=False).iloc[0]["CLIENTNUM"]
        row_position = sample_predictions.index[sample_predictions["CLIENTNUM"] == customer_id][0]
        explanation_cases[outcome_name] = {
            "CLIENTNUM": int(customer_id),
            "row_position": int(row_position),
        }
        plt.figure()
        shap.plots.waterfall(shap_values[row_position], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(PATHS.figures / f"shap_waterfall_{outcome_name}.png", dpi=180, bbox_inches="tight")
        plt.close()

    save_json(explanation_cases, PATHS.output_metrics / "local_explanation_cases.json")
    save_frame(pd.DataFrame(transformed_df.head(50)), PATHS.output_metrics / "shap_sample_transformed.csv")
