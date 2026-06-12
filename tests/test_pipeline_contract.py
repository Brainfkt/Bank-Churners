import numpy as np

from src.data.load import load_raw_dataset, prepare_base_dataset
from src.features.engineering import add_engineered_features, apply_unknown_strategy
from src.modeling.evaluate import build_calibration_table, build_threshold_sensitivity, optimize_threshold
from src.modeling.train import build_dataset_bundle
from src.segmentation.clustering import SEGMENTATION_FEATURES


def test_missing_strategy_creates_indicator_columns():
    df = prepare_base_dataset(load_raw_dataset())
    featured = add_engineered_features(df)
    transformed = apply_unknown_strategy(featured, "missing")
    for column in ["Education_Level", "Marital_Status", "Income_Category"]:
        assert f"{column}_was_unknown" in transformed.columns


def test_dataset_bundle_split_sizes_are_conservative():
    df = prepare_base_dataset(load_raw_dataset())
    bundle = build_dataset_bundle(df, unknown_strategy="keep")
    total = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)
    assert total == len(df)
    assert bundle.y_train.mean() > 0
    assert bundle.y_test.mean() > 0


def test_threshold_optimizer_outputs_valid_cutoff():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.55, 0.75, 0.9])
    payload = optimize_threshold(y_true, probabilities, min_precision=0.3)
    assert 0.05 <= payload["threshold"] <= 0.95
    assert payload["f2"] >= 0


def test_threshold_sensitivity_contract():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.55, 0.75, 0.9])
    table = build_threshold_sensitivity(y_true, probabilities, thresholds=np.array([0.3, 0.6]))
    expected_columns = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "targeted_customers",
        "targeted_rate",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
    }
    assert expected_columns.issubset(table.columns)
    assert table["targeted_rate"].between(0, 1).all()


def test_calibration_table_contract():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
    table = build_calibration_table(y_true, probabilities, n_bins=5)
    assert {"customers", "observed_churn_rate", "mean_predicted_score", "lift_vs_base_rate"}.issubset(table.columns)
    assert table["customers"].sum() == len(y_true)


def test_segmentation_feature_contract():
    df = prepare_base_dataset(load_raw_dataset())
    for column in SEGMENTATION_FEATURES:
        assert column in df.columns
