from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.features.engineering import add_engineered_features, apply_unknown_strategy, split_features_and_target
from src.features.preprocessing import build_preprocessor, infer_feature_types
from src.modeling.evaluate import optimize_threshold, summarize_predictions
from src.utils.config import PATHS, RANDOM_STATE
from src.utils.io import save_frame, save_json


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_val: pd.Series
    ids_test: pd.Series
    X_full: pd.DataFrame
    y_full: pd.Series
    ids_full: pd.Series


def build_dataset_bundle(df: pd.DataFrame, unknown_strategy: str) -> DatasetBundle:
    featured = add_engineered_features(df)
    featured = apply_unknown_strategy(featured, unknown_strategy)
    X, y, customer_ids = split_features_and_target(featured)

    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp,
        y_temp,
        ids_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return DatasetBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        ids_train=ids_train,
        ids_val=ids_val,
        ids_test=ids_test,
        X_full=X,
        y_full=y,
        ids_full=customer_ids,
    )


def compare_unknown_strategies(df: pd.DataFrame) -> pd.DataFrame:
    scorers = []
    for strategy in ("keep", "missing"):
        bundle = build_dataset_bundle(df, strategy)
        scorers.extend(
            [
                _cross_validated_pr_auc(bundle.X_train, bundle.y_train, strategy, model_family="logistic"),
                _cross_validated_pr_auc(bundle.X_train, bundle.y_train, strategy, model_family="xgboost"),
            ]
        )
    comparison = pd.DataFrame(scorers)
    pivot = comparison.groupby("unknown_strategy", as_index=False)["mean_pr_auc"].mean()
    keep_score = float(pivot.loc[pivot["unknown_strategy"] == "keep", "mean_pr_auc"].iloc[0])
    missing_score = float(pivot.loc[pivot["unknown_strategy"] == "missing", "mean_pr_auc"].iloc[0])
    selected = "keep" if (missing_score - keep_score) < 0.005 else "missing"
    comparison["selected_strategy"] = selected
    save_frame(comparison, PATHS.output_metrics / "unknown_strategy_comparison.csv")
    save_json(
        {
            "keep_mean_pr_auc": keep_score,
            "missing_mean_pr_auc": missing_score,
            "selected_strategy": selected,
            "selection_rule": "Missing retained only if mean PR-AUC gain >= 0.005 across Logistic Regression and XGBoost.",
        },
        PATHS.output_metrics / "unknown_strategy_decision.json",
    )
    return comparison


def train_full_benchmark(df: pd.DataFrame, unknown_strategy: str) -> dict[str, object]:
    bundle = build_dataset_bundle(df, unknown_strategy)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    positive_weight = (bundle.y_train == 0).sum() / max((bundle.y_train == 1).sum(), 1)
    search_results: list[dict[str, object]] = []

    search_specs = [
        _build_logistic_search(bundle.X_train, bundle.y_train, cv=cv, balanced=False, smote=False),
        _build_logistic_search(bundle.X_train, bundle.y_train, cv=cv, balanced=True, smote=False),
        _build_logistic_search(bundle.X_train, bundle.y_train, cv=cv, balanced=False, smote=True),
        _build_random_forest_search(bundle.X_train, bundle.y_train, cv=cv, balanced=False, smote=False),
        _build_random_forest_search(bundle.X_train, bundle.y_train, cv=cv, balanced=True, smote=False),
        _build_random_forest_search(bundle.X_train, bundle.y_train, cv=cv, balanced=False, smote=True),
        _build_xgboost_search(bundle.X_train, bundle.y_train, cv=cv, scale_pos_weight=positive_weight),
    ]

    for spec in search_specs:
        searcher = spec["search"]
        searcher.fit(bundle.X_train, bundle.y_train)
        best_model = searcher.best_estimator_
        val_probabilities = best_model.predict_proba(bundle.X_val)[:, 1]
        val_summary = summarize_predictions(bundle.y_val, val_probabilities, threshold=0.50)
        search_results.append(
            {
                "candidate_name": spec["name"],
                "model_family": spec["family"],
                "imbalance_strategy": spec["imbalance_strategy"],
                "cv_best_pr_auc": float(searcher.best_score_),
                "validation_pr_auc": val_summary.pr_auc,
                "validation_recall": val_summary.recall,
                "validation_precision": val_summary.precision,
                "validation_f1": val_summary.f1,
                "validation_f2": val_summary.f2,
                "validation_roc_auc": val_summary.roc_auc,
                "best_params": searcher.best_params_,
                "estimator": best_model,
            }
        )

    benchmark_df = pd.DataFrame(search_results).sort_values(
        ["validation_pr_auc", "validation_recall", "validation_precision"],
        ascending=False,
    ).reset_index(drop=True)
    best_entry = benchmark_df.iloc[0].to_dict()
    best_model = best_entry["estimator"]

    calibration_summary = _evaluate_calibration(best_model, bundle)
    if calibration_summary["selected"] == "calibrated":
        best_model = calibration_summary["model"]
        best_entry["candidate_name"] = f"{best_entry['candidate_name']}_calibrated"

    validation_probabilities = best_model.predict_proba(bundle.X_val)[:, 1]
    threshold_payload = optimize_threshold(bundle.y_val, validation_probabilities, min_precision=0.30)
    chosen_threshold = float(threshold_payload["threshold"])

    test_probabilities = best_model.predict_proba(bundle.X_test)[:, 1]
    validation_summary = summarize_predictions(bundle.y_val, validation_probabilities, threshold=chosen_threshold)
    test_summary = summarize_predictions(bundle.y_test, test_probabilities, threshold=chosen_threshold)

    full_fit_model = clone(best_model)
    X_train_val = pd.concat([bundle.X_train, bundle.X_val], axis=0)
    y_train_val = pd.concat([bundle.y_train, bundle.y_val], axis=0)
    full_fit_model.fit(X_train_val, y_train_val)
    all_probabilities = full_fit_model.predict_proba(bundle.X_full)[:, 1]
    scored_population = pd.DataFrame(
        {
            "CLIENTNUM": bundle.ids_full,
            "churn_probability": all_probabilities,
            "predicted_label_0_5": (all_probabilities >= 0.50).astype(int),
            "predicted_label_recommended": (all_probabilities >= chosen_threshold).astype(int),
        }
    )
    scored_population["risk_decile"] = pd.qcut(scored_population["churn_probability"], q=10, labels=False, duplicates="drop") + 1
    save_frame(scored_population, PATHS.output_predictions / "customer_risk_scores.csv")

    benchmark_to_save = benchmark_df.drop(columns=["estimator"]).copy()
    benchmark_to_save["best_params"] = benchmark_to_save["best_params"].astype(str)
    save_frame(benchmark_to_save, PATHS.output_metrics / "model_benchmark.csv")
    save_json(
        {
            "selected_model": best_entry["candidate_name"],
            "selected_unknown_strategy": unknown_strategy,
            "threshold_policy": threshold_payload,
            "calibration_decision": {k: v for k, v in calibration_summary.items() if k != "model"},
            "validation_metrics": validation_summary.__dict__,
            "test_metrics": test_summary.__dict__,
        },
        PATHS.output_metrics / "model_selection_summary.json",
    )
    joblib.dump(full_fit_model, PATHS.models / "final_model.joblib")

    return {
        "bundle": bundle,
        "benchmark": benchmark_df,
        "best_model_validation": best_model,
        "best_model_full": full_fit_model,
        "threshold": chosen_threshold,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "scores": scored_population,
    }


def _cross_validated_pr_auc(X: pd.DataFrame, y: pd.Series, strategy: str, model_family: str) -> dict[str, object]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    numeric_features, categorical_features = infer_feature_types(X)
    if model_family == "logistic":
        preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid={"classifier__C": [0.1, 1.0, 5.0, 10.0]},
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
        )
    else:
        preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [3, 4, 5],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
            },
            n_iter=4,
            random_state=RANDOM_STATE,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
        )

    search.fit(X, y)
    return {
        "unknown_strategy": strategy,
        "model_family": model_family,
        "mean_pr_auc": float(search.best_score_),
    }


def _build_logistic_search(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold, balanced: bool, smote: bool) -> dict[str, object]:
    numeric_features, categorical_features = infer_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    classifier = LogisticRegression(
        max_iter=2_000,
        random_state=RANDOM_STATE,
        class_weight="balanced" if balanced else None,
    )
    steps = [("preprocessor", preprocessor)]
    if smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("classifier", classifier))
    pipeline = ImbPipeline(steps=steps)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"classifier__C": [0.1, 1.0, 5.0, 10.0]},
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
    )
    label = "logistic_smote" if smote else ("logistic_balanced" if balanced else "logistic_plain")
    imbalance_strategy = "smote" if smote else ("class_weight" if balanced else "none")
    return {
        "name": label,
        "family": "Logistic Regression",
        "imbalance_strategy": imbalance_strategy,
        "search": search,
    }


def _build_random_forest_search(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold, balanced: bool, smote: bool) -> dict[str, object]:
    numeric_features, categorical_features = infer_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced_subsample" if balanced else None,
    )
    steps = [("preprocessor", preprocessor)]
    if smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("classifier", classifier))
    pipeline = ImbPipeline(steps=steps)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "classifier__n_estimators": [250, 400, 550],
            "classifier__max_depth": [6, 10, None],
            "classifier__min_samples_leaf": [1, 5, 10],
            "classifier__max_features": ["sqrt", 0.7, None],
        },
        n_iter=6 if not smote else 4,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    label = "random_forest_smote" if smote else ("random_forest_balanced" if balanced else "random_forest_plain")
    imbalance_strategy = "smote" if smote else ("class_weight" if balanced else "none")
    return {
        "name": label,
        "family": "Random Forest",
        "imbalance_strategy": imbalance_strategy,
        "search": search,
    }


def _build_xgboost_search(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold, scale_pos_weight: float) -> dict[str, object]:
    numeric_features, categorical_features = infer_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=1,
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "classifier__n_estimators": [200, 300, 450],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
        },
        n_iter=8,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    return {
        "name": "xgboost_weighted",
        "family": "XGBoost",
        "imbalance_strategy": "scale_pos_weight",
        "search": search,
    }


def _evaluate_calibration(best_model, bundle: DatasetBundle) -> dict[str, object]:
    raw_probabilities = best_model.predict_proba(bundle.X_val)[:, 1]
    raw_brier = summarize_predictions(bundle.y_val, raw_probabilities, threshold=0.50).brier

    calibrated_model = CalibratedClassifierCV(best_model, cv=5, method="sigmoid")
    calibrated_model.fit(bundle.X_train, bundle.y_train)
    calibrated_probabilities = calibrated_model.predict_proba(bundle.X_val)[:, 1]
    calibrated_summary = summarize_predictions(bundle.y_val, calibrated_probabilities, threshold=0.50)
    raw_pr_auc = average_precision_score(bundle.y_val, raw_probabilities)

    selected = "calibrated" if calibrated_summary.brier < (raw_brier - 0.002) and calibrated_summary.pr_auc >= (raw_pr_auc - 0.003) else "raw"
    return {
        "selected": selected,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_summary.brier,
        "raw_pr_auc": raw_pr_auc,
        "calibrated_pr_auc": calibrated_summary.pr_auc,
        "model": calibrated_model if selected == "calibrated" else best_model,
    }
