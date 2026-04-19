from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), list(numeric_features)),
            ("categorical", Pipeline(steps=categorical_steps), list(categorical_features)),
        ],
    )


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            encoder = transformer.named_steps["encoder"]
            feature_names.extend(encoder.get_feature_names_out(columns).tolist())
        else:
            feature_names.extend(list(columns))
    return feature_names
