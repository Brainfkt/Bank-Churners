from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.config import (
    IDENTIFIER_COLUMN,
    NAIVE_BAYES_PREFIX,
    NEGATIVE_CLASS_LABEL,
    PATHS,
    POSITIVE_CLASS_LABEL,
    TARGET_COLUMN,
    TARGET_NAME,
)


@dataclass
class AuditSummary:
    row_count: int
    column_count: int
    duplicated_rows: int
    duplicated_ids: int
    churn_rate: float


def load_raw_dataset(path=PATHS.raw_dataset) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_base_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    drop_columns = [column for column in cleaned.columns if column.startswith(NAIVE_BAYES_PREFIX)]
    cleaned = cleaned.drop(columns=drop_columns)
    cleaned[TARGET_NAME] = cleaned[TARGET_COLUMN].map(
        {
            POSITIVE_CLASS_LABEL: 1,
            NEGATIVE_CLASS_LABEL: 0,
        }
    )
    if cleaned[TARGET_NAME].isna().any():
        raise ValueError("Le mapping de la cible a échoué sur certaines lignes.")
    return cleaned


def build_audit_tables(df: pd.DataFrame) -> tuple[AuditSummary, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = AuditSummary(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        duplicated_rows=int(df.duplicated().sum()),
        duplicated_ids=int(df[IDENTIFIER_COLUMN].duplicated().sum()),
        churn_rate=float(df[TARGET_NAME].mean()),
    )

    missing_table = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_rate": df.isna().mean().values,
                "unknown_count": [int((df[column] == "Unknown").sum()) if column in df.columns else 0 for column in df.columns],
            }
        )
        .sort_values(["missing_rate", "unknown_count"], ascending=False)
        .reset_index(drop=True)
    )

    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_NAME], errors="ignore")
    numeric_profile = numeric_df.describe().T.reset_index().rename(columns={"index": "feature"})
    if not numeric_df.empty:
        numeric_profile["skewness"] = numeric_df.skew(numeric_only=True).values
        numeric_profile["iqr"] = numeric_profile["75%"] - numeric_profile["25%"]
        numeric_profile["outlier_count_iqr"] = [
            int(((numeric_df[column] < (numeric_df[column].quantile(0.25) - 1.5 * (numeric_df[column].quantile(0.75) - numeric_df[column].quantile(0.25)))) |
                 (numeric_df[column] > (numeric_df[column].quantile(0.75) + 1.5 * (numeric_df[column].quantile(0.75) - numeric_df[column].quantile(0.25))))).sum())
            for column in numeric_df.columns
        ]
    else:
        numeric_profile["skewness"] = []
        numeric_profile["iqr"] = []
        numeric_profile["outlier_count_iqr"] = []

    categorical_frames = []
    for column in _categorical_columns(df.columns):
        value_counts = df[column].astype(str).value_counts(dropna=False).reset_index()
        value_counts.columns = ["value", "count"]
        value_counts.insert(0, "column", column)
        value_counts["share"] = value_counts["count"] / len(df)
        categorical_frames.append(value_counts)
    categorical_profile = pd.concat(categorical_frames, ignore_index=True)

    return summary, missing_table, numeric_profile, categorical_profile


def _categorical_columns(columns: Iterable[str]) -> list[str]:
    excluded = {TARGET_NAME, TARGET_COLUMN, IDENTIFIER_COLUMN}
    return [
        column
        for column in columns
        if column not in excluded
        and column
        in {
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        }
    ]
