from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import IDENTIFIER_COLUMN, TARGET_COLUMN, TARGET_NAME, UNKNOWN_COLUMNS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["is_monoproduct"] = (featured["Total_Relationship_Count"] <= 1).astype(int)
    featured["is_dormant_3m"] = (featured["Months_Inactive_12_mon"] >= 3).astype(int)
    featured["transaction_amount_per_txn"] = featured["Total_Trans_Amt"] / featured["Total_Trans_Ct"].clip(lower=1)
    featured["high_contact_low_activity"] = (
        (featured["Contacts_Count_12_mon"] >= 4) & (featured["Total_Trans_Ct"] <= featured["Total_Trans_Ct"].median())
    ).astype(int)
    featured["declining_amt_flag"] = (featured["Total_Amt_Chng_Q4_Q1"] < 0.75).astype(int)
    featured["declining_count_flag"] = (featured["Total_Ct_Chng_Q4_Q1"] < 0.75).astype(int)
    featured["tenure_band"] = pd.cut(
        featured["Months_on_book"],
        bins=[0, 24, 36, 48, featured["Months_on_book"].max() + 1],
        labels=["early", "growing", "established", "mature"],
        include_lowest=True,
        right=False,
    ).astype(str)
    featured["utilization_band"] = pd.cut(
        featured["Avg_Utilization_Ratio"],
        bins=[-0.001, 0.1, 0.3, 0.6, 1.0],
        labels=["very_low", "low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return featured


def apply_unknown_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy not in {"keep", "missing"}:
        raise ValueError(f"Unknown strategy unsupported: {strategy}")

    transformed = df.copy()
    if strategy == "keep":
        return transformed

    for column in UNKNOWN_COLUMNS:
        indicator_name = f"{column}_was_unknown"
        transformed[indicator_name] = (transformed[column] == "Unknown").astype(int)
        transformed[column] = transformed[column].replace("Unknown", np.nan)
    return transformed


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN, TARGET_NAME], errors="ignore")
    customer_ids = X[IDENTIFIER_COLUMN].copy()
    X = X.drop(columns=[IDENTIFIER_COLUMN], errors="ignore")
    y = df[TARGET_NAME].copy()
    return X, y, customer_ids
