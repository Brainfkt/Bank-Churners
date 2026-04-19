from __future__ import annotations

import os

from src.utils.config import PATHS

os.environ.setdefault("MPLCONFIGDIR", str(PATHS.root / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay


sns.set_theme(style="whitegrid", palette="deep")


def plot_eda_suite(df: pd.DataFrame) -> None:
    _plot_target_distribution(df)
    _plot_churn_by_category(df, "Gender")
    _plot_churn_by_category(df, "Education_Level")
    _plot_churn_by_category(df, "Marital_Status")
    _plot_churn_by_category(df, "Income_Category")
    _plot_churn_by_category(df, "Card_Category")
    _plot_correlation_heatmap(df)
    for column in ["Credit_Limit", "Total_Trans_Amt", "Total_Trans_Ct", "Months_Inactive_12_mon", "Avg_Utilization_Ratio"]:
        _plot_boxplot(df, column)
    _plot_bivariate_risk_profiles(df)


def plot_model_curves(y_true, probabilities, predictions, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=axes[0], colorbar=False)
    axes[0].set_title(f"{title_prefix} - Confusion matrix")
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axes[1])
    axes[1].set_title(f"{title_prefix} - ROC")
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axes[2])
    axes[2].set_title(f"{title_prefix} - Precision-Recall")
    fig.tight_layout()
    fig.savefig(PATHS.figures / f"{title_prefix.lower().replace(' ', '_')}_model_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["churn_flag"].map({0: "Non churn", 1: "Churn"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#8FB339", "#D1495B"])
    ax.set_title("Churn distribution")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(PATHS.figures / "eda_churn_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_churn_by_category(df: pd.DataFrame, column: str) -> None:
    frame = (
        df.groupby(column, dropna=False)["churn_flag"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x=column, y="churn_flag", ax=ax, color="#3D5A80")
    ax.set_title(f"Churn rate by {column}")
    ax.set_ylabel("Churn rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PATHS.figures / f"eda_churn_by_{column.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    correlation = df.select_dtypes(include="number").drop(columns=["CLIENTNUM"], errors="ignore").corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(correlation, cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(PATHS.figures / "eda_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplot(df: pd.DataFrame, column: str) -> None:
    plot_df = df.copy()
    plot_df["churn_label"] = plot_df["churn_flag"].map({0: "Non churn", 1: "Churn"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=plot_df, x="churn_label", y=column, hue="churn_label", legend=False, ax=ax, palette=["#8FB339", "#D1495B"])
    ax.set_title(f"{column} by churn status")
    fig.tight_layout()
    fig.savefig(PATHS.figures / f"eda_violin_{column.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bivariate_risk_profiles(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df.sample(min(1500, len(df)), random_state=42),
        x="Total_Trans_Ct",
        y="Total_Trans_Amt",
        hue="churn_flag",
        ax=ax,
        alpha=0.7,
        palette={0: "#8FB339", 1: "#D1495B"},
    )
    ax.set_title("Transaction intensity and churn")
    fig.tight_layout()
    fig.savefig(PATHS.figures / "eda_transaction_intensity_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
