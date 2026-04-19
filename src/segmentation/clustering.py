from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.config import PATHS, RANDOM_STATE
from src.utils.io import save_frame, save_json


SEGMENTATION_FEATURES = [
    "Months_Inactive_12_mon",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Avg_Utilization_Ratio",
    "Total_Relationship_Count",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Months_on_book",
]


def run_segmentation(base_df: pd.DataFrame, scored_population: pd.DataFrame) -> dict[str, object]:
    segmentation_df = base_df[["CLIENTNUM", "churn_flag", *SEGMENTATION_FEATURES]].merge(
        scored_population[["CLIENTNUM", "churn_probability", "risk_decile"]],
        on="CLIENTNUM",
        how="left",
    )
    scaler = StandardScaler()
    matrix = scaler.fit_transform(segmentation_df[SEGMENTATION_FEATURES])

    best_score = -1.0
    best_model = None
    best_k = None
    candidates = []
    for k in range(3, 7):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(matrix)
        silhouette = silhouette_score(matrix, labels)
        candidates.append({"k": k, "silhouette": silhouette})
        if silhouette > best_score:
            best_score = silhouette
            best_model = model
            best_k = k

    candidate_df = pd.DataFrame(candidates)
    save_frame(candidate_df, PATHS.output_segmentation / "cluster_candidate_scores.csv")

    if best_score < 0.25:
        fallback = segmentation_df.copy()
        fallback["cluster_label"] = fallback.apply(_fallback_segment_label, axis=1)
        cluster_profile = (
            fallback.groupby("cluster_label")
            .agg(
                customers=("CLIENTNUM", "count"),
                churn_rate=("churn_flag", "mean"),
                avg_risk=("churn_probability", "mean"),
                avg_transactions=("Total_Trans_Ct", "mean"),
                avg_contacts=("Contacts_Count_12_mon", "mean"),
            )
            .reset_index()
        )
        save_frame(fallback, PATHS.output_segmentation / "customer_segments.csv")
        save_frame(cluster_profile, PATHS.output_segmentation / "cluster_profiles.csv")
        save_json(
            {
                "segmentation_mode": "fallback_personas",
                "best_silhouette": best_score,
                "reason": "Silhouette score below 0.25, clusters judged too weak for strong business interpretation.",
            },
            PATHS.output_segmentation / "segmentation_summary.json",
        )
        return {
            "mode": "fallback_personas",
            "best_silhouette": best_score,
            "customer_segments": fallback,
            "cluster_profiles": cluster_profile,
        }

    segmentation_df["cluster_label"] = best_model.fit_predict(matrix)
    cluster_profile = (
        segmentation_df.groupby("cluster_label")
        .agg(
            customers=("CLIENTNUM", "count"),
            churn_rate=("churn_flag", "mean"),
            avg_risk=("churn_probability", "mean"),
            avg_transactions=("Total_Trans_Ct", "mean"),
            avg_contacts=("Contacts_Count_12_mon", "mean"),
            avg_credit_limit=("Credit_Limit", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk", ascending=False)
        .reset_index(drop=True)
    )
    save_frame(segmentation_df, PATHS.output_segmentation / "customer_segments.csv")
    save_frame(cluster_profile, PATHS.output_segmentation / "cluster_profiles.csv")
    save_json(
        {
            "segmentation_mode": "kmeans",
            "selected_k": best_k,
            "best_silhouette": best_score,
        },
        PATHS.output_segmentation / "segmentation_summary.json",
    )
    return {
        "mode": "kmeans",
        "selected_k": best_k,
        "best_silhouette": best_score,
        "customer_segments": segmentation_df,
        "cluster_profiles": cluster_profile,
    }


def _fallback_segment_label(row: pd.Series) -> str:
    if row["churn_probability"] >= 0.55 and row["Months_Inactive_12_mon"] >= 3:
        return "Dormant high-risk"
    if row["churn_probability"] >= 0.50 and row["Contacts_Count_12_mon"] >= 3:
        return "Contacted but fragile"
    if row["Total_Relationship_Count"] <= 2:
        return "Mono-product exposed"
    return "Lower-risk active"
