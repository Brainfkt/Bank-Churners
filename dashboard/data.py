from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from dashboard.content import (
    DISPLAY_TABLE_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_LABELS,
    FILTER_LABELS,
    MODEL_FAMILY_LABELS,
    MODEL_LABELS,
    PERSONA_DESCRIPTIONS,
    PERSONA_LABELS,
    PERSONA_ACTIONS,
    RISK_BAND_DESCRIPTIONS,
    UI_LABELS,
    VALUE_TRANSLATIONS,
    IMBALANCE_LABELS,
)


@dataclass
class DashboardBundle:
    population: pd.DataFrame
    benchmark: pd.DataFrame
    feature_importance: pd.DataFrame
    model_summary: dict
    unknown_strategy: dict
    local_explanations: dict
    segmentation_summary: dict
    threshold_sensitivity: pd.DataFrame
    calibration_table: pd.DataFrame
    slice_metrics: pd.DataFrame
    model_stability: dict
    run_manifest: dict
    figures_dir: Path
    root: Path


def load_dashboard_bundle(root: Path) -> DashboardBundle:
    outputs = root / "outputs"
    figures_dir = root / "reports" / "figures"
    base = pd.read_csv(root / "data" / "processed" / "bank_churners_base.csv")
    scores = pd.read_csv(outputs / "predictions" / "customer_risk_scores_with_segments.csv")
    predictions = pd.read_csv(outputs / "predictions" / "test_set_predictions.csv")
    benchmark = pd.read_csv(outputs / "metrics" / "model_benchmark.csv")
    feature_importance = pd.read_csv(outputs / "metrics" / "feature_importance_top20.csv")
    threshold_sensitivity = pd.read_csv(outputs / "metrics" / "threshold_sensitivity.csv")
    calibration_table = pd.read_csv(outputs / "metrics" / "calibration_table.csv")
    slice_metrics = pd.read_csv(outputs / "metrics" / "slice_metrics.csv")
    model_summary = _load_json(outputs / "metrics" / "model_selection_summary.json")
    unknown_strategy = _load_json(outputs / "metrics" / "unknown_strategy_decision.json")
    local_explanations = _load_json(outputs / "metrics" / "local_explanation_cases.json")
    segmentation_summary = _load_json(outputs / "segmentation" / "segmentation_summary.json")
    model_stability = _load_json(outputs / "metrics" / "model_stability.json")
    run_manifest = _load_json(outputs / "metrics" / "run_manifest.json")

    population = _prepare_population(base, scores, predictions, model_summary["threshold_policy"]["threshold"])
    benchmark_display = _prepare_benchmark(benchmark)
    importance_display = _prepare_feature_importance(feature_importance)
    slice_metrics_display = _prepare_slice_metrics(slice_metrics)

    return DashboardBundle(
        population=population,
        benchmark=benchmark_display,
        feature_importance=importance_display,
        model_summary=model_summary,
        unknown_strategy=unknown_strategy,
        local_explanations=local_explanations,
        segmentation_summary=segmentation_summary,
        threshold_sensitivity=threshold_sensitivity,
        calibration_table=calibration_table,
        slice_metrics=slice_metrics_display,
        model_stability=model_stability,
        run_manifest=run_manifest,
        figures_dir=figures_dir,
        root=root,
    )


def available_filter_options(df: pd.DataFrame) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for column in FILTER_LABELS:
        values = sorted([value for value in df[column].dropna().unique().tolist() if value != ""])
        options[column] = values
    return options


def apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = df.copy()
    for column, selected in filters.items():
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    return filtered


def format_active_filters(filters: dict[str, list[str]]) -> list[str]:
    summary = []
    for column, values in filters.items():
        if values:
            summary.append(f"{FILTER_LABELS[column]} : {', '.join(values)}")
    return summary


def summarize_population(df: pd.DataFrame) -> dict[str, object]:
    dominant_segment = df["segment_client"].mode().iloc[0] if not df.empty else "Aucun"
    return {
        "clients": int(df.shape[0]),
        "taux_churn": float(df["churn_observe"].mean()) if not df.empty else 0.0,
        "score_moyen": float(df["score_churn"].mean()) if not df.empty else 0.0,
        "part_haut_risque": float(df["alerte_seuil_recommande"].mean()) if not df.empty else 0.0,
        "segment_dominant": dominant_segment,
    }


def build_executive_story(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {
            "resume": "Aucun client n'est visible dans le périmètre courant.",
            "signal": "Les filtres actifs excluent tout le portefeuille.",
            "priorite": "Aucune priorisation n'est possible sur ce sous-ensemble vide.",
        }

    summary = summarize_population(df)
    high_risk = df[df["bande_risque"].isin(["Élevé", "Très élevé"])]
    dominant_activity = df["statut_activite"].mode().iloc[0] if not df["statut_activite"].empty else "Non disponible"

    if high_risk.empty:
        signal = (
            "Le portefeuille filtré reste majoritairement sous contrôle : les clients en risque élevé "
            "ou très élevé y sont minoritaires."
        )
    else:
        high_risk_share = len(high_risk) / len(df)
        top_high_risk_segment = high_risk["segment_client"].mode().iloc[0]
        signal = (
            f"{high_risk_share:.1%} du périmètre se situe déjà en risque élevé ou très élevé, "
            f"avec une concentration notable sur le segment « {top_high_risk_segment} »."
        )

    return {
        "resume": (
            f"Le périmètre affiché contient {summary['clients']:,} clients, avec un taux de churn observé de "
            f"{summary['taux_churn']:.1%} et un score moyen de {summary['score_moyen']:.3f}."
        ).replace(",", " "),
        "signal": signal,
        "priorite": (
            f"Le segment le plus présent est « {summary['segment_dominant']} », tandis que le statut d'activité "
            f"le plus fréquent est « {dominant_activity} »."
        ),
    }


def build_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["segment_client", "description_segment", "part_population", "taux_churn", "score_moyen", "transactions_moyennes", "contacts_moyens"]
        )

    total = len(df)
    grouped = (
        df.groupby("segment_client")
        .agg(
            clients=("identifiant_client", "count"),
            taux_churn=("churn_observe", "mean"),
            score_moyen=("score_churn", "mean"),
            transactions_moyennes=("nb_transactions_total", "mean"),
            contacts_moyens=("nb_contacts_12m", "mean"),
            mois_inactifs_moyens=("mois_inactifs_12m", "mean"),
            relations_moyennes=("nb_relations_total", "mean"),
        )
        .reset_index()
    )
    grouped["part_population"] = grouped["clients"] / total
    grouped["description_segment"] = grouped["segment_client"].map(PERSONA_DESCRIPTIONS)
    grouped = grouped.sort_values(["score_moyen", "taux_churn"], ascending=False).reset_index(drop=True)
    return grouped


def build_persona_difference_table(persona_frame: pd.DataFrame, baseline_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Indicateur", "Persona", "Reste du périmètre", "Écart relatif"]
    if persona_frame.empty or baseline_frame.empty:
        return pd.DataFrame(columns=columns)

    metrics = {
        "Score moyen de churn": "score_churn",
        "Taux de churn observé": "churn_observe",
        "Nombre moyen de transactions": "nb_transactions_total",
        "Mois d'inactivité moyens": "mois_inactifs_12m",
        "Nombre moyen de contacts": "nb_contacts_12m",
        "Nombre moyen de relations": "nb_relations_total",
    }
    rows = []
    for label, column in metrics.items():
        persona_value = float(persona_frame[column].mean())
        baseline_value = float(baseline_frame[column].mean())
        if baseline_value == 0:
            relative_gap = np.nan
        else:
            relative_gap = (persona_value - baseline_value) / baseline_value
        rows.append(
            {
                "Indicateur": label,
                "Persona": persona_value,
                "Reste du périmètre": baseline_value,
                "Écart relatif": relative_gap,
            }
        )
    return pd.DataFrame(rows)


def build_dimension_profile(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[dimension, "clients", "taux_churn", "score_moyen", "montant_moyen", "transactions_moyennes"])
    grouped = (
        df.groupby(dimension)
        .agg(
            clients=("identifiant_client", "count"),
            taux_churn=("churn_observe", "mean"),
            score_moyen=("score_churn", "mean"),
            montant_moyen=("montant_transactions_total", "mean"),
            transactions_moyennes=("nb_transactions_total", "mean"),
        )
        .reset_index()
        .sort_values("taux_churn", ascending=False)
    )
    return grouped


def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df[DISPLAY_TABLE_COLUMNS].copy()
    display["score_churn"] = display["score_churn"].round(3)
    display["alerte_seuil_recommande"] = display["alerte_seuil_recommande"].map({1: "Oui", 0: "Non"})
    return display.rename(columns={column: UI_LABELS.get(column, column) for column in display.columns})


def build_work_queue(df: pd.DataFrame, threshold: float, limit: int = 200) -> pd.DataFrame:
    queue = df[df["score_churn"] >= threshold].copy()
    if queue.empty:
        queue = df.copy()
    queue = queue.sort_values(["score_churn", "mois_inactifs_12m", "nb_contacts_12m"], ascending=False).head(limit)
    display = pd.DataFrame(
        {
            "Référence client": queue["identifiant_client"],
            "Score": queue["score_churn"].round(3),
            "Bande": queue["bande_risque"],
            "Persona": queue["segment_client"],
            "Action": queue.apply(_recommended_customer_action, axis=1),
            "Mois inactifs": queue["mois_inactifs_12m"].astype(int),
            "Contacts 12m": queue["nb_contacts_12m"].astype(int),
        }
    )
    return display.reset_index(drop=True)


def build_decile_churn_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Décile", "Taux de churn", "Score moyen", "Clients"])
    grouped = (
        df.groupby("score_decile")
        .agg(
            clients=("identifiant_client", "count"),
            taux_churn=("churn_observe", "mean"),
            score_moyen=("score_churn", "mean"),
        )
        .reset_index()
        .sort_values("score_decile")
    )
    grouped["Décile"] = grouped["score_decile"].map(lambda value: f"D{int(value)}")
    return grouped.rename(
        columns={
            "clients": "Clients",
            "taux_churn": "Taux de churn",
            "score_moyen": "Score moyen",
        }
    )[["Décile", "Taux de churn", "Score moyen", "Clients"]]


def find_customer_by_reference(df: pd.DataFrame, reference: str | None) -> pd.Series | None:
    if not reference:
        return None
    match = df[df["identifiant_client"] == reference]
    if match.empty:
        return None
    return match.iloc[0]


def build_customer_explanation(row: pd.Series | None) -> dict[str, str]:
    if row is None:
        return {
            "reference": "Aucun client sélectionné",
            "risk": "Sélectionnez une ligne dans la file d'actions pour ouvrir une lecture locale.",
            "signals": "Aucun signal individuel n'est affiché sans sélection explicite.",
            "action": "Aucune action recommandée.",
        }
    risk = (
        f"{row['identifiant_client']} se situe en bande « {row['bande_risque']} » "
        f"avec un score de churn de {float(row['score_churn']):.3f}."
    )
    signals = (
        f"Le profil combine {int(row['mois_inactifs_12m'])} mois d'inactivité sur 12 mois, "
        f"{int(row['nb_transactions_total'])} transactions et {int(row['nb_contacts_12m'])} contacts récents."
    )
    action = PERSONA_ACTIONS.get(row["segment_client"], "Qualifier le contexte client avant toute action commerciale.")
    return {
        "reference": str(row["identifiant_client"]),
        "risk": risk,
        "signals": signals,
        "action": action,
    }


def build_campaign_editor_seed(df: pd.DataFrame) -> pd.DataFrame:
    action_table = build_action_table(df)
    if action_table.empty:
        return pd.DataFrame(columns=["Persona", "Action", "Cap clients", "Priorité"])
    seed = action_table[["segment_client", "triage_action", "clients", "priorite"]].copy()
    seed["Cap clients"] = seed["clients"].map(lambda value: int(min(max(value * 0.25, 25), 500)))
    seed = seed.rename(columns={"segment_client": "Persona", "triage_action": "Action", "priorite": "Priorité"})
    return seed[["Persona", "Action", "Cap clients", "Priorité"]]


def build_streamlit_capability_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Capacité Streamlit": "st.navigation / st.Page", "Usage métier": "Navigation applicative par workflows"},
            {"Capacité Streamlit": "st.popover", "Usage métier": "Filtres avancés sans saturer l'écran"},
            {"Capacité Streamlit": "st.pills / st.segmented_control", "Usage métier": "Filtres rapides et modes de lecture"},
            {"Capacité Streamlit": "st.metric", "Usage métier": "KPI exécutifs avec contexte"},
            {"Capacité Streamlit": "st.plotly_chart(on_select)", "Usage métier": "Sélection de persona depuis un graphique"},
            {"Capacité Streamlit": "st.dataframe(on_select)", "Usage métier": "Sélection d'un client à investiguer"},
            {"Capacité Streamlit": "st.dialog", "Usage métier": "Explication locale du risque client"},
            {"Capacité Streamlit": "st.data_editor", "Usage métier": "Simulation prudente d'un plan de contact"},
            {"Capacité Streamlit": "st.download_button", "Usage métier": "Export opérationnel des listes visibles"},
            {"Capacité Streamlit": "st.status / st.toast", "Usage métier": "Fraîcheur des artefacts et feedback utilisateur"},
            {"Capacité Streamlit": "st.query_params", "Usage métier": "Vue filtrée partageable par URL"},
            {"Capacité Streamlit": "st.cache_data", "Usage métier": "Chargement rapide des artefacts analytiques"},
        ]
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def threshold_row(threshold_sensitivity: pd.DataFrame, threshold: float) -> pd.Series:
    row_index = (threshold_sensitivity["threshold"] - threshold).abs().idxmin()
    return threshold_sensitivity.loc[row_index]


def build_artifact_health_table(root: Path, manifest: dict) -> pd.DataFrame:
    rows = []
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        rows.append(
            {
                "Artefact": artifact["path"],
                "Présent": bool(path.exists()),
                "Taille Ko": round(path.stat().st_size / 1024, 1) if path.exists() else 0.0,
                "Dernière modification": pd.to_datetime(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S") if path.exists() else "n.d.",
            }
        )
    return pd.DataFrame(rows)


def get_filtered_test_subset(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    subset = df[df["est_dans_test"] == 1].copy()
    if subset.empty:
        return subset, "Aucun client du portefeuille filtré n'appartient à l'échantillon de test. Les métriques filtrées ne sont donc pas interprétables."
    if len(subset) < 30:
        return subset, "Le sous-échantillon de test filtré est trop petit pour une lecture fiable des métriques."
    if subset["churn_reel_test"].nunique() < 2:
        return subset, "Le sous-échantillon filtré ne contient pas à la fois des churners et des non churners. La matrice de confusion locale serait trompeuse."
    return subset, None


def compute_filtered_metrics(test_subset: pd.DataFrame) -> dict[str, object]:
    y_true = test_subset["churn_reel_test"].astype(int)
    y_score = test_subset["score_churn"].astype(float)
    y_pred = test_subset["prediction_seuil_test"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(test_subset)),
    }


def persona_differences(persona_frame: pd.DataFrame, baseline_frame: pd.DataFrame) -> list[str]:
    if persona_frame.empty or baseline_frame.empty:
        return []

    comparisons = {
        "mois_inactifs_12m": ("plus d'inactivité", True),
        "nb_transactions_total": ("plus de transactions", False),
        "nb_contacts_12m": ("plus de contacts", False),
        "score_churn": ("un score de churn plus élevé", False),
        "nb_relations_total": ("une relation plus profonde", False),
    }
    insights: list[str] = []
    for column, (label, invert) in comparisons.items():
        persona_mean = persona_frame[column].mean()
        baseline_mean = baseline_frame[column].mean()
        if baseline_mean == 0:
            continue
        ratio = persona_mean / baseline_mean
        if ratio >= 1.15:
            insights.append(f"{label.capitalize()} que le reste du périmètre filtré ({persona_mean:.1f} contre {baseline_mean:.1f}).")
        elif ratio <= 0.85:
            if invert:
                insights.append(f"Moins d'inactivité que le reste du périmètre ({persona_mean:.1f} contre {baseline_mean:.1f}).")
            else:
                insights.append(f"Moins de {label.split('plus de ')[-1]} que le reste du périmètre ({persona_mean:.1f} contre {baseline_mean:.1f}).")
    return insights[:3]


def build_action_table(df: pd.DataFrame) -> pd.DataFrame:
    segment_table = build_segment_table(df)
    if segment_table.empty:
        return segment_table
    action_df = segment_table.copy()
    action_df["priorite"] = np.select(
        [
            action_df["score_moyen"] >= 0.80,
            action_df["score_moyen"] >= 0.40,
        ],
        ["Priorité très élevée", "Priorité élevée"],
        default="Priorité de veille",
    )
    action_df["hypothese_action"] = action_df["segment_client"].map(PERSONA_ACTIONS)
    action_df["triage_action"] = np.select(
        [
            action_df["score_moyen"] >= 0.80,
            (action_df["score_moyen"] >= 0.45) | (action_df["taux_churn"] >= 0.25),
            action_df["score_moyen"] >= 0.20,
        ],
        [
            "Agir maintenant",
            "Cibler rapidement",
            "Surveiller de près",
        ],
        default="Veille légère",
    )
    action_df["logique_priorisation"] = np.select(
        [
            action_df["triage_action"] == "Agir maintenant",
            action_df["triage_action"] == "Cibler rapidement",
            action_df["triage_action"] == "Surveiller de près",
        ],
        [
            "Score moyen très élevé : le segment concentre déjà des signaux de désengagement nets.",
            "Risque significatif ou churn observé déjà marqué : un ciblage rapide peut éviter des pertes additionnelles.",
            "Risque intermédiaire : ces clients méritent une veille active avant dégradation plus nette.",
        ],
        default="Le segment reste utile à suivre, mais ne justifie pas une activation prioritaire immédiate.",
    )
    return action_df


def build_high_risk_comparison(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Indicateur", "Clients à risque élevé", "Reste du portefeuille", "Écart relatif"]
    high_risk = df[df["bande_risque"].isin(["Élevé", "Très élevé"])]
    rest = df[~df["bande_risque"].isin(["Élevé", "Très élevé"])]
    if high_risk.empty or rest.empty:
        return pd.DataFrame(columns=columns)

    metrics = {
        "Score moyen de churn": "score_churn",
        "Taux de churn observé": "churn_observe",
        "Mois d'inactivité moyens": "mois_inactifs_12m",
        "Nombre moyen de transactions": "nb_transactions_total",
        "Nombre moyen de contacts": "nb_contacts_12m",
        "Nombre moyen de relations": "nb_relations_total",
    }
    rows = []
    for label, column in metrics.items():
        risk_value = float(high_risk[column].mean())
        rest_value = float(rest[column].mean())
        gap = np.nan if rest_value == 0 else (risk_value - rest_value) / rest_value
        rows.append(
            {
                "Indicateur": label,
                "Clients à risque élevé": risk_value,
                "Reste du portefeuille": rest_value,
                "Écart relatif": gap,
            }
        )
    return pd.DataFrame(rows)


def build_local_explanation_table(df: pd.DataFrame, cases: dict) -> pd.DataFrame:
    if not cases:
        return pd.DataFrame(
            columns=["Cas", "Référence client", "Segment client", "Bande de risque", "Score de churn", "Issue de prédiction"]
        )

    rows = []
    labels = {
        "true_positive": "Churner bien détecté",
        "false_positive": "Fausse alerte",
        "false_negative": "Churner manqué",
    }
    for case_key, payload in cases.items():
        client_id = payload.get("CLIENTNUM")
        match = df[df["clientnum_source"] == client_id]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "Cas": labels.get(case_key, case_key),
                "Référence client": row["identifiant_client"],
                "Segment client": row["segment_client"],
                "Bande de risque": row["bande_risque"],
                "Score de churn": float(row["score_churn"]),
                "Issue de prédiction": row["prediction_resultat"],
            }
        )
    return pd.DataFrame(rows)


def _recommended_customer_action(row: pd.Series) -> str:
    if row["bande_risque"] == "Très élevé":
        return "Appel proactif"
    if row["statut_activite"] in {"Très faible activité", "En retrait"}:
        return "Réactivation"
    if row["profil_produit"] == "Mono-produit":
        return "Offre relationnelle"
    if row["nb_contacts_12m"] >= 4:
        return "Revue qualitative"
    return "Veille ciblée"


def _prepare_population(base: pd.DataFrame, scores: pd.DataFrame, predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    merged = (
        base.merge(scores, on="CLIENTNUM", how="inner")
        .merge(
            predictions[
                [
                    "CLIENTNUM",
                    "actual_churn",
                    "predicted_label_recommended",
                    "prediction_outcome",
                ]
            ].rename(
                columns={
                    "actual_churn": "churn_reel_test",
                    "predicted_label_recommended": "prediction_seuil_test",
                    "prediction_outcome": "prediction_resultat_brut",
                }
            ),
            on="CLIENTNUM",
            how="left",
        )
        .copy()
    )

    merged["clientnum_source"] = merged["CLIENTNUM"]
    merged["identifiant_client"] = merged["CLIENTNUM"].map(_anonymize_client_id)
    merged["age_client"] = merged["Customer_Age"]
    merged["nb_personnes_a_charge"] = merged["Dependent_count"]
    merged["anciennete_mois"] = merged["Months_on_book"]
    merged["nb_relations_total"] = merged["Total_Relationship_Count"]
    merged["mois_inactifs_12m"] = merged["Months_Inactive_12_mon"]
    merged["nb_contacts_12m"] = merged["Contacts_Count_12_mon"]
    merged["limite_credit"] = merged["Credit_Limit"]
    merged["encours_renouvelable"] = merged["Total_Revolving_Bal"]
    merged["credit_disponible"] = merged["Avg_Open_To_Buy"]
    merged["variation_montant_t4_t1"] = merged["Total_Amt_Chng_Q4_Q1"]
    merged["montant_transactions_total"] = merged["Total_Trans_Amt"]
    merged["nb_transactions_total"] = merged["Total_Trans_Ct"]
    merged["variation_nombre_t4_t1"] = merged["Total_Ct_Chng_Q4_Q1"]
    merged["ratio_utilisation_moyen"] = merged["Avg_Utilization_Ratio"]
    merged["churn_observe"] = merged["churn_flag"]
    merged["score_churn"] = merged["churn_probability"]
    merged["alerte_seuil_recommande"] = merged["predicted_label_recommended"]
    merged["score_decile"] = merged["risk_decile"]
    merged["est_dans_test"] = merged["churn_reel_test"].notna().astype(int)
    merged["prediction_resultat"] = merged["prediction_resultat_brut"].map(VALUE_TRANSLATIONS["prediction_outcome"]).fillna("Hors échantillon de test")

    merged["genre"] = merged["Gender"].map(VALUE_TRANSLATIONS["Gender"]).fillna("Non renseigné")
    merged["niveau_etude"] = merged["Education_Level"].map(VALUE_TRANSLATIONS["Education_Level"]).fillna("Non renseigné")
    merged["statut_marital"] = merged["Marital_Status"].map(VALUE_TRANSLATIONS["Marital_Status"]).fillna("Non renseigné")
    merged["categorie_revenu"] = merged["Income_Category"].map(VALUE_TRANSLATIONS["Income_Category"]).fillna("Non renseigné")
    merged["categorie_carte"] = merged["Card_Category"].map(VALUE_TRANSLATIONS["Card_Category"]).fillna("Non renseigné")

    merged["tranche_age"] = pd.cut(
        merged["age_client"],
        bins=[25, 39, 49, 59, 80],
        labels=["26 à 39 ans", "40 à 49 ans", "50 à 59 ans", "60 ans et plus"],
        include_lowest=True,
    ).astype(str)
    merged["tranche_anciennete"] = pd.cut(
        merged["anciennete_mois"],
        bins=[0, 24, 36, 48, 100],
        labels=["Moins de 24 mois", "24 à 36 mois", "37 à 48 mois", "49 mois et plus"],
        include_lowest=True,
    ).astype(str)
    merged["profil_produit"] = np.where(merged["nb_relations_total"] == 1, "Mono-produit", "Multi-produit")
    merged["statut_activite"] = np.select(
        [
            (merged["mois_inactifs_12m"] >= 4) & (merged["nb_transactions_total"] < 50),
            (merged["mois_inactifs_12m"] >= 3),
            (merged["nb_transactions_total"] >= 80),
        ],
        [
            "Très faible activité",
            "En retrait",
            "Très actif",
        ],
        default="Activité intermédiaire",
    )
    merged["bande_risque"] = pd.cut(
        merged["score_churn"],
        bins=[-0.001, 0.15, threshold, 0.50, 1.0],
        labels=["Faible", "Moyen", "Élevé", "Très élevé"],
        include_lowest=True,
    ).astype(str)
    merged["segment_client"] = merged["cluster_label"].map(PERSONA_LABELS).fillna("Segment non disponible")
    merged["description_segment"] = merged["segment_client"].map(PERSONA_DESCRIPTIONS).fillna("")
    return merged


def _prepare_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    display = benchmark.copy()
    display["candidate_name"] = display["candidate_name"].map(MODEL_LABELS).fillna(display["candidate_name"])
    display["model_family"] = display["model_family"].map(MODEL_FAMILY_LABELS).fillna(display["model_family"])
    display["imbalance_strategy"] = display["imbalance_strategy"].map(IMBALANCE_LABELS).fillna(display["imbalance_strategy"])
    display = display.rename(
        columns={
            "candidate_name": "Modèle",
            "model_family": "Famille",
            "imbalance_strategy": "Rééquilibrage",
            "cv_best_pr_auc": "PR-AUC CV",
            "validation_pr_auc": "PR-AUC validation",
            "validation_recall": "Recall churn validation",
            "validation_precision": "Précision churn validation",
            "validation_f1": "F1 validation",
            "validation_f2": "F2 validation",
            "validation_roc_auc": "ROC-AUC validation",
            "best_params": "Paramètres retenus",
        }
    )
    return display


def _prepare_feature_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    display = feature_importance.copy()
    display["variable"] = display["feature"].map(FEATURE_LABELS).fillna(display["feature"])
    display["famille"] = display["variable"].map(FEATURE_GROUPS).fillna("Autres signaux")
    display = display.rename(columns={"importance": "importance_relative"})
    return display[["variable", "famille", "importance_relative"]]


def _prepare_slice_metrics(slice_metrics: pd.DataFrame) -> pd.DataFrame:
    display = slice_metrics.copy()
    dimension_labels = {
        "Gender": "Genre",
        "Income_Category": "Catégorie de revenu",
        "Card_Category": "Catégorie de carte",
        "age_band": "Tranche d'âge",
        "cluster_label": "Segment client",
        "risk_band": "Bande de risque",
    }
    display["dimension_label"] = display["dimension"].map(dimension_labels).fillna(display["dimension"])
    display["value_label"] = display.apply(_translate_slice_value, axis=1)
    return display


def _translate_slice_value(row: pd.Series) -> str:
    value = row["value"]
    dimension = row["dimension"]
    if dimension == "Gender":
        return VALUE_TRANSLATIONS["Gender"].get(value, value)
    if dimension == "Income_Category":
        return VALUE_TRANSLATIONS["Income_Category"].get(value, value)
    if dimension == "Card_Category":
        return VALUE_TRANSLATIONS["Card_Category"].get(value, value)
    if dimension == "cluster_label":
        return PERSONA_LABELS.get(value, value)
    return value


def _anonymize_client_id(client_id: object) -> str:
    digest = hashlib.sha1(str(client_id).encode("utf-8")).hexdigest()[:8].upper()
    return f"BC-{digest}"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
