from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import methodological_note, section_intro, section_transition, small_definition, takeaway
from dashboard.content import COLUMN_EXPLANATIONS, PERSONA_ACTIONS, PERSONA_DESCRIPTIONS, RISK_BAND_DESCRIPTIONS, UI_LABELS
from dashboard.data import (
    build_action_table,
    build_dimension_profile,
    build_executive_story,
    build_high_risk_comparison,
    build_local_explanation_table,
    build_persona_difference_table,
    build_segment_table,
    compute_filtered_metrics,
    get_filtered_test_subset,
    persona_differences,
    prepare_display_table,
    summarize_population,
)


PROFILE_DIMENSIONS = {
    "segment_client": "Segment client",
    "bande_risque": "Bande de risque",
    "categorie_carte": "Catégorie de carte",
    "categorie_revenu": "Catégorie de revenu",
    "tranche_age": "Tranche d'âge",
    "statut_activite": "Statut d'activité",
    "profil_produit": "Mono-produit / multi-produit",
    "niveau_etude": "Niveau d'étude",
}


def render_executive_overview(bundle, filtered: pd.DataFrame) -> None:
    st.header("Executive Overview")
    section_intro("Cette vue d'ensemble permet de comprendre, en un coup d'œil, où se concentre le risque sur le portefeuille actuellement affiché.")
    summary = summarize_population(filtered)
    story = build_executive_story(filtered)
    action_table = build_action_table(filtered)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients dans le périmètre", f"{summary['clients']:,}".replace(",", " "), help="Volume de clients correspondant aux filtres actifs.")
    col2.metric("Taux de churn observé", f"{summary['taux_churn']:.1%}", help="Part de churners observée dans la population filtrée.")
    col3.metric("Score moyen de churn", f"{summary['score_moyen']:.3f}", help="Risque moyen estimé par le modèle sur la population filtrée.")
    col4.metric("Part ciblée par le seuil", f"{summary['part_haut_risque']:.1%}", help="Part des clients que le modèle placerait en alerte avec le seuil recommandé.")

    with st.container(border=True):
        st.markdown("**Synthèse exécutive du périmètre affiché**")
        st.markdown(
            f"""
            - **Lecture d'ensemble** : {story['resume']}
            - **Signal de vigilance** : {story['signal']}
            - **Point d'attention métier** : {story['priorite']}
            """
        )

    with st.container(border=True):
        st.markdown("**Où se concentre le risque par segment client ?**")
        st.caption("Chaque barre représente un segment client. Sa longueur montre son poids dans le périmètre, tandis que les couleurs montrent la répartition entre risque faible, moyen, élevé et très élevé.")
        fig, segment_priority = _build_executive_overview_figure(filtered)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Lecture rapide des priorités**")
    st.caption("Le tableau ci-dessous réunit, en un seul endroit, la taille du segment, son niveau de risque et la logique d'action recommandée.")
    st.caption(
        "**Lecture des deux indicateurs de risque** : le **taux de churn observé** correspond au churn historiquement constaté dans le segment ; "
        "le **score moyen de churn** correspond au risque moyen prédit par le modèle pour les clients de ce segment."
    )
    if not action_table.empty:
        executive_table = (
            action_table.drop(columns=["description_segment"], errors="ignore").merge(
                segment_priority[
                    [
                        "segment_client",
                        "clients_haut_risque",
                        "part_haut_risque_segment_pct",
                        "description_segment",
                    ]
                ],
                on="segment_client",
                how="left",
            )
            .sort_values(["clients_haut_risque", "score_moyen", "taux_churn"], ascending=False)
            [
                [
                    "segment_client",
                    "triage_action",
                    "part_population",
                    "taux_churn",
                    "score_moyen",
                    "clients_haut_risque",
                    "part_haut_risque_segment_pct",
                    "hypothese_action",
                    "description_segment",
                ]
            ]
            .rename(
                columns={
                    "segment_client": "Segment client",
                    "triage_action": "Décision suggérée",
                    "part_population": "Part de la population (%)",
                    "taux_churn": "Taux de churn observé",
                    "score_moyen": "Score moyen de churn",
                    "clients_haut_risque": "Clients déjà à risque",
                    "part_haut_risque_segment_pct": "Part des déjà à risque dans le segment (%)",
                    "hypothese_action": "Hypothèse d'action",
                    "description_segment": "Description du segment",
                }
            )
        )
        executive_table["Part de la population (%)"] = executive_table["Part de la population (%)"] * 100
        st.dataframe(
            executive_table[
                [
                    "Segment client",
                    "Décision suggérée",
                    "Part de la population (%)",
                    "Taux de churn observé",
                    "Score moyen de churn",
                    "Clients déjà à risque",
                    "Part des déjà à risque dans le segment (%)",
                    "Hypothèse d'action",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Segment client": st.column_config.TextColumn(
                    "Segment client",
                    help="Nom métier du persona utilisé pour relire le portefeuille.",
                ),
                "Décision suggérée": st.column_config.TextColumn(
                    "Décision suggérée",
                    help="Niveau de priorité opérationnelle recommandé pour ce segment.",
                ),
                "Part de la population (%)": st.column_config.NumberColumn(
                    "Part de la population (%)",
                    help="Poids du segment dans le périmètre actuellement affiché.",
                    format="%.1f%%",
                ),
                "Taux de churn observé": st.column_config.NumberColumn(
                    "Taux de churn observé",
                    help="Part des clients de ce segment qui ont effectivement churné dans les données historiques.",
                    format="%.1f%%",
                ),
                "Score moyen de churn": st.column_config.NumberColumn(
                    "Score moyen de churn",
                    help="Risque moyen estimé par le modèle pour les clients de ce segment.",
                    format="%.3f",
                ),
                "Clients déjà à risque": st.column_config.ProgressColumn(
                    "Clients déjà à risque",
                    help="Nombre de clients déjà classés en risque élevé ou très élevé dans ce segment.",
                    min_value=0,
                    max_value=int(executive_table["Clients déjà à risque"].max()) if not executive_table.empty else 1,
                    format="%d",
                ),
                "Part des déjà à risque dans le segment (%)": st.column_config.NumberColumn(
                    "Part des déjà à risque dans le segment (%)",
                    help="Part du segment qui se situe déjà en risque élevé ou très élevé.",
                    format="%.1f%%",
                ),
                "Hypothèse d'action": st.column_config.TextColumn(
                    "Hypothèse d'action",
                    help="Piste d'action prudente issue de la lecture analytique du segment.",
                ),
            },
        )

        st.markdown("**Détail des segments**")
        st.caption("Ouvrez un segment pour lire sa description complète et la logique d'action avec plus de nuance.")
        for _, row in executive_table.iterrows():
            with st.expander(row["Segment client"], expanded=False):
                action_hypothesis = row["Hypothèse d'action"]
                st.markdown(f"**Description du segment** : {row['Description du segment']}")
                st.markdown(f"**Décision suggérée** : {row['Décision suggérée']}")
                st.markdown(f"**Hypothèse d'action** : {action_hypothesis}")
                st.markdown(
                    f"**Lecture rapide** : {row['Clients déjà à risque']:.0f} clients du segment sont déjà en risque élevé ou très élevé, "
                    f"soit {row['Part des déjà à risque dans le segment (%)']:.1f}% du segment."
                )


def _build_executive_overview_figure(filtered: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    risk_order = ["Faible", "Moyen", "Élevé", "Très élevé"]
    risk_colors = {
        "Faible": "#4F86C6",
        "Moyen": "#F4C95D",
        "Élevé": "#F28C28",
        "Très élevé": "#D7263D",
    }

    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucun client n'est disponible dans le périmètre affiché.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        empty = pd.DataFrame(
            columns=[
                "segment_client",
                "description_segment",
                "clients_haut_risque",
                "part_haut_risque_segment_pct",
            ]
        )
        return fig, empty

    segment_summary = (
        filtered.groupby("segment_client", as_index=False)
        .agg(
            clients_segment=("identifiant_client", "count"),
            churn_segment=("churn_observe", "mean"),
            score_moyen=("score_churn", "mean"),
        )
    )
    segment_band = (
        filtered.assign(bande_risque=pd.Categorical(filtered["bande_risque"], categories=risk_order, ordered=True))
        .groupby(["segment_client", "bande_risque"], observed=False)
        .agg(clients=("identifiant_client", "count"))
        .reset_index()
        .merge(segment_summary, on="segment_client", how="left")
    )
    segment_band["part_segment_pct"] = np.where(
        segment_band["clients_segment"] > 0,
        segment_band["clients"] / segment_band["clients_segment"] * 100,
        0.0,
    )

    segment_order = (
        segment_band[segment_band["bande_risque"].isin(["Élevé", "Très élevé"])]
        .groupby("segment_client", as_index=False)["clients"]
        .sum()
        .rename(columns={"clients": "clients_haut_risque"})
        .merge(segment_summary[["segment_client", "clients_segment", "churn_segment", "score_moyen"]], on="segment_client", how="right")
        .fillna({"clients_haut_risque": 0})
        .sort_values(["clients_haut_risque", "score_moyen", "clients_segment"], ascending=False)
    )
    ordered_segments = segment_order["segment_client"].tolist()
    segment_band["segment_client"] = pd.Categorical(segment_band["segment_client"], categories=ordered_segments, ordered=True)
    segment_band = segment_band.sort_values(["segment_client", "bande_risque"])

    fig = go.Figure()
    for risk_band in risk_order:
        subset = segment_band[segment_band["bande_risque"] == risk_band]
        fig.add_trace(
            go.Bar(
                y=subset["segment_client"],
                x=subset["clients"],
                orientation="h",
                name=risk_band,
                marker_color=risk_colors[risk_band],
                marker_line_width=0,
                customdata=np.column_stack(
                    [
                        subset["part_segment_pct"],
                        subset["clients_segment"],
                        subset["churn_segment"],
                        subset["score_moyen"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Bande de risque : " + risk_band + "<br>"
                    "Clients dans cette bande : %{x:,.0f}<br>"
                    "Part du segment : %{customdata[0]:.1f}%<br>"
                    "Taille totale du segment : %{customdata[1]:,.0f}<br>"
                    "Taux de churn observé du segment : %{customdata[2]:.1%}<br>"
                    "Score moyen de churn du segment : %{customdata[3]:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        legend_title_text="Bande de risque",
        xaxis_title="Nombre de clients",
        yaxis_title="Segment client",
        bargap=0.22,
        margin=dict(t=10, r=20, b=10, l=10),
    )
    segment_priority = segment_order.copy()
    segment_priority["part_haut_risque_segment_pct"] = np.where(
        segment_priority["clients_segment"] > 0,
        segment_priority["clients_haut_risque"] / segment_priority["clients_segment"] * 100,
        0.0,
    )
    segment_priority["description_segment"] = segment_priority["segment_client"].map(PERSONA_DESCRIPTIONS)
    return fig, segment_priority.reset_index(drop=True)


def render_customer_profiles(filtered: pd.DataFrame) -> None:
    st.header("Customer Profiles")
    section_intro("Cette section aide à comparer les profils de clientèle selon des dimensions métier simples et à voir où les comportements diffèrent le plus.")
    with st.container(border=True):
        st.markdown("**Par où commencer ?**")
        st.markdown(
            """
            - Commencez souvent par **Segment client** ou **Bande de risque** pour repérer les zones de tension.
            - Utilisez ensuite **Statut d'activité**, **Mono-produit / multi-produit** et **Catégorie de carte** pour affiner la lecture opérationnelle.
            - Recherchez les profils où le **taux de churn observé** et le **score moyen** sont simultanément élevés.
            """
        )

    selected_dimension = st.selectbox(
        "Dimension à comparer",
        options=list(PROFILE_DIMENSIONS.keys()),
        format_func=lambda key: PROFILE_DIMENSIONS[key],
    )
    st.caption("Les comparaisons ci-dessous sont recalculées sur le périmètre filtré.")

    profile_df = build_dimension_profile(filtered, selected_dimension)
    if profile_df.empty:
        st.warning("Aucune donnée disponible pour cette combinaison de filtres.")
        return

    st.markdown("**Taux de churn et score moyen par profil**")
    st.caption("Ce graphique aide à distinguer les groupes qui concentrent à la fois du churn observé et un score moyen élevé.")
    fig = px.bar(
        profile_df,
        x=selected_dimension,
        y=["taux_churn", "score_moyen"],
        barmode="group",
        labels={
            selected_dimension: PROFILE_DIMENSIONS[selected_dimension],
            "value": "Valeur",
            "variable": "Indicateur",
            "taux_churn": "Taux de churn observé",
            "score_moyen": "Score moyen de churn",
        },
    )
    fig.update_layout(legend_title_text="Indicateur", xaxis_title=PROFILE_DIMENSIONS[selected_dimension], yaxis_title="Valeur")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("Un profil peut afficher un taux de churn observé élevé, un score élevé, ou les deux. C'est ce croisement qui guide la priorisation.")

    st.markdown("**Comportement transactionnel moyen**")
    st.caption("On compare ici l'intensité de relation économique des groupes observés.")
    fig = px.bar(
        profile_df,
        x=selected_dimension,
        y=["transactions_moyennes", "montant_moyen"],
        barmode="group",
        labels={
            selected_dimension: PROFILE_DIMENSIONS[selected_dimension],
            "value": "Valeur moyenne",
            "variable": "Indicateur",
            "transactions_moyennes": "Nombre moyen de transactions",
            "montant_moyen": "Montant moyen des transactions",
        },
    )
    fig.update_layout(legend_title_text="Indicateur", xaxis_title=PROFILE_DIMENSIONS[selected_dimension], yaxis_title="Valeur moyenne")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("La lecture conjointe du volume et du montant transactionnel aide à différencier les clients réellement engagés des clients plus passifs.")

    st.markdown("**Table de lecture détaillée**")
    st.caption("Les colonnes sont volontairement exprimées en vocabulaire métier pour faciliter la lecture non technique.")
    display_df = profile_df.rename(
        columns={
            selected_dimension: PROFILE_DIMENSIONS[selected_dimension],
            "clients": "Clients",
            "taux_churn": "Taux de churn observé",
            "score_moyen": "Score moyen de churn",
            "montant_moyen": "Montant moyen des transactions",
            "transactions_moyennes": "Nombre moyen de transactions",
        }
    )
    st.dataframe(display_df, use_container_width=True)
    top_row = profile_df.sort_values(["taux_churn", "score_moyen"], ascending=False).iloc[0]
    takeaway(
        f"Le profil le plus exposé sur cette dimension est « {top_row[selected_dimension]} », avec un taux de churn observé de {top_row['taux_churn']:.1%} et un score moyen de {top_row['score_moyen']:.3f}."
    )


def render_churn_drivers(bundle, filtered: pd.DataFrame) -> None:
    st.header("Churn Drivers")
    section_intro("Le modèle retient surtout des signaux liés à l'activité transactionnelle, à l'engagement récent et à la profondeur de relation. Il faut les lire comme des indicateurs associés au churn, pas comme des causes prouvées.")

    importance_df = bundle.feature_importance.copy()
    st.markdown("**Variables les plus utilisées par le modèle**")
    st.caption("Cette vue montre les signaux qui comptent le plus dans le classement du risque.")
    fig = px.bar(
        importance_df.sort_values("importance_relative"),
        x="importance_relative",
        y="variable",
        color="famille",
        orientation="h",
        labels={
            "importance_relative": "Importance relative",
            "variable": "Variable",
            "famille": "Famille de signaux",
        },
    )
    fig.update_layout(legend_title_text="Famille de signaux", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("Les variables dominantes racontent surtout un désengagement d'usage : moins d'activité, plus d'inactivité, ou une relation bancaire moins profonde.")

    st.markdown("**Lecture par grande famille de signaux**")
    st.caption("Cette agrégation rend la hiérarchie des drivers plus compréhensible pour un public métier.")
    grouped = (
        importance_df.groupby("famille", as_index=False)["importance_relative"]
        .sum()
        .sort_values("importance_relative", ascending=True)
    )
    fig = px.bar(
        grouped,
        x="importance_relative",
        y="famille",
        orientation="h",
        labels={
            "importance_relative": "Importance cumulée",
            "famille": "Famille de signaux",
        },
        color="importance_relative",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    methodological_note("Un driver très prédictif peut n'être qu'un proxy. Par exemple, l'inactivité récente capture un retrait de l'usage, mais ne démontre pas à elle seule pourquoi le client part.")

    st.markdown("**Heatmap de corrélation sur le périmètre filtré**")
    st.caption("Cette heatmap se recalcule avec les filtres globaux du dashboard. Elle montre comment les indicateurs évoluent ensemble dans le sous-portefeuille actuellement observé.")
    correlation_candidates = [
        "churn_observe",
        "score_churn",
        "nb_transactions_total",
        "montant_transactions_total",
        "mois_inactifs_12m",
        "nb_contacts_12m",
        "nb_relations_total",
        "anciennete_mois",
        "limite_credit",
        "encours_renouvelable",
        "credit_disponible",
        "ratio_utilisation_moyen",
        "variation_nombre_t4_t1",
        "variation_montant_t4_t1",
    ]
    default_correlation_view = [
        "churn_observe",
        "score_churn",
        "nb_transactions_total",
        "montant_transactions_total",
        "mois_inactifs_12m",
        "nb_contacts_12m",
        "nb_relations_total",
        "ratio_utilisation_moyen",
    ]
    selected_correlation_columns = st.multiselect(
        "Indicateurs à afficher dans la heatmap",
        options=correlation_candidates,
        default=default_correlation_view,
        format_func=lambda column: UI_LABELS.get(column, column),
        help="La heatmap reflète toujours le périmètre filtré. Réduisez ou élargissez les variables pour faciliter la lecture.",
    )
    correlation_fig = _build_correlation_heatmap(filtered, selected_correlation_columns)
    if correlation_fig is None:
        methodological_note("Sélectionnez au moins deux indicateurs numériques pour afficher une heatmap de corrélation lisible.")
    else:
        st.plotly_chart(correlation_fig, use_container_width=True)
        takeaway("La heatmap aide à distinguer les variables qui évoluent ensemble de celles qui apportent une information plus indépendante. Elle ne prouve pas une relation causale.")

    with st.expander("Visualisations SHAP avancées", expanded=False):
        st.caption("Ces vues proviennent directement du pipeline d'explicabilité. Elles conservent certains libellés techniques du modèle et sont donc réservées à une lecture data plus experte.")
        for image_name, caption in [
            ("shap_summary.png", "Vue globale des contributions SHAP"),
            ("shap_bar.png", "Importance SHAP agrégée"),
        ]:
            image_path = bundle.figures_dir / image_name
            if image_path.exists():
                st.image(str(image_path), caption=caption, use_container_width=True)

    st.markdown("**Cas locaux pour interpréter le modèle sans le sur-vendre**")
    st.caption("Ces trois exemples concrets montrent comment lire une bonne détection, une fausse alerte et un churner manqué.")
    case_table = build_local_explanation_table(bundle.population, getattr(bundle, "local_explanations", {}))
    if not case_table.empty:
        st.dataframe(case_table, use_container_width=True, hide_index=True)
    case_meta = [
        ("true_positive", "Churner bien détecté", "shap_waterfall_true_positive.png"),
        ("false_positive", "Fausse alerte", "shap_waterfall_false_positive.png"),
        ("false_negative", "Churner manqué", "shap_waterfall_false_negative.png"),
    ]
    for case_key, title, image_name in case_meta:
        image_path = bundle.figures_dir / image_name
        row = case_table[case_table["Cas"] == title]
        if row.empty:
            continue
        descriptor = row.iloc[0]
        with st.expander(title, expanded=False):
            st.markdown(
                f"""
                - **Client** : {int(descriptor['Identifiant client'])}
                - **Segment client** : {descriptor['Segment client']}
                - **Bande de risque** : {descriptor['Bande de risque']}
                - **Score de churn** : {descriptor['Score de churn']:.3f}
                - **Lecture** : {descriptor['Issue de prédiction']}
                """
            )
            if image_path.exists():
                st.image(str(image_path), caption=f"Visualisation locale SHAP : {title}", use_container_width=True)
    methodological_note("Les cas locaux sont utiles pour comprendre comment le modèle raisonne sur quelques clients concrets. Ils ne doivent pas être extrapolés comme des règles universelles.")
    takeaway("Les vues traduites du dashboard doivent guider l'action. Les visuels SHAP détaillés restent disponibles en lecture complémentaire pour un public plus technique.")


def _build_correlation_heatmap(filtered: pd.DataFrame, selected_columns: list[str]) -> go.Figure | None:
    if len(selected_columns) < 2:
        return None

    correlation_frame = filtered[selected_columns].copy()
    if correlation_frame.dropna(how="all").shape[0] < 3:
        return None

    corr = correlation_frame.corr(numeric_only=True)
    if corr.shape[0] < 2:
        return None

    labels = [UI_LABELS.get(column, column) for column in corr.columns]
    corr.index = labels
    corr.columns = labels

    fig = px.imshow(
        corr.round(2),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels={"x": "Indicateurs", "y": "Indicateurs", "color": "Corrélation"},
    )
    fig.update_layout(
        margin=dict(t=20, r=20, b=20, l=20),
        coloraxis_colorbar_title="Corrélation",
    )
    return fig


def render_model_performance(bundle, filtered: pd.DataFrame) -> None:
    st.header("Model Performance")
    section_intro("Le modèle retenu doit servir un objectif de rétention : identifier un maximum de churners potentiels, même si cela implique quelques fausses alertes.")

    summary = bundle.model_summary
    validation = summary["validation_metrics"]
    test = summary["test_metrics"]
    threshold = summary["threshold_policy"]["threshold"]
    calibration = summary["calibration_decision"]

    st.markdown("**Comparaison des modèles candidats**")
    st.caption("Les modèles sont comparés sur la qualité de détection des churners, avec une attention particulière portée à la PR-AUC et au recall.")
    st.dataframe(bundle.benchmark, use_container_width=True)
    takeaway("Le modèle XGBoost pondéré est retenu car il domine les autres candidats en PR-AUC tout en conservant un recall très élevé, cohérent avec une logique de rétention proactive.")

    st.markdown("**Pourquoi ce modèle est privilégié**")
    st.markdown(
        f"""
        Le modèle final maximise la détection des churners sans s'effondrer en précision. Concrètement :

        - il atteint un **PR-AUC test de {test['pr_auc']:.3f}**, très élevé pour un problème déséquilibré ;
        - il conserve un **recall churn de {test['recall']:.1%}**, ce qui limite fortement les churners manqués ;
        - sa **précision churn de {test['precision']:.1%}** reste suffisamment élevée pour éviter une logique d'alerte trop large.

        Le seuil recommandé est fixé à **{threshold:.3f}**. Il a été choisi parce qu'il permet de capter presque tous les churners sur validation, tout en gardant une précision encore exploitable pour une équipe de rétention.
        """
    )

    st.markdown("**Lecture rapide des indicateurs clés**")
    st.markdown(
        """
        - **PR-AUC** : mesure la qualité de classement des churners dans un contexte où ils sont rares ; c'est l'indicateur principal ici.
        - **Recall churn** : part des churners détectés ; il est prioritaire car manquer un churner coûte cher en rétention.
        - **Précision churn** : part des alertes qui correspondent vraiment à un churn ; elle évite de sur-solliciter le portefeuille.
        - **ROC-AUC** : indicateur global de séparation, utile mais moins central que la PR-AUC sur un problème déséquilibré.
        """
    )

    global_metrics_cols = st.columns(4)
    global_metrics_cols[0].metric("PR-AUC test", f"{test['pr_auc']:.3f}", help="Qualité du classement des churners dans un contexte déséquilibré.")
    global_metrics_cols[1].metric("Recall churn test", f"{test['recall']:.1%}", help="Part des churners correctement détectés.")
    global_metrics_cols[2].metric("Précision churn test", f"{test['precision']:.1%}", help="Part des alertes qui correspondent réellement à un churn.")
    global_metrics_cols[3].metric("Seuil recommandé", f"{threshold:.3f}", help="Point de coupure utilisé pour passer du score à l'alerte opérationnelle.")

    st.markdown("**Choix méthodologiques à connaître**")
    method_cols = st.columns(3)
    method_cols[0].metric(
        "Stratégie Unknown retenue",
        "Conserver comme modalité",
        help="Le gain moyen de PR-AUC de l'alternative traitant Unknown comme manquant était insuffisant pour justifier un changement."
        if bundle.unknown_strategy["selected_strategy"] == "keep"
        else "La modalité Unknown a été traitée comme manquante car le gain moyen de PR-AUC était jugé suffisant.",
    )
    method_cols[1].metric(
        "Calibration retenue",
        "Probabilités brutes" if calibration["selected"] == "raw" else "Probabilités calibrées",
        help="La calibration n'a pas été retenue car le léger gain de Brier score s'accompagnait d'une petite baisse de PR-AUC."
        if calibration["selected"] == "raw"
        else "La calibration a été retenue car elle améliore suffisamment la qualité probabiliste sans dégrader la capacité de classement.",
    )
    method_cols[2].metric(
        "Objectif métier prioritaire",
        "Recall churn élevé",
        help="Le projet privilégie l'identification du plus grand nombre possible de churners, quitte à générer quelques fausses alertes supplémentaires.",
    )

    st.markdown("**Conséquences opérationnelles du seuil recommandé**")
    threshold_table = pd.DataFrame(
        [
            {
                "Lecture": "Churners bien détectés",
                "Volume test": int(test["tp"]),
                "Ce que cela signifie": "Clients à risque effectivement captés par la campagne de rétention.",
            },
            {
                "Lecture": "Fausses alertes",
                "Volume test": int(test["fp"]),
                "Ce que cela signifie": "Clients contactés alors qu'ils seraient restés ; coût commercial, mais moindre que des churners ratés dans cette stratégie.",
            },
            {
                "Lecture": "Churners manqués",
                "Volume test": int(test["fn"]),
                "Ce que cela signifie": "Clients perdus sans alerte ; c'est le cas que le seuil cherche d'abord à limiter.",
            },
            {
                "Lecture": "Clients stables correctement ignorés",
                "Volume test": int(test["tn"]),
                "Ce que cela signifie": "Clients laissés hors campagne à bon escient.",
            },
        ]
    )
    st.dataframe(threshold_table, use_container_width=True, hide_index=True)
    takeaway("Le seuil ne cherche pas à éliminer toutes les fausses alertes. Il privilégie d'abord la détection des churners, car l'enjeu métier est de réduire les départs manqués.")

    for image_name, caption in [
        ("validation_model_curves.png", "Courbes globales sur validation"),
        ("test_model_curves.png", "Courbes globales sur test"),
    ]:
        image_path = bundle.figures_dir / image_name
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)

    st.markdown("**Lecture du périmètre filtré sur l'échantillon de test**")
    st.caption("Cette lecture locale n'est affichée que si le sous-échantillon filtré contient suffisamment de cas et les deux classes.")
    filtered_test, warning_message = get_filtered_test_subset(filtered)
    if warning_message:
        methodological_note(warning_message)
    else:
        metrics = compute_filtered_metrics(filtered_test)
        local_cols = st.columns(4)
        local_cols[0].metric("Clients test dans le périmètre", metrics["n"])
        local_cols[1].metric("Recall churn local", f"{metrics['recall']:.1%}")
        local_cols[2].metric("Précision churn locale", f"{metrics['precision']:.1%}")
        local_cols[3].metric("PR-AUC locale", f"{metrics['pr_auc']:.3f}")

        matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Prédit stable", "Prédit churn"],
                y=["Réel stable", "Réel churn"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig.update_layout(title="Matrice de confusion sur le sous-échantillon filtré", xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        takeaway("Cette matrice locale permet de vérifier si la logique d'alerte reste cohérente pour le sous-ensemble actuellement analysé.")


def render_risk_scoring(filtered: pd.DataFrame) -> None:
    st.header("Risk Scoring")
    section_intro("Le score de churn est une probabilité estimée. Il permet de hiérarchiser les clients à surveiller, mais il ne doit jamais être interprété comme une certitude individuelle.")
    small_definition("Score de churn", "score_churn", COLUMN_EXPLANATIONS)

    st.markdown("**Répartition des bandes de risque**")
    st.caption("Cette vue simplifie la lecture du score en quatre niveaux de risque.")
    band_counts = (
        filtered.groupby("bande_risque")
        .agg(clients=("identifiant_client", "count"), score_moyen=("score_churn", "mean"))
        .reset_index()
    )
    fig = px.bar(
        band_counts,
        x="bande_risque",
        y="clients",
        color="bande_risque",
        labels={"bande_risque": "Bande de risque", "clients": "Nombre de clients"},
        text="score_moyen",
    )
    fig.update_traces(texttemplate="score moyen %{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    takeaway("Les bandes de risque servent à transformer un score continu en priorités d'action plus lisibles pour les équipes métier.")

    st.markdown("**Comment lire les bandes de risque**")
    for band, description in RISK_BAND_DESCRIPTIONS.items():
        st.markdown(f"**{band}** : {description}")

    high_risk_only = st.toggle("Afficher uniquement les clients en risque élevé ou très élevé", value=False)
    working_df = filtered[filtered["bande_risque"].isin(["Élevé", "Très élevé"])] if high_risk_only else filtered

    compare_dimension = st.selectbox(
        "Dimension de lecture des clients à risque",
        options=list(PROFILE_DIMENSIONS.keys()),
        format_func=lambda key: PROFILE_DIMENSIONS[key],
        key="risk_dimension",
    )
    dimension_profile = build_dimension_profile(working_df, compare_dimension)
    st.markdown("**Composition des clients affichés**")
    st.caption("Cette vue aide à repérer quels profils dominent parmi les clients retenus dans le périmètre courant.")
    fig = px.bar(
        dimension_profile,
        x=compare_dimension,
        y="clients",
        color="bande_risque" if compare_dimension == "bande_risque" else None,
        labels={compare_dimension: PROFILE_DIMENSIONS[compare_dimension], "clients": "Nombre de clients"},
    )
    st.plotly_chart(fig, use_container_width=True)
    takeaway("Le score aide à prioriser. Il ne remplace pas une validation métier ni une décision commerciale contextualisée.")

    st.markdown("**Qui compose le haut du portefeuille à risque ?**")
    high_risk = filtered[filtered["bande_risque"].isin(["Élevé", "Très élevé"])]
    if high_risk.empty:
        methodological_note("Le périmètre filtré ne contient actuellement aucun client dans les bandes de risque élevé ou très élevé.")
    else:
        high_risk_segments = (
            high_risk.groupby("segment_client")
            .agg(clients=("identifiant_client", "count"), score_moyen=("score_churn", "mean"))
            .reset_index()
            .sort_values("clients", ascending=False)
        )
        st.caption("Cette vue isole les personas qui dominent parmi les clients déjà au-dessus du niveau de vigilance renforcée.")
        fig = px.bar(
            high_risk_segments,
            x="segment_client",
            y="clients",
            color="score_moyen",
            color_continuous_scale="OrRd",
            labels={"segment_client": "Segment client", "clients": "Clients", "score_moyen": "Score moyen de churn"},
        )
        fig.update_layout(coloraxis_colorbar_title="Score moyen")
        st.plotly_chart(fig, use_container_width=True)

        comparison = build_high_risk_comparison(filtered)
        if not comparison.empty:
            comparison_display = comparison.copy()
            comparison_display["Écart relatif"] = comparison_display["Écart relatif"].map(lambda value: f"{value:+.1%}")
            st.caption("Le tableau ci-dessous compare les clients à risque élevé au reste du portefeuille actuellement affiché.")
            st.dataframe(comparison_display, use_container_width=True, hide_index=True)
        takeaway("Les clients à risque élevé se lisent mieux quand on distingue leur poids, leur persona dominant et leurs écarts comportementaux par rapport au reste du portefeuille.")

    st.markdown("**Table de clients à investiguer**")
    st.caption("La table reste triable et permet une lecture opérationnelle rapide avec un vocabulaire non technique.")
    display_table = prepare_display_table(working_df.sort_values("score_churn", ascending=False).head(100))
    st.dataframe(display_table, use_container_width=True)


def render_customer_segmentation(bundle, filtered: pd.DataFrame) -> None:
    st.header("Customer Segmentation")
    section_intro("Le projet n'expose pas de clusters prétendument robustes. Il présente des personas de risque compréhensibles, parce que la qualité interne des clusters initiaux s'est révélée trop faible pour une lecture forte.")
    methodological_note(
        f"La segmentation statistique d'origine est jugée faible (silhouette = {bundle.segmentation_summary['best_silhouette']:.3f}). Le dashboard privilégie donc des personas de risque plus honnêtes et plus actionnables."
    )

    segment_table = build_segment_table(filtered)
    if segment_table.empty:
        st.warning("Aucun segment disponible pour ce périmètre.")
        return

    st.markdown("**Vue d'ensemble des personas**")
    st.caption("Chaque persona combine une lecture du risque, du churn observé et de quelques traits comportementaux clés.")
    fig = px.bar(
        segment_table,
        x="segment_client",
        y=["taux_churn", "score_moyen"],
        barmode="group",
        labels={
            "segment_client": "Segment client",
            "value": "Valeur",
            "variable": "Indicateur",
            "taux_churn": "Taux de churn observé",
            "score_moyen": "Score moyen de churn",
        },
    )
    fig.update_layout(legend_title_text="Indicateur")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("Les personas les plus prioritaires sont ceux qui cumulent un score moyen élevé, un taux de churn observé fort et un poids non négligeable dans la population.")

    st.markdown("**Lecture détaillée des personas**")
    segment_display = segment_table.rename(
        columns={
            "segment_client": "Segment client",
            "description_segment": "Description",
            "clients": "Clients",
            "part_population": "Part de population",
            "taux_churn": "Taux de churn observé",
            "score_moyen": "Score moyen de churn",
            "transactions_moyennes": "Transactions moyennes",
            "contacts_moyens": "Contacts moyens",
            "mois_inactifs_moyens": "Mois inactifs moyens",
            "relations_moyennes": "Relations moyennes",
        }
    )
    st.dataframe(
        segment_display,
        column_config={
            "Clients": st.column_config.ProgressColumn(
                "Clients",
                help="Barre visuelle pour comparer la taille des personas.",
                min_value=0,
                max_value=int(segment_display["Clients"].max()) if not segment_display.empty else 1,
                format="%d",
            )
        },
        use_container_width=True,
    )

    segments_available = segment_table["segment_client"].tolist()
    default_left = segments_available[0]
    default_right = segments_available[1] if len(segments_available) > 1 else segments_available[0]
    left_segment, right_segment = st.columns(2)
    selected_left = left_segment.selectbox("Segment de gauche", options=segments_available, index=segments_available.index(default_left))
    selected_right = right_segment.selectbox("Segment de droite", options=segments_available, index=segments_available.index(default_right), key="right_segment")

    compare_cols = st.columns(2)
    for column, selected in zip(compare_cols, [selected_left, selected_right]):
        subset = filtered[filtered["segment_client"] == selected]
        stats = summarize_population(subset)
        column.markdown(f"**{selected}**")
        column.caption(PERSONA_DESCRIPTIONS.get(selected, ""))
        metric_cols = column.columns(3)
        metric_cols[0].metric("Clients", f"{stats['clients']:,}".replace(",", " "))
        metric_cols[1].metric("Taux de churn", f"{stats['taux_churn']:.1%}")
        metric_cols[2].metric("Score moyen", f"{stats['score_moyen']:.3f}")
        insights = persona_differences(subset, filtered)
        if insights:
            for insight in insights:
                column.markdown(f"- {insight}")
        diff_table = build_persona_difference_table(subset, filtered)
        if not diff_table.empty:
            diff_table["Écart relatif"] = diff_table["Écart relatif"].map(lambda value: f"{value:+.1%}" if pd.notna(value) else "n.d.")
            column.caption("Ce tableau compare le persona au reste du périmètre filtré.")
            column.dataframe(diff_table, use_container_width=True, hide_index=True)


def render_business_actions(filtered: pd.DataFrame) -> None:
    st.header("Business Actions")
    section_intro("Les recommandations ci-dessous traduisent les constats analytiques en hypothèses d'action. Elles restent prudentes et ne doivent pas être lues comme des preuves causales.")

    action_table = build_action_table(filtered)
    if action_table.empty:
        st.warning("Aucune recommandation ne peut être formulée sur un périmètre vide.")
        return

    st.markdown("**Segments à prioriser dans le périmètre affiché**")
    st.caption("La priorité est construite à partir du score moyen, du churn observé et du poids du segment dans la population filtrée.")
    st.dataframe(
        action_table[
            [
                "segment_client",
                "part_population",
                "taux_churn",
                "score_moyen",
                "triage_action",
                "priorite",
                "logique_priorisation",
                "hypothese_action",
            ]
        ].rename(
            columns={
                "segment_client": "Segment client",
                "part_population": "Part de population",
                "taux_churn": "Taux de churn observé",
                "score_moyen": "Score moyen de churn",
                "triage_action": "Décision suggérée",
                "priorite": "Priorité",
                "logique_priorisation": "Pourquoi maintenant",
                "hypothese_action": "Hypothèse d'action",
            }
        ),
        use_container_width=True,
    )

    st.markdown("**Triage recommandé sur le périmètre courant**")
    triage_order = ["Agir maintenant", "Cibler rapidement", "Surveiller de près", "Veille légère"]
    for triage_label in triage_order:
        subset = action_table[action_table["triage_action"] == triage_label]
        if subset.empty:
            continue
        with st.container(border=True):
            st.markdown(f"**{triage_label}**")
            st.caption("Cette recommandation est formulée à partir du score moyen de churn, du churn observé et du poids des personas dans le périmètre affiché.")
            for _, row in subset.sort_values(["score_moyen", "taux_churn"], ascending=False).iterrows():
                st.markdown(
                    f"""
                    - **{row['segment_client']}** : {row['logique_priorisation']}
                      Hypothèse d'action : {PERSONA_ACTIONS.get(row['segment_client'], row['hypothese_action'])}
                    """
                )

    st.markdown("**Signaux à monitorer**")
    st.markdown(
        """
        - hausse des mois d'inactivité ;
        - baisse récente du nombre ou du montant des transactions ;
        - multiplication des contacts sans amélioration visible du risque ;
        - profils mono-produit peu engagés.
        """
    )

    st.markdown("**Ce que le dashboard aide à décider, et ce qu'il ne permet pas de conclure**")
    st.markdown(
        """
        - Le dashboard aide à **prioriser** des clients et des personas à investiguer ou à contacter.
        - Il aide aussi à **adapter le type d'approche** : réactivation rapide, veille active, ou contact plus qualitatif.
        - En revanche, il ne démontre pas la **cause** du churn et ne remplace ni le jugement métier ni des tests d'impact réels.
        """
    )

    methodological_note(
        "Le dashboard n'identifie pas des causes certaines du churn. Il suggère des hypothèses d'action à tester, puis à confronter à l'expérience métier et, si possible, à des expérimentations contrôlées."
    )
