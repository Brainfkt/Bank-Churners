from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import card_title, metric_card, page_header, takeaway
from dashboard.data import (
    build_campaign_editor_seed,
    build_customer_explanation,
    build_streamlit_capability_table,
    build_work_queue,
    find_customer_by_reference,
    summarize_population,
    threshold_row,
    to_csv_bytes,
)
from dashboard.sections import (
    render_business_actions,
    render_churn_drivers,
    render_customer_profiles,
    render_customer_segmentation,
    render_data_artifact_health,
    render_decision_lab,
    render_model_performance,
    render_model_reliability,
    render_risk_scoring,
)


RISK_COLORS = {
    "Faible": "#20B2A6",
    "Moyen": "#F59E0B",
    "Élevé": "#FB7A4F",
    "Très élevé": "#FB5A4F",
}


def render_decision_hub(bundle: Any, filtered: pd.DataFrame) -> None:
    page_header(
        "Bank Churners Decision Hub",
        "Vue courte pour prioriser les clients à risque, ajuster le seuil opérationnel et ouvrir une explication individuelle sans quitter le workflow.",
        status=_artifact_status_label(bundle),
    )
    _render_kpi_strip(filtered, bundle.population)

    main_left, main_right = st.columns([1.18, 1], gap="large")
    with main_left:
        card_title(
            "Concentration du risque par persona",
            "Sélectionnez une barre pour orienter la file d'actions vers un persona spécifique.",
        )
        selected_segment = _render_risk_concentration_chart(filtered)
    with main_right:
        _render_decision_lab_panel(bundle, filtered)

    if selected_segment:
        st.session_state["filter_segment_client"] = [selected_segment]
        st.toast(f"Filtre persona appliqué : {selected_segment}")
        st.rerun()

    _render_action_queue(bundle, filtered)


def render_customer_explorer(bundle: Any, filtered: pd.DataFrame) -> None:
    page_header(
        "Customer Explorer",
        "Exploration des profils client, des personas et des signaux associés au churn. Les constats restent descriptifs et ne doivent pas être lus comme des preuves causales.",
    )
    profile_tab, segment_tab, drivers_tab = st.tabs(["Profils", "Personas", "Drivers"])
    with profile_tab:
        render_customer_profiles(filtered)
    with segment_tab:
        render_customer_segmentation(bundle, filtered)
    with drivers_tab:
        render_churn_drivers(bundle, filtered)


def render_model_studio(bundle: Any, filtered: pd.DataFrame) -> None:
    page_header(
        "Model Studio",
        "Lecture data du modèle : performance, stabilité, calibration et explicabilité. L'objectif est de rendre le score utile sans sur-vendre sa certitude.",
    )
    performance_tab, reliability_tab, explanation_tab = st.tabs(["Performance", "Reliability", "Explainability"])
    with performance_tab:
        render_model_performance(bundle, filtered)
    with reliability_tab:
        render_model_reliability(bundle, filtered)
    with explanation_tab:
        render_churn_drivers(bundle, filtered)


def render_retention_workbench(bundle: Any, filtered: pd.DataFrame) -> None:
    page_header(
        "Retention Workbench",
        "Espace opérationnel pour tester le seuil, préparer une liste de travail et cadrer un plan de contact prudent.",
    )
    threshold_tab, scoring_tab, campaign_tab = st.tabs(["Seuil", "Scoring", "Plan de contact"])
    with threshold_tab:
        render_decision_lab(bundle, filtered)
    with scoring_tab:
        render_risk_scoring(filtered)
    with campaign_tab:
        _render_campaign_editor(filtered)
        render_business_actions(filtered)


def render_data_health(bundle: Any, filtered: pd.DataFrame) -> None:
    page_header(
        "Data Health",
        "Contrats de reproductibilité, fraîcheur des artefacts et inventaire des capacités Streamlit utilisées dans l'interface.",
    )
    state = "error" if bundle.run_manifest.get("git", {}).get("dirty") else "complete"
    with st.status("Contrôle des artefacts du dernier run", expanded=False, state=state):
        st.write(f"Run généré : {bundle.run_manifest.get('generated_at_utc', 'n.d.')}")
        st.write(f"Modèle : {bundle.run_manifest.get('model', {}).get('selected_model', 'n.d.')}")
        st.write("Working tree dirty : " + str(bundle.run_manifest.get("git", {}).get("dirty", "n.d.")))

    health_tab, streamlit_tab = st.tabs(["Artefacts", "Streamlit capabilities"])
    with health_tab:
        render_data_artifact_health(bundle, filtered)
    with streamlit_tab:
        st.dataframe(build_streamlit_capability_table(), width="stretch", hide_index=True)
        takeaway(
            "Les composants Streamlit sont intégrés aux usages métier : filtrer, simuler, sélectionner, expliquer, exporter et vérifier la fraîcheur des artefacts."
        )


def _render_kpi_strip(filtered: pd.DataFrame, baseline: pd.DataFrame) -> None:
    summary = summarize_population(filtered)
    baseline_summary = summarize_population(baseline)
    cols = st.columns(4)
    with cols[0]:
        metric_card(
            "Clients",
            f"{summary['clients']:,}".replace(",", " "),
            "Volume de clients correspondant aux filtres actifs.",
            delta=_relative_delta(summary["clients"], baseline_summary["clients"]),
            delta_color="off",
        )
    with cols[1]:
        metric_card(
            "Taux de churn",
            f"{summary['taux_churn']:.1%}",
            "Part de churners observée dans le périmètre visible.",
            delta=_point_delta(summary["taux_churn"], baseline_summary["taux_churn"]),
            delta_color="inverse",
        )
    with cols[2]:
        metric_card(
            "Score moyen",
            f"{summary['score_moyen']:.3f}",
            "Risque moyen estimé par le modèle.",
            delta=f"{summary['score_moyen'] - baseline_summary['score_moyen']:+.3f}",
            delta_color="inverse",
        )
    with cols[3]:
        metric_card(
            "Clients ciblés",
            f"{summary['part_haut_risque']:.1%}",
            "Part du périmètre au-dessus du seuil recommandé.",
            delta=_point_delta(summary["part_haut_risque"], baseline_summary["part_haut_risque"]),
            delta_color="inverse",
        )


def _render_risk_concentration_chart(filtered: pd.DataFrame) -> str | None:
    mode = st.segmented_control(
        "Mode de lecture",
        options=["Risque", "Churn", "Volume"],
        default="Risque",
        key="risk_chart_mode",
    )
    risk_order = ["Très élevé", "Élevé", "Moyen", "Faible"]

    if mode == "Risque":
        grouped = (
            filtered.groupby(["segment_client", "bande_risque"])
            .size()
            .rename("clients")
            .reset_index()
        )
        if grouped.empty:
            st.info("Aucun client disponible pour ce périmètre.")
            return None
        totals = grouped.groupby("segment_client")["clients"].transform("sum")
        grouped["share"] = grouped["clients"] / totals
        fig = px.bar(
            grouped,
            y="segment_client",
            x="share",
            color="bande_risque",
            orientation="h",
            category_orders={"bande_risque": risk_order},
            color_discrete_map=RISK_COLORS,
            labels={"segment_client": "Persona", "share": "Part du persona", "bande_risque": "Bande"},
            custom_data=["segment_client", "bande_risque", "clients"],
        )
        fig.update_layout(barmode="stack", xaxis_tickformat=".0%", legend_title_text="Bande de risque")
        fig.update_traces(
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} : %{customdata[2]} clients<br>Part : %{x:.1%}<extra></extra>"
        )
    else:
        metric = "churn_observe" if mode == "Churn" else "identifiant_client"
        y_label = "Taux de churn observé" if mode == "Churn" else "Clients"
        grouped = (
            filtered.groupby("segment_client")
            .agg(value=(metric, "mean" if mode == "Churn" else "count"))
            .reset_index()
            .sort_values("value", ascending=True)
        )
        if grouped.empty:
            st.info("Aucun client disponible pour ce périmètre.")
            return None
        fig = px.bar(
            grouped,
            y="segment_client",
            x="value",
            orientation="h",
            color_discrete_sequence=["#2563EB"],
            labels={"segment_client": "Persona", "value": y_label},
            custom_data=["segment_client"],
        )
        if mode == "Churn":
            fig.update_layout(xaxis_tickformat=".0%")
            fig.update_traces(hovertemplate="%{customdata[0]}<br>Taux de churn : %{x:.1%}<extra></extra>")
        else:
            fig.update_traces(hovertemplate="%{customdata[0]}<br>Clients : %{x}<extra></extra>")
    fig.update_layout(
        margin=dict(t=8, r=12, b=24, l=8),
        height=330,
    )
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="risk_persona_chart")
    points = _extract_selection_points(event)
    if points:
        customdata = points[0].get("customdata") or []
        if customdata:
            return str(customdata[0])
    return None


def _render_decision_lab_panel(bundle: Any, filtered: pd.DataFrame) -> None:
    card_title(
        "Decision Lab",
        "Le seuil transforme un score en liste d'action. Le simulateur montre le compromis attendu, pas l'effet causal d'une campagne.",
    )
    recommended = float(bundle.model_summary["threshold_policy"]["threshold"])
    st.session_state.setdefault("decision_threshold", recommended)
    threshold = st.slider(
        "Seuil de score (probabilité de churn)",
        min_value=0.05,
        max_value=0.95,
        step=0.005,
        format="%.3f",
        key="decision_threshold",
    )
    selected_row = threshold_row(bundle.threshold_sensitivity, threshold)
    targeted = filtered[filtered["score_churn"] >= threshold]
    targeted_rate = len(targeted) / len(filtered) if len(filtered) else 0.0

    cols = st.columns(3)
    cols[0].metric("Clients ciblés", f"{len(targeted):,}".replace(",", " "), f"{targeted_rate:.1%}", border=True)
    cols[1].metric("Recall test", f"{selected_row['recall']:.1%}", border=True)
    cols[2].metric("Précision test", f"{selected_row['precision']:.1%}", border=True)

    sensitivity = bundle.threshold_sensitivity
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensitivity["threshold"], y=sensitivity["recall"], mode="lines", name="Recall", line=dict(color="#FB5A4F")))
    fig.add_trace(go.Scatter(x=sensitivity["threshold"], y=sensitivity["precision"], mode="lines", name="Précision", line=dict(color="#20B2A6")))
    fig.add_trace(go.Scatter(x=sensitivity["threshold"], y=sensitivity["targeted_rate"], mode="lines", name="Part ciblée", line=dict(color="#2563EB")))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#2563EB")
    fig.update_layout(
        height=240,
        yaxis_tickformat=".0%",
        xaxis_title="Seuil",
        yaxis_title="Valeur",
        legend_title_text="Indicateur",
        margin=dict(t=10, r=12, b=20, l=8),
    )
    st.plotly_chart(fig, width="stretch", key="decision_lab_threshold_curve")
    if st.button("Appliquer ce seuil", width="stretch", key="apply_decision_threshold"):
        st.toast(f"Seuil appliqué : {threshold:.3f}")


def _render_action_queue(bundle: Any, filtered: pd.DataFrame) -> None:
    threshold = float(st.session_state.get("decision_threshold", bundle.model_summary["threshold_policy"]["threshold"]))
    queue = build_work_queue(filtered, threshold, limit=250)
    st.markdown("### File d'actions - clients sélectionnés")
    st.caption("Sélectionnez une ligne pour ouvrir une explication locale. L'export reprend uniquement la file visible.")
    table_state = st.dataframe(
        queue,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="action_queue_table",
        column_config={
            "Score": st.column_config.NumberColumn("Score", format="%.3f"),
            "Bande": st.column_config.TextColumn("Bande"),
            "Action": st.column_config.TextColumn("Action"),
        },
    )
    selected_reference = _selected_reference_from_table(table_state, queue)
    if selected_reference:
        st.session_state["selected_client_ref"] = selected_reference

    actions = st.columns([1, 1, 3])
    with actions[0]:
        st.download_button(
            "Exporter CSV",
            data=to_csv_bytes(queue),
            file_name=f"bank_churners_action_queue_{threshold:.3f}.csv",
            mime="text/csv",
            width="stretch",
        )
    with actions[1]:
        if st.button("Expliquer", width="stretch", disabled=not st.session_state.get("selected_client_ref")):
            selected = find_customer_by_reference(filtered, st.session_state.get("selected_client_ref"))
            _customer_explanation_dialog(build_customer_explanation(selected))


def _render_campaign_editor(filtered: pd.DataFrame) -> None:
    card_title(
        "Simulation prudente du plan de contact",
        "Ajustez les plafonds de clients par persona. Cette simulation organise le ciblage, elle ne mesure pas un uplift causal.",
    )
    seed = build_campaign_editor_seed(filtered)
    edited = st.data_editor(
        seed,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "Cap clients": st.column_config.NumberColumn("Cap clients", min_value=0, max_value=2000, step=25),
            "Action": st.column_config.SelectboxColumn(
                "Action",
                options=["Agir maintenant", "Cibler rapidement", "Surveiller de près", "Veille légère"],
            ),
        },
        key="campaign_plan_editor",
    )
    st.download_button(
        "Télécharger le plan simulé",
        data=to_csv_bytes(edited),
        file_name="bank_churners_plan_contact_simule.csv",
        mime="text/csv",
        width="stretch",
    )


@st.dialog("Explication locale du risque")
def _customer_explanation_dialog(payload: dict[str, str]) -> None:
    st.markdown(f"**Référence client** : {payload['reference']}")
    st.markdown(payload["risk"])
    st.markdown(payload["signals"])
    st.info("Action prudente : " + payload["action"])
    st.caption("Cette lecture explique une association prédictive locale. Elle ne prouve pas une cause de churn.")


def _selected_reference_from_table(table_state: Any, queue: pd.DataFrame) -> str | None:
    rows: list[int] = []
    try:
        rows = list(table_state.selection.rows)
    except AttributeError:
        try:
            rows = list(table_state["selection"]["rows"])
        except (KeyError, TypeError):
            rows = []
    if not rows:
        return None
    row_index = rows[0]
    if row_index >= len(queue):
        return None
    return str(queue.iloc[row_index]["Référence client"])


def _extract_selection_points(event: Any) -> list[dict[str, Any]]:
    try:
        return list(event.selection.points)
    except AttributeError:
        try:
            return list(event["selection"]["points"])
        except (KeyError, TypeError):
            return []


def _artifact_status_label(bundle: Any) -> str:
    manifest = bundle.run_manifest
    if manifest.get("git", {}).get("dirty"):
        return "Artefacts générés sur working tree dirty"
    return "Données à jour"


def _relative_delta(value: float, baseline: float) -> str:
    if baseline == 0:
        return "n.d."
    return f"{(value - baseline) / baseline:+.1%} vs portefeuille"


def _point_delta(value: float, baseline: float) -> str:
    return f"{(value - baseline) * 100:+.1f} pt vs portefeuille"
