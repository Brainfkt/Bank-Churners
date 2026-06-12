from pathlib import Path

from dashboard.data import (
    apply_filters,
    available_filter_options,
    build_artifact_health_table,
    build_action_table,
    build_customer_explanation,
    build_streamlit_capability_table,
    build_work_queue,
    build_executive_story,
    build_high_risk_comparison,
    build_local_explanation_table,
    build_persona_difference_table,
    compute_filtered_metrics,
    find_customer_by_reference,
    get_filtered_test_subset,
    load_dashboard_bundle,
    threshold_row,
)
from dashboard.state import parse_query_state, serialize_query_state


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_bundle_contains_business_columns():
    bundle = load_dashboard_bundle(ROOT)
    required_columns = {
        "genre",
        "tranche_age",
        "categorie_revenu",
        "statut_marital",
        "niveau_etude",
        "categorie_carte",
        "tranche_anciennete",
        "profil_produit",
        "statut_activite",
        "bande_risque",
        "segment_client",
        "score_churn",
    }
    assert required_columns.issubset(bundle.population.columns)
    assert bundle.population["segment_client"].nunique() >= 3
    assert set(bundle.local_explanations) >= {"true_positive", "false_positive", "false_negative"}
    assert "clientnum_source" in bundle.population.columns
    assert bundle.population["identifiant_client"].astype(str).str.startswith("BC-").all()
    assert not (bundle.population["identifiant_client"].astype(str) == bundle.population["clientnum_source"].astype(str)).any()


def test_dashboard_bundle_contains_v2_reliability_artifacts():
    bundle = load_dashboard_bundle(ROOT)
    assert not bundle.threshold_sensitivity.empty
    assert not bundle.calibration_table.empty
    assert not bundle.slice_metrics.empty
    assert bundle.model_stability["test"]["pr_auc"] > 0
    assert bundle.run_manifest["model"]["selected_model"]

    row = threshold_row(bundle.threshold_sensitivity, bundle.model_summary["threshold_policy"]["threshold"])
    assert 0 <= row["targeted_rate"] <= 1

    artifact_table = build_artifact_health_table(ROOT, bundle.run_manifest)
    assert not artifact_table.empty
    assert artifact_table["Présent"].all()


def test_filter_options_are_translated_and_filtering_works():
    bundle = load_dashboard_bundle(ROOT)
    options = available_filter_options(bundle.population)
    assert "Femme" in options["genre"]
    filtered = apply_filters(bundle.population, {"genre": ["Femme"], "bande_risque": ["Très élevé"]})
    assert not filtered.empty
    assert set(filtered["genre"].unique()) == {"Femme"}
    assert set(filtered["bande_risque"].unique()) == {"Très élevé"}


def test_filtered_metrics_guard_and_metrics_are_valid():
    bundle = load_dashboard_bundle(ROOT)
    subset, warning = get_filtered_test_subset(bundle.population[bundle.population["segment_client"] == "Actifs à risque contenu"])
    if warning is None:
        metrics = compute_filtered_metrics(subset)
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert metrics["n"] == len(subset)


def test_executive_story_and_action_tables_are_populated():
    bundle = load_dashboard_bundle(ROOT)
    filtered = bundle.population[bundle.population["bande_risque"].isin(["Élevé", "Très élevé"])]
    story = build_executive_story(filtered)
    assert "resume" in story and story["resume"]
    assert "signal" in story and story["signal"]

    action_table = build_action_table(bundle.population)
    assert not action_table.empty
    assert set(action_table["triage_action"].unique()).issubset(
        {"Agir maintenant", "Cibler rapidement", "Surveiller de près", "Veille légère"}
    )


def test_local_explanations_and_comparison_tables_are_available():
    bundle = load_dashboard_bundle(ROOT)
    local_cases = build_local_explanation_table(bundle.population, bundle.local_explanations)
    assert not local_cases.empty
    assert {"Cas", "Référence client", "Segment client", "Issue de prédiction"}.issubset(local_cases.columns)
    assert local_cases["Référence client"].astype(str).str.startswith("BC-").all()

    persona_name = bundle.population["segment_client"].mode().iloc[0]
    persona_frame = bundle.population[bundle.population["segment_client"] == persona_name]
    diff_table = build_persona_difference_table(persona_frame, bundle.population)
    assert not diff_table.empty
    assert {"Indicateur", "Persona", "Reste du périmètre", "Écart relatif"} == set(diff_table.columns)

    risk_comparison = build_high_risk_comparison(bundle.population)
    assert not risk_comparison.empty
    assert {"Indicateur", "Clients à risque élevé", "Reste du portefeuille", "Écart relatif"} == set(risk_comparison.columns)


def test_v3_work_queue_and_customer_explanation_are_safe():
    bundle = load_dashboard_bundle(ROOT)
    threshold = bundle.model_summary["threshold_policy"]["threshold"]
    queue = build_work_queue(bundle.population, threshold, limit=25)
    assert not queue.empty
    assert {"Référence client", "Score", "Bande", "Persona", "Action"}.issubset(queue.columns)
    assert queue["Référence client"].astype(str).str.startswith("BC-").all()
    assert "CLIENTNUM" not in queue.columns

    selected = find_customer_by_reference(bundle.population, queue.iloc[0]["Référence client"])
    explanation = build_customer_explanation(selected)
    assert explanation["reference"].startswith("BC-")
    assert "CLIENTNUM" not in " ".join(explanation.values())


def test_v3_query_state_rejects_invalid_values_and_serializes_valid_state():
    bundle = load_dashboard_bundle(ROOT)
    options = available_filter_options(bundle.population)
    state = parse_query_state(
        {
            "risk": ["Très élevé", "Valeur inconnue"],
            "segment": "Dormants à très haut risque",
            "threshold": "0.285",
            "client": "BC-1234ABCD",
        },
        options,
        default_threshold=0.2,
    )
    assert state.filters["bande_risque"] == ["Très élevé"]
    assert state.filters["segment_client"] == ["Dormants à très haut risque"]
    assert state.threshold == 0.285
    assert state.selected_client == "BC-1234ABCD"

    serialized = serialize_query_state(state.filters, state.threshold, state.selected_client)
    assert serialized["risk"] == ["Très élevé"]
    assert serialized["threshold"] == "0.285"
    assert serialized["client"] == "BC-1234ABCD"


def test_v3_streamlit_capability_inventory_is_business_oriented():
    table = build_streamlit_capability_table()
    assert table.shape[0] >= 10
    assert {"Capacité Streamlit", "Usage métier"} == set(table.columns)
    assert table["Capacité Streamlit"].str.contains("st.dialog").any()
    assert table["Capacité Streamlit"].str.contains("st.query_params").any()
