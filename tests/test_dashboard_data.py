from pathlib import Path

from dashboard.data import (
    apply_filters,
    available_filter_options,
    build_action_table,
    build_executive_story,
    build_high_risk_comparison,
    build_local_explanation_table,
    build_persona_difference_table,
    compute_filtered_metrics,
    get_filtered_test_subset,
    load_dashboard_bundle,
)


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
    assert {"Cas", "Segment client", "Issue de prédiction"}.issubset(local_cases.columns)

    persona_name = bundle.population["segment_client"].mode().iloc[0]
    persona_frame = bundle.population[bundle.population["segment_client"] == persona_name]
    diff_table = build_persona_difference_table(persona_frame, bundle.population)
    assert not diff_table.empty
    assert {"Indicateur", "Persona", "Reste du périmètre", "Écart relatif"} == set(diff_table.columns)

    risk_comparison = build_high_risk_comparison(bundle.population)
    assert not risk_comparison.empty
    assert {"Indicateur", "Clients à risque élevé", "Reste du portefeuille", "Écart relatif"} == set(risk_comparison.columns)
