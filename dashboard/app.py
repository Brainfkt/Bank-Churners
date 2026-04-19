from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.components import glossary_block
from dashboard.content import FILTER_LABELS, FILTER_ORDER, SECTION_ORDER
from dashboard.data import available_filter_options, apply_filters, format_active_filters, load_dashboard_bundle
from dashboard.sections import (
    render_business_actions,
    render_churn_drivers,
    render_customer_profiles,
    render_customer_segmentation,
    render_executive_overview,
    render_model_performance,
    render_risk_scoring,
)


st.set_page_config(
    page_title="Bank Churners Decision Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dashboard_bundle():
    return load_dashboard_bundle(ROOT)


def load_bundle_with_cache_guard():
    bundle = get_dashboard_bundle()
    required_attrs = {"population", "benchmark", "feature_importance", "model_summary", "unknown_strategy", "local_explanations", "segmentation_summary"}
    if not required_attrs.issubset(set(vars(bundle).keys())):
        get_dashboard_bundle.clear()
        bundle = get_dashboard_bundle()
    return bundle


def main() -> None:
    try:
        bundle = load_bundle_with_cache_guard()
    except FileNotFoundError:
        st.warning("Les artefacts nécessaires au dashboard sont introuvables. Lancez `python -m src.project_runner` avant d'ouvrir l'application.")
        return

    section, filters = render_sidebar(bundle.population)
    filtered = apply_filters(bundle.population, filters)
    if filtered.empty:
        st.warning("Aucun client ne correspond aux filtres actifs. Ajustez le périmètre dans la barre latérale.")
        return

    if section == "Executive Overview":
        render_executive_overview(bundle, filtered)
    elif section == "Customer Profiles":
        render_customer_profiles(filtered)
    elif section == "Churn Drivers":
        render_churn_drivers(bundle, filtered)
    elif section == "Model Performance":
        render_model_performance(bundle, filtered)
    elif section == "Risk Scoring":
        render_risk_scoring(filtered)
    elif section == "Customer Segmentation":
        render_customer_segmentation(bundle, filtered)
    elif section == "Business Actions":
        render_business_actions(filtered)


def render_sidebar(population):
    st.sidebar.title("Bank Churners")
    st.sidebar.caption("Tableau de bord d’aide à la décision pour la rétention, conçu pour un public métier et data.")
    st.sidebar.markdown("### Pages du rapport")
    section = st.sidebar.radio(
        "Pages du rapport",
        SECTION_ORDER,
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### Filtres")
    filter_keys = [f"filter_{column}" for column in FILTER_ORDER]
    if st.sidebar.button("Réinitialiser les filtres", use_container_width=True):
        for key in filter_keys:
            if key in st.session_state:
                st.session_state[key] = []

    options = available_filter_options(population)
    filters = {}
    for column in FILTER_ORDER:
        filters[column] = st.sidebar.multiselect(
            FILTER_LABELS[column],
            options=options[column],
            key=f"filter_{column}",
        )

    st.sidebar.caption("Les vues descriptives se recalculent sur le périmètre filtré. Les vues de performance prédictive sont recalculées uniquement si le sous-échantillon de test reste interprétable.")
    glossary_block(st.sidebar)
    return section, filters


if __name__ == "__main__":
    main()
