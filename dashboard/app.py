from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.components import glossary_block, render_active_filter_pills, render_app_styles
from dashboard.content import FILTER_LABELS, FILTER_ORDER
from dashboard.data import available_filter_options, apply_filters, format_active_filters, load_dashboard_bundle
from dashboard.state import parse_query_state, serialize_query_state
from dashboard.workflows import (
    render_customer_explorer,
    render_data_health,
    render_decision_hub,
    render_model_studio,
    render_retention_workbench,
)


st.set_page_config(
    page_title="Bank Churners Decision Hub",
    page_icon=":bar_chart:",
    layout="wide",
)


PAGE_RENDERERS: dict[str, Callable[[], None]] = {
    "Decision Hub": lambda: render_decision_hub(st.session_state["dashboard_bundle"], st.session_state["dashboard_filtered"]),
    "Customer Explorer": lambda: render_customer_explorer(st.session_state["dashboard_bundle"], st.session_state["dashboard_filtered"]),
    "Model Studio": lambda: render_model_studio(st.session_state["dashboard_bundle"], st.session_state["dashboard_filtered"]),
    "Retention Workbench": lambda: render_retention_workbench(st.session_state["dashboard_bundle"], st.session_state["dashboard_filtered"]),
    "Data Health": lambda: render_data_health(st.session_state["dashboard_bundle"], st.session_state["dashboard_filtered"]),
}


@st.cache_data(show_spinner=False)
def get_dashboard_bundle():
    return load_dashboard_bundle(ROOT)


def load_bundle_with_cache_guard():
    bundle = get_dashboard_bundle()
    required_attrs = {
        "population",
        "benchmark",
        "feature_importance",
        "model_summary",
        "unknown_strategy",
        "local_explanations",
        "segmentation_summary",
        "threshold_sensitivity",
        "calibration_table",
        "slice_metrics",
        "model_stability",
        "run_manifest",
    }
    if not required_attrs.issubset(set(vars(bundle).keys())):
        get_dashboard_bundle.clear()
        bundle = get_dashboard_bundle()
    return bundle


def main() -> None:
    render_app_styles()
    try:
        bundle = load_bundle_with_cache_guard()
    except FileNotFoundError:
        st.warning("Les artefacts nécessaires au dashboard sont introuvables. Lancez `python -m src.project_runner` avant d'ouvrir l'application.")
        return

    options = available_filter_options(bundle.population)
    recommended_threshold = float(bundle.model_summary["threshold_policy"]["threshold"])
    _initialize_state(options, recommended_threshold)

    page_name = _render_navigation()
    filters = _render_filter_bar(options, recommended_threshold)
    filtered = apply_filters(bundle.population, filters)
    if filtered.empty:
        st.warning("Aucun client ne correspond aux filtres actifs. Ajustez le périmètre avec le bouton Filtres.")
        return

    st.session_state["dashboard_bundle"] = bundle
    st.session_state["dashboard_filtered"] = filtered
    st.session_state["dashboard_filters"] = filters
    _sync_query_params(filters)

    if page_name is None:
        page = st.session_state["dashboard_page"]
        page.run()
    else:
        PAGE_RENDERERS[page_name]()


def _render_navigation() -> str | None:
    if _supports_top_navigation():
        pages = [
            st.Page(PAGE_RENDERERS["Decision Hub"], title="Decision Hub", icon=":material/home:", url_path="decision-hub", default=True),
            st.Page(PAGE_RENDERERS["Customer Explorer"], title="Customer Explorer", icon=":material/search:", url_path="customer-explorer"),
            st.Page(PAGE_RENDERERS["Model Studio"], title="Model Studio", icon=":material/model_training:", url_path="model-studio"),
            st.Page(PAGE_RENDERERS["Retention Workbench"], title="Retention Workbench", icon=":material/call:", url_path="retention-workbench"),
            st.Page(PAGE_RENDERERS["Data Health"], title="Data Health", icon=":material/verified:", url_path="data-health"),
        ]
        st.session_state["dashboard_page"] = st.navigation(pages, position="top")
        return None

    return st.segmented_control(
        "Navigation",
        options=list(PAGE_RENDERERS),
        default="Decision Hub",
        key="fallback_page",
        label_visibility="collapsed",
    )


def _render_filter_bar(options: dict[str, list[str]], recommended_threshold: float) -> dict[str, list[str]]:
    controls = st.columns([0.9, 3.6, 0.8], vertical_alignment="center")
    with controls[2]:
        if st.button("Réinitialiser", width="stretch", key="reset_filters"):
            for column in FILTER_ORDER:
                st.session_state[f"filter_{column}"] = []
            st.session_state["decision_threshold"] = recommended_threshold
            st.session_state["selected_client_ref"] = None
            st.query_params.clear()
            st.toast("Filtres réinitialisés")

    with controls[0]:
        with st.popover("Filtres", icon=":material/filter_alt:", width="stretch"):
            st.caption("Les filtres descriptifs recalculent les vues métier. Les métriques prédictives locales restent soumises au volume test disponible.")
            st.pills(
                "Bande de risque",
                options=options["bande_risque"],
                selection_mode="multi",
                key="filter_bande_risque",
            )
            st.pills(
                "Statut d'activité",
                options=options["statut_activite"],
                selection_mode="multi",
                key="filter_statut_activite",
            )
            for column in FILTER_ORDER:
                if column in {"bande_risque", "statut_activite"}:
                    continue
                st.multiselect(FILTER_LABELS[column], options=options[column], key=f"filter_{column}")
            glossary_block(st)

    filters = {column: list(st.session_state.get(f"filter_{column}", [])) for column in FILTER_ORDER}
    active_filters = format_active_filters(filters)
    with controls[1]:
        render_active_filter_pills(active_filters)
    return filters


def _initialize_state(options: dict[str, list[str]], recommended_threshold: float) -> None:
    if "query_state_loaded" not in st.session_state:
        parsed = parse_query_state(dict(st.query_params), options, recommended_threshold)
        for column in FILTER_ORDER:
            st.session_state[f"filter_{column}"] = parsed.filters.get(column, [])
        st.session_state["decision_threshold"] = parsed.threshold
        st.session_state["selected_client_ref"] = parsed.selected_client
        st.session_state["query_state_loaded"] = True
    else:
        for column in FILTER_ORDER:
            st.session_state.setdefault(f"filter_{column}", [])
        st.session_state.setdefault("decision_threshold", recommended_threshold)
        st.session_state.setdefault("selected_client_ref", None)


def _sync_query_params(filters: dict[str, list[str]]) -> None:
    params = serialize_query_state(
        filters,
        float(st.session_state.get("decision_threshold", 0.0)),
        st.session_state.get("selected_client_ref"),
    )
    try:
        st.query_params.clear()
        for key, value in params.items():
            st.query_params[key] = value
    except Exception:
        pass


def _supports_top_navigation() -> bool:
    try:
        position = inspect.signature(st.navigation).parameters["position"]
    except (KeyError, ValueError):
        return False
    return "top" in str(position.annotation)


if __name__ == "__main__":
    main()
