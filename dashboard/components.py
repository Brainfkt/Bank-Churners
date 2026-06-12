from __future__ import annotations

import inspect

import streamlit as st

from dashboard.content import GLOSSARY


def render_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bank-bg: #f8fafc;
            --bank-surface: #ffffff;
            --bank-border: #d9e1ec;
            --bank-text: #182033;
            --bank-muted: #667085;
            --bank-blue: #2563eb;
            --bank-coral: #fb5a4f;
            --bank-teal: #20b2a6;
            --bank-amber: #f59e0b;
        }
        .stApp {
            background: var(--bank-bg);
        }
        [data-testid="stHeader"] {
            border-bottom: 1px solid rgba(217, 225, 236, 0.85);
        }
        .block-container {
            padding-top: 4.6rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stMetric"] {
            background: var(--bank-surface);
            border-color: var(--bank-border);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(24, 32, 51, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: var(--bank-muted);
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: var(--bank-text);
        }
        .bank-page-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.35rem 0 1rem 0;
        }
        .bank-page-title {
            font-size: clamp(1.9rem, 4vw, 2.55rem);
            font-weight: 760;
            line-height: 1.08;
            letter-spacing: 0;
            color: var(--bank-text);
            margin: 0;
        }
        .bank-page-subtitle {
            margin: 0.55rem 0 0 0;
            max-width: 780px;
            color: #31415f;
            font-size: 1rem;
            line-height: 1.55;
        }
        .bank-chip-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            margin: 0.15rem 0 0.8rem 0;
        }
        .bank-chip {
            display: inline-flex;
            align-items: center;
            min-height: 30px;
            padding: 0.25rem 0.62rem;
            border: 1px solid var(--bank-border);
            border-radius: 999px;
            background: var(--bank-surface);
            color: #31415f;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .bank-chip strong {
            color: var(--bank-blue);
            font-weight: 750;
            margin-left: 0.25rem;
        }
        .bank-card-title {
            font-weight: 740;
            color: var(--bank-text);
            margin-bottom: 0.15rem;
        }
        .bank-card-caption {
            color: var(--bank-muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .bank-mini-note {
            padding: 0.7rem 0.85rem;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            background: #eff6ff;
            color: #1e3a8a;
            font-size: 0.9rem;
        }
        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 4.2rem;
            }
            .bank-page-header {
                display: block;
            }
            .bank-chip-row {
                gap: 0.35rem;
            }
            .bank-chip {
                max-width: 100%;
                white-space: normal;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, status: str | None = None) -> None:
    status_html = f"<span class='bank-chip'>{status}</span>" if status else ""
    st.markdown(
        f"""
        <div class="bank-page-header">
            <div>
                <h1 class="bank-page-title">{title}</h1>
                <p class="bank-page-subtitle">{subtitle}</p>
            </div>
            <div>{status_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_active_filter_pills(active_filters: list[str]) -> None:
    chips = active_filters or ["Aucun filtre actif"]
    chip_html = "".join(f"<span class='bank-chip'>{chip}</span>" for chip in chips[:8])
    if len(chips) > 8:
        chip_html += f"<span class='bank-chip'><strong>+{len(chips) - 8}</strong></span>"
    st.markdown(f"<div class='bank-chip-row'>{chip_html}</div>", unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    help_text: str,
    delta: str | None = None,
    chart_data: list[float] | None = None,
    delta_color: str = "normal",
) -> None:
    kwargs = {
        "label": label,
        "value": value,
        "delta": delta,
        "delta_color": delta_color,
        "help": help_text,
        "border": True,
    }
    metric_parameters = inspect.signature(st.metric).parameters
    if chart_data and "chart_data" in metric_parameters:
        kwargs["chart_data"] = chart_data
        kwargs["chart_type"] = "line"
    st.metric(**kwargs)


def card_title(title: str, caption: str | None = None) -> None:
    st.markdown(f"<div class='bank-card-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='bank-card-caption'>{caption}</div>", unsafe_allow_html=True)


def section_intro(text: str) -> None:
    st.markdown(f"<div style='padding:0.6rem 0 1rem 0;font-size:1.0rem;color:#34495e;'>{text}</div>", unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.info(f"À retenir : {text}")


def methodological_note(text: str) -> None:
    st.warning(f"Point de méthode : {text}")


def section_transition(text: str) -> None:
    st.caption(text)


def glossary_block(container=None) -> None:
    container = container or st
    with container.expander("Glossaire / lecture des indicateurs", expanded=False):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** : {definition}")


def metric_help(label: str, value: str, help_text: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta, help=help_text)


def small_definition(label: str, column_name: str, explanations: dict[str, str]) -> None:
    if column_name in explanations:
        st.caption(f"{label} : {explanations[column_name]}")
