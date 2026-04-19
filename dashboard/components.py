from __future__ import annotations

import streamlit as st

from dashboard.content import GLOSSARY


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
