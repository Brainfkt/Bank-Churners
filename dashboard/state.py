from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


FILTER_QUERY_KEYS = {
    "risk": "bande_risque",
    "segment": "segment_client",
    "activity": "statut_activite",
    "card": "categorie_carte",
    "income": "categorie_revenu",
    "age": "tranche_age",
    "product": "profil_produit",
}

QUERY_FILTER_KEYS = {column: key for key, column in FILTER_QUERY_KEYS.items()}
THRESHOLD_QUERY_KEY = "threshold"
CLIENT_QUERY_KEY = "client"


@dataclass(frozen=True)
class DashboardQueryState:
    filters: dict[str, list[str]]
    threshold: float
    selected_client: str | None


def parse_query_state(
    params: Mapping[str, object],
    options: Mapping[str, list[str]],
    default_threshold: float,
) -> DashboardQueryState:
    filters: dict[str, list[str]] = {column: [] for column in QUERY_FILTER_KEYS}
    for query_key, column in FILTER_QUERY_KEYS.items():
        allowed = set(options.get(column, []))
        selected = [value for value in _coerce_values(params.get(query_key)) if value in allowed]
        filters[column] = selected

    threshold = _parse_threshold(params.get(THRESHOLD_QUERY_KEY), default_threshold)
    selected_client_values = _coerce_values(params.get(CLIENT_QUERY_KEY))
    selected_client = selected_client_values[0] if selected_client_values else None
    if selected_client and not selected_client.startswith("BC-"):
        selected_client = None

    return DashboardQueryState(filters=filters, threshold=threshold, selected_client=selected_client)


def serialize_query_state(
    filters: Mapping[str, list[str]],
    threshold: float | None,
    selected_client: str | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {}
    for column, values in filters.items():
        query_key = QUERY_FILTER_KEYS.get(column)
        if query_key and values:
            params[query_key] = values
    if threshold is not None:
        params[THRESHOLD_QUERY_KEY] = f"{threshold:.3f}"
    if selected_client:
        params[CLIENT_QUERY_KEY] = selected_client
    return params


def _coerce_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    return [item for item in values if item]


def _parse_threshold(value: object, default_threshold: float) -> float:
    values = _coerce_values(value)
    if not values:
        return default_threshold
    try:
        threshold = float(values[0])
    except ValueError:
        return default_threshold
    return min(0.95, max(0.05, threshold))
