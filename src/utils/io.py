from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Any, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default)


def save_frame(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_parent(path)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")
