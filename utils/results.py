"""Loading saved result files for the dashboard. No Streamlit code."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def selected_epoch(history: dict) -> int | None:
    """1-based epoch with the highest validation macro-F1 (first maximum), as train.py selects it."""
    scores = history.get("val_macro_f1")
    if not scores:
        return None
    return int(np.argmax(scores)) + 1


def load_results() -> dict:
    return {
        "metrics": load_json(config.EVALUATION_METRICS_PATH),
        "history": load_json(config.HISTORY_PATH),
        "calibration": load_json(config.CALIBRATION_PATH),
    }

def split_summary(index_path=config.DATASET_INDEX_PATH, split_path=config.SPLIT_PATH) -> dict | None:
    """Actors and clip counts per split, from the saved index and split files."""
    split, index_path = load_json(split_path), Path(index_path)
    if split is None or not index_path.exists():
        return None
    import pandas as pd

    index = pd.read_csv(index_path)
    return {
        name: {"actors": list(actors), "clips": int(index["actor"].isin(actors).sum())}
        for name, actors in split.items()
    }