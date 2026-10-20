"""Loading cached features and creating leakage-free, actor-wise splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config


def load_features(path=config.FEATURES_PATH, feature: str = "logmel") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_features.py first.")
    with np.load(path, allow_pickle=False) as data:
        X = data["X_logmel"] if feature == "logmel" else data["X_mfcc"]
        return {"X": X, "y": data["y"], "actor": data["actor"], "path": data["path"]}


def make_actor_split(
    actors,
    seed: int = config.SEED,
    val_per_gender: int = config.VAL_ACTORS_PER_GENDER,
    test_per_gender: int = config.TEST_ACTORS_PER_GENDER,
) -> dict[str, list[int]]:
    """Split by speaker. RAVDESS: odd actor IDs are male, even are female."""
    unique = sorted({int(a) for a in actors})
    rng = np.random.default_rng(seed)
    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for odd in (True, False):
        group = [a for a in unique if (a % 2 == 1) == odd]
        if len(group) <= val_per_gender + test_per_gender:
            raise ValueError("Not enough actors per gender for the requested split.")
        shuffled = [int(a) for a in rng.permutation(group)]
        split["test"] += shuffled[:test_per_gender]
        split["val"] += shuffled[test_per_gender:test_per_gender + val_per_gender]
        split["train"] += shuffled[test_per_gender + val_per_gender:]

    return {name: sorted(ids) for name, ids in split.items()}


def split_indices(actor_array, split: dict[str, list[int]]) -> dict[str, np.ndarray]:
    return {name: np.flatnonzero(np.isin(actor_array, ids)) for name, ids in split.items()}


def assert_no_leakage(split: dict[str, list[int]]) -> None:
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    if (train & val) or (train & test) or (val & test):
        raise ValueError("Actor overlap between splits (data leakage).")