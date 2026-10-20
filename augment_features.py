"""Phase 3c: build augmented log-Mel features from TRAINING actors only."""
import json
import sys

import numpy as np
import pandas as pd

from utils.dataset import resolve_audio_path 

from utils.data import make_actor_split

import config
from preprocessing import extract_features
from utils.audio_processing import (
    AudioValidationError, augment_waveform, load_audio,
    standardize_length, trim_silence, validate_waveform,
)


def main() -> int:
    if not config.DATASET_INDEX_PATH.exists():
        print("Run inspect_dataset.py first.")
        return 1

    index = pd.read_csv(config.DATASET_INDEX_PATH)
    split = make_actor_split(index["actor"])  # the same deterministic split train.py uses
    train = index[index["actor"].isin(split["train"])].reset_index(drop=True)
    rng = np.random.default_rng(config.SEED)

    X, y, actors = [], [], []
    for i, row in train.iterrows():
        try:
            wave = load_audio(resolve_audio_path(row["path"]))
            validate_waveform(wave)
            wave = trim_silence(wave)
        except AudioValidationError as exc:
            print(f"Skipping {row['filename']}: {exc}")
            continue
        for _ in range(config.AUG_COPIES):
            aug = standardize_length(augment_waveform(wave, rng))
            X.append(extract_features(aug)["logmel"])
            y.append(int(row["label_id"]))
            actors.append(int(row["actor"]))
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(train)} source clips")

    X = np.stack(X)
    assert np.isfinite(X).all(), "Non-finite augmented features"
    assert set(actors) <= set(split["train"]), "Augmented data includes non-training actors"

    np.savez_compressed(
        config.AUG_FEATURES_PATH,
        X_logmel=X, y=np.array(y, dtype=np.int64), actor=np.array(actors, dtype=np.int64),
    )
    print(f"\nAugmented clips: {len(X)}  shape: {X.shape}")
    print(f"Saved: {config.AUG_FEATURES_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())