"""Phase 2: extract features for every supported recording and cache them."""
import sys

import numpy as np
import pandas as pd

from utils.dataset import resolve_audio_path 

import config
from preprocessing import extract_features, save_preprocessing_config
from utils.audio_processing import AudioValidationError, prepare_waveform


def main() -> int:
    if not config.DATASET_INDEX_PATH.exists():
        print("Run `python inspect_dataset.py` first (Phase 1).")
        return 1

    index = pd.read_csv(config.DATASET_INDEX_PATH)
    logmels, mfccs, kept_rows, trimmed_secs, truncated = [], [], [], [], []
    failures = []

    for i, row in index.iterrows():
        try:
            y, info = prepare_waveform(resolve_audio_path(row["path"]))
            feats = extract_features(y)
        except (AudioValidationError, ValueError) as exc:
            failures.append((row["filename"], str(exc)))
            continue
        logmels.append(feats["logmel"])
        mfccs.append(feats["mfcc"])
        kept_rows.append(i)
        trimmed_secs.append(info["trimmed_seconds"])
        truncated.append(info["truncated"])
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(index)}")

    if not kept_rows:
        print("No files could be processed.")
        return 1

    kept = index.loc[kept_rows]
    X_logmel = np.stack(logmels)
    X_mfcc = np.stack(mfccs)

    assert np.isfinite(X_logmel).all() and np.isfinite(X_mfcc).all(), "Non-finite features"

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.FEATURES_PATH,
        X_logmel=X_logmel,
        X_mfcc=X_mfcc,
        y=kept["label_id"].to_numpy(dtype=np.int64),
        actor=kept["actor"].to_numpy(dtype=np.int64),
        path=kept["path"].to_numpy(dtype=str),
    )
    save_preprocessing_config()

    secs = np.array(trimmed_secs)
    print(f"\nProcessed: {len(kept)} / {len(index)}   Failed: {len(failures)}")
    for name, reason in failures[:10]:
        print(f"  FAILED {name}: {reason}")
    print(f"X_logmel shape: {X_logmel.shape}")
    print(f"X_mfcc shape:   {X_mfcc.shape}")
    print("\nDuration after silence trimming (seconds):")
    print(f"  min {secs.min():.2f}  median {np.median(secs):.2f}  "
          f"95th pct {np.percentile(secs, 95):.2f}  max {secs.max():.2f}")
    print(f"  Truncated to {config.CLIP_DURATION}s: {sum(truncated)} clips "
          f"({100 * np.mean(truncated):.1f}%)")
    print(f"\nSaved: {config.FEATURES_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"Saved: {config.PREPROCESSING_CONFIG_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())