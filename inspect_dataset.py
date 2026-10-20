"""Phase 1: inspect RAVDESS, apply the four-class mapping, save artifacts."""
import json
import sys

import config
from utils.dataset import build_dataset_index, filter_supported


def main() -> int:
    if not config.DATA_DIR.exists():
        print(f"Dataset folder not found: {config.DATA_DIR}")
        print("See data/README.md for download instructions.")
        return 1

    index = build_dataset_index()
    if index.empty:
        print(f"No RAVDESS speech .wav files found under {config.DATA_DIR}")
        print("Expected files like data/raw/Actor_01/03-01-05-01-02-01-01.wav")
        return 1

    print(f"Speech recordings found: {len(index)}")
    print(f"Actors found: {index['actor'].nunique()}")
    print("\nAll emotions in the dataset:")
    print(index["emotion_raw"].value_counts().to_string())

    df = filter_supported(index)
    dropped = len(index) - len(df)
    print(f"\nDropped {dropped} recordings with unsupported emotions.")
    print(f"Kept {len(df)} recordings for {config.TARGET_EMOTIONS}")
    print("\nClass counts after filtering:")
    print(df["emotion"].value_counts().reindex(config.TARGET_EMOTIONS).to_string())
    print("\nRecordings per actor (min / max):",
          df.groupby("actor").size().min(), "/", df.groupby("actor").size().max())

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_INDEX_PATH, index=False)

    mapping = {
        "target_emotions": config.TARGET_EMOTIONS,
        "label_to_id": config.LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in config.ID_TO_LABEL.items()},
        "ravdess_emotion_codes": config.RAVDESS_EMOTION_CODES,
        "excluded_emotions": sorted(
            set(config.RAVDESS_EMOTION_CODES.values()) - set(config.TARGET_EMOTIONS)
        ),
    }
    config.LABEL_MAPPING_PATH.write_text(json.dumps(mapping, indent=2))

    print(f"\nSaved: {config.DATASET_INDEX_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"Saved: {config.LABEL_MAPPING_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



