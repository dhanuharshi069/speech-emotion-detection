"""Checks that the app's inference path reproduces evaluate.py's test predictions."""
import json
import sys

import numpy as np

import config

from utils.dataset import resolve_audio_path 

from predict import EmotionPredictor
from preprocessing import apply_normalizer, load_normalizer
from utils.data import load_features, split_indices


def main() -> int:
    for p in (config.MODEL_PATH, config.NORMALIZATION_PATH, config.SPLIT_PATH, config.FEATURES_PATH):
        if not p.exists():
            print(f"Missing {p}.")
            return 1

    split = json.loads(config.SPLIT_PATH.read_text())
    data = load_features()
    idx = split_indices(data["actor"], split)["test"]
    paths, y_true = data["path"][idx], data["y"][idx]

    predictor = EmotionPredictor()

    # Path A: cached features, as evaluate.py does it.
    X = apply_normalizer(data["X"][idx], load_normalizer())
    cached = np.asarray(predictor.model.predict(X, verbose=0), dtype=np.float64)

    # Path B: the app's path, from the audio files.
    app = np.array([
        [predictor.predict_source(resolve_audio_path(p)).raw_probabilities[l] for l in config.TARGET_EMOTIONS]
        for p in paths
    ])

    same = int((cached.argmax(axis=1) == app.argmax(axis=1)).sum())
    acc_app = float((app.argmax(axis=1) == y_true).mean())
    max_diff = float(np.abs(cached - app).max())

    print(f"Test clips: {len(paths)}")
    print(f"Identical predictions: {same}/{len(paths)}")
    print(f"Accuracy via app path: {acc_app:.4f}")
    if config.EVALUATION_METRICS_PATH.exists():
        reported = json.loads(config.EVALUATION_METRICS_PATH.read_text())["accuracy"]
        print(f"Accuracy in evaluation_metrics.json: {reported:.4f}")
    print(f"Max absolute probability difference: {max_diff:.2e}")
    return 0 if same == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())