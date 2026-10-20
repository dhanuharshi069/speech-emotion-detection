"""Validation-only diagnostics for the saved model. Never touches the test split."""
import json
import sys

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config
from preprocessing import apply_normalizer, load_normalizer
from utils.data import load_features, split_indices


def main() -> int:
    for p in (config.MODEL_PATH, config.HISTORY_PATH, config.SPLIT_PATH, config.NORMALIZATION_PATH):
        if not p.exists():
            print(f"Missing {p}. Run train.py first.")
            return 1

    history = json.loads(config.HISTORY_PATH.read_text())
    keys = [k for k in ("accuracy", "loss", "val_accuracy", "val_loss", "val_macro_f1") if k in history]
    table = pd.DataFrame({k: history[k] for k in keys})
    table.index = table.index + 1
    table.index.name = "epoch"
    print(table.round(4).to_string())
    print(f"\nBest val_loss at epoch {int(table['val_loss'].idxmin())}")

    split = json.loads(config.SPLIT_PATH.read_text())
    data = load_features()
    idx = split_indices(data["actor"], split)["val"]
    X_val = apply_normalizer(data["X"][idx], load_normalizer())
    y_val = data["y"][idx]

    model = keras.models.load_model(config.MODEL_PATH)
    pred = model.predict(X_val, verbose=0).argmax(axis=1)

    labels = config.TARGET_EMOTIONS
    cm = confusion_matrix(y_val, pred, labels=range(len(labels)))
    print("\nValidation confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print("\nPer-class recall:", {l: round(float(cm[i, i] / max(cm[i].sum(), 1)), 3) for i, l in enumerate(labels)})
    print("Predicted counts:", {l: int((pred == i).sum()) for i, l in enumerate(labels)})
    return 0


if __name__ == "__main__":
    sys.exit(main())