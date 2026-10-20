"""Phase 4: final evaluation on the held-out test actors.

Run once the model is final. The test split is not used anywhere else.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from math import sqrt

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config
from preprocessing import apply_normalizer, load_normalizer
from utils.data import assert_no_leakage, load_features, split_indices

FIGURES_DIR = config.ARTIFACTS_DIR / "figures"
METRICS_PATH = config.ARTIFACTS_DIR / "evaluation_metrics.json"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(config.TARGET_EMOTIONS)))
    kw = {"labels": labels, "zero_division": 0}
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, **kw)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", **kw)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", **kw)
    correct = int((y_true == y_pred).sum())
    lo, hi = wilson_interval(correct, len(y_true))
    return {
        "accuracy": float(correct / len(y_true)),
        "accuracy_ci95": [float(lo), float(hi)],
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_f1": float(wf),
        "labels": config.TARGET_EMOTIONS,
        "per_class": {
            config.ID_TO_LABEL[i]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#DEDED7")
    ax.tick_params(colors="#686963")


def plot_confusion_matrix(cm, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = np.asarray(cm)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    labels = config.TARGET_EMOTIONS
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("forest", ["#FFFFFF", "#356B52"]), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted", color="#20211F")
    ax.set_ylabel("True", color="#20211F")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="#FFFFFF" if norm[i, j] > 0.5 else "#20211F")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_history(history: dict, key: str, title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    ax.plot(range(1, len(history[key]) + 1), history[key], color="#356B52", label="Training")
    ax.plot(range(1, len(history[f"val_{key}"]) + 1), history[f"val_{key}"], color="#686963", label="Validation")
    ax.set_title(title, loc="left", color="#20211F")
    ax.set_xlabel("Epoch", color="#20211F")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    required = (config.MODEL_PATH, config.NORMALIZATION_PATH, config.SPLIT_PATH,
                config.MODEL_CONFIG_PATH, config.FEATURES_PATH)
    for p in required:
        if not p.exists():
            print(f"Missing {p}. Run the earlier phases first.")
            return 1

    import keras

    model_cfg = json.loads(config.MODEL_CONFIG_PATH.read_text())
    if model_cfg["labels"] != config.TARGET_EMOTIONS:
        print("Label order in model_config.json does not match config.TARGET_EMOTIONS.")
        return 1

    split = json.loads(config.SPLIT_PATH.read_text())
    assert_no_leakage(split)

    data = load_features()
    idx = split_indices(data["actor"], split)["test"]
    X_test = apply_normalizer(data["X"][idx], load_normalizer())
    y_test, actors = data["y"][idx], data["actor"][idx]

    model = keras.models.load_model(config.MODEL_PATH)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred)
    metrics.update(
        n_test=int(len(y_test)),
        test_actors=split["test"],
        per_actor_accuracy={str(a): float((y_pred[actors == a] == y_test[actors == a]).mean())
                            for a in split["test"]},
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        model_file=config.MODEL_PATH.name,
    )

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(metrics["confusion_matrix"], FIGURES_DIR / "confusion_matrix.png")
    if config.HISTORY_PATH.exists():
        history = json.loads(config.HISTORY_PATH.read_text())
        plot_history(history, "accuracy", "Training vs validation accuracy", FIGURES_DIR / "accuracy_curve.png")
        plot_history(history, "loss", "Training vs validation loss", FIGURES_DIR / "loss_curve.png")

    lo, hi = metrics["accuracy_ci95"]
    print(f"Test clips: {metrics['n_test']}   Test actors: {split['test']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"Macro precision {metrics['macro_precision']:.4f}  recall {metrics['macro_recall']:.4f}  "
          f"F1 {metrics['macro_f1']:.4f}   Weighted F1 {metrics['weighted_f1']:.4f}")
    print("\nPer class:")
    for name, m in metrics["per_class"].items():
        print(f"  {name:8s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} support={m['support']}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("  order:", config.TARGET_EMOTIONS)
    for name, row in zip(config.TARGET_EMOTIONS, metrics["confusion_matrix"]):
        print(f"  {name:8s}", row)
    print("\nPer-actor accuracy:", {a: round(v, 3) for a, v in metrics["per_actor_accuracy"].items()})
    print(f"\nSaved: {METRICS_PATH.relative_to(config.PROJECT_ROOT)} and artifacts/figures/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())