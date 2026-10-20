"""Phase 3: actor-wise split, normalization, and CNN-BiLSTM training.

The test split is NOT used here. It is evaluated once in Phase 4.
"""
import json
import sys

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import config
from model import build_model
from preprocessing import apply_normalizer, fit_normalizer, save_normalizer
from utils.data import assert_no_leakage, load_features, make_actor_split, split_indices


def spec_augment(x, y):
    """Mask one random frequency band and one random time span (zero = mean after normalization)."""
    n_mels, n_frames = config.N_MELS, config.N_FRAMES
    f = tf.random.uniform([], 0, config.FREQ_MASK_MAX + 1, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, n_mels - f + 1, dtype=tf.int32)
    t = tf.random.uniform([], 0, config.TIME_MASK_MAX + 1, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, n_frames - t + 1, dtype=tf.int32)
    fi, ti = tf.range(n_mels), tf.range(n_frames)
    fmask = tf.logical_and(fi >= f0, fi < f0 + f)
    tmask = tf.logical_and(ti >= t0, ti < t0 + t)
    mask = tf.logical_or(fmask[:, None], tmask[None, :])[..., None]
    return tf.where(mask, tf.zeros_like(x), x), y


def make_dataset(X, y, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=config.SEED).map(spec_augment)
    return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def class_counts(y):
    return {config.ID_TO_LABEL[i]: int((y == i).sum()) for i in range(len(config.TARGET_EMOTIONS))}

class MacroF1Selection(keras.callbacks.Callback):
    """Keep the weights with the best validation macro-F1; stop after `patience` stale epochs."""

    def __init__(self, X_val, y_val, patience):
        super().__init__()
        self.X_val, self.y_val, self.patience = X_val, y_val, patience
        self.best, self.best_epoch, self.best_weights, self.wait = -1.0, 0, None, 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        score = f1_score(self.y_val, pred, average="macro")
        if logs is not None:
            logs["val_macro_f1"] = score
        if score > self.best:
            self.best, self.best_epoch, self.wait = score, epoch + 1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def main() -> int:
    keras.utils.set_random_seed(config.SEED)
    try:
        data = load_features()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    X, y, actor = data["X"], data["y"], data["actor"]
    split = make_actor_split(actor)
    assert_no_leakage(split)
    idx = split_indices(actor, split)

    print("Actor split:")
    for name in ("train", "val", "test"):
        print(f"  {name:5s} actors={split[name]}  clips={len(idx[name])}  {class_counts(y[idx[name]])}")

    X_tr, y_tr = X[idx["train"]], y[idx["train"]]
    if config.AUG_FEATURES_PATH.exists():
        with np.load(config.AUG_FEATURES_PATH, allow_pickle=False) as aug:
            if not set(np.unique(aug["actor"]).tolist()) <= set(split["train"]):
                raise ValueError("Augmented data contains non-training actors (leakage).")
            X_tr = np.concatenate([X_tr, aug["X_logmel"]])
            y_tr = np.concatenate([y_tr, aug["y"]])
        print(f"Using augmented training set: {len(X_tr)} clips")

    stats = fit_normalizer(X_tr)
    save_normalizer(stats)
    X_train, y_train = apply_normalizer(X_tr, stats), y_tr
    X_val, y_val = apply_normalizer(X[idx["val"]], stats), y[idx["val"]]

    n_classes = len(config.TARGET_EMOTIONS)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("Class weights:", {config.ID_TO_LABEL[i]: round(w, 3) for i, w in class_weight.items()})

    model = build_model()
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    selector = MacroF1Selection(X_val, y_val, patience=config.EARLY_STOP_PATIENCE)
    callbacks = [
        selector,
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]
    history = model.fit(
        make_dataset(X_train, y_train, training=True),
        validation_data=make_dataset(X_val, y_val, training=False),
        epochs=config.EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    print(f"Best epoch (by val macro-F1): {selector.best_epoch}  score={selector.best:.4f}")

    # Best weights are restored by EarlyStopping; save them.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)

    val_loss, val_acc = model.evaluate(make_dataset(X_val, y_val, training=False), verbose=0)
    val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    print(f"\nValidation  loss={val_loss:.4f}  accuracy={val_acc:.4f}  macro-F1={val_f1:.4f}")
    print(f"Epochs run: {len(history.history['loss'])}")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.HISTORY_PATH.write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2)
    )
    config.SPLIT_PATH.write_text(json.dumps(split, indent=2))
    config.MODEL_CONFIG_PATH.write_text(json.dumps({
        "architecture": "CNN-BiLSTM",
        "feature": "logmel",
        "input_shape": [config.N_MELS, config.N_FRAMES, 1],
        "labels": config.TARGET_EMOTIONS,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "class_weight": {config.ID_TO_LABEL[i]: w for i, w in class_weight.items()},
        "seed": config.SEED,
        "train_actors": split["train"],
        "val_actors": split["val"],
        "test_actors": split["test"],
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
    }, indent=2))

    print(f"Saved: {config.MODEL_PATH.relative_to(config.PROJECT_ROOT)}")
    for p in (config.NORMALIZATION_PATH, config.HISTORY_PATH, config.SPLIT_PATH, config.MODEL_CONFIG_PATH):
        print(f"Saved: {p.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())