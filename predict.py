"""Inference. Turns an audio source into emotion probabilities. No Streamlit code here."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from preprocessing import apply_normalizer, extract_features, load_normalizer
from utils.audio_processing import prepare_waveform
from utils.calibration import apply_temperature


class ModelNotFoundError(FileNotFoundError):
    """Raised when trained model files are missing."""


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict          # in config.TARGET_EMOTIONS order, calibrated if enabled
    raw_probabilities: dict
    temperature: float
    waveform: np.ndarray         # the processed clip the model analyzed
    features: dict               # un-normalized "logmel" and "mfcc"
    info: dict                   # original_seconds, trimmed_seconds, truncated


class EmotionPredictor:
    def __init__(
        self,
        model_path=config.MODEL_PATH,
        normalizer_path=config.NORMALIZATION_PATH,
        calibration_path=config.CALIBRATION_PATH,
        model_config_path=config.MODEL_CONFIG_PATH,
        use_calibration: bool = config.USE_CALIBRATION,
    ):
        missing = [Path(p).name for p in (model_path, normalizer_path) if not Path(p).exists()]
        if missing:
            raise ModelNotFoundError(
                "Missing model files: " + ", ".join(missing) + ". Run train.py first."
            )
        if model_config_path is not None and Path(model_config_path).exists():
            labels = json.loads(Path(model_config_path).read_text())["labels"]
            if labels != config.TARGET_EMOTIONS:
                raise ValueError("Label order in model_config.json does not match config.TARGET_EMOTIONS.")

        import keras

        self.model = keras.models.load_model(model_path)
        if self.model.output_shape[-1] != len(config.TARGET_EMOTIONS):
            raise ValueError("The model output size does not match the supported emotions.")
        self.stats = load_normalizer(normalizer_path)

        self.temperature = 1.0
        if use_calibration and calibration_path is not None and Path(calibration_path).exists():
            self.temperature = float(json.loads(Path(calibration_path).read_text())["temperature"])

    def predict_source(self, source) -> Prediction:
        """source: a file path or a file-like object. Raises AudioValidationError for bad audio."""
        waveform, info = prepare_waveform(source)
        features = extract_features(waveform)
        x = apply_normalizer(features["logmel"][np.newaxis], self.stats)
        raw = np.asarray(self.model(x, training=False), dtype=np.float64)[0]
        probs = apply_temperature(raw[np.newaxis], self.temperature)[0]

        labels = config.TARGET_EMOTIONS
        top = int(np.argmax(probs))
        return Prediction(
            label=labels[top],
            confidence=float(probs[top]),
            probabilities={l: float(p) for l, p in zip(labels, probs)},
            raw_probabilities={l: float(p) for l, p in zip(labels, raw)},
            temperature=self.temperature,
            waveform=waveform,
            features=features,
            info=info,
        )


def get_model_status() -> dict:
    """Cheap status check for the dashboard. Does not load TensorFlow."""
    missing = [p.name for p in (config.MODEL_PATH, config.NORMALIZATION_PATH) if not p.exists()]
    status = {"ready": not missing, "missing": missing}
    for key, path in (
        ("config", config.MODEL_CONFIG_PATH),
        ("metrics", config.EVALUATION_METRICS_PATH),
        ("calibration", config.CALIBRATION_PATH),
    ):
        status[key] = json.loads(path.read_text()) if path.exists() else None
    return status


def get_dataset_info() -> dict | None:
    if not config.DATASET_INDEX_PATH.exists():
        return None
    import pandas as pd

    df = pd.read_csv(config.DATASET_INDEX_PATH)
    counts = df["emotion"].value_counts().reindex(config.TARGET_EMOTIONS).fillna(0).astype(int)
    return {
        "recordings": int(len(df)),
        "actors": int(df["actor"].nunique()),
        "class_counts": counts.to_dict(),
    }