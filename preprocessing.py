"""Feature extraction. The single definition of how audio becomes model input."""
from __future__ import annotations

import json

import joblib

import librosa
import numpy as np

import config


def extract_log_mel(y: np.ndarray) -> np.ndarray:
    """Log-Mel spectrogram in dB, shape (N_MELS, N_FRAMES)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS,
    )
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """MFCCs, shape (N_MFCC, N_FRAMES)."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS,
    )
    return mfcc.astype(np.float32)


def extract_features(y: np.ndarray) -> dict[str, np.ndarray]:
    """Extract all features from a standardized waveform of length N_SAMPLES."""
    if len(y) != config.N_SAMPLES:
        raise ValueError(f"Expected {config.N_SAMPLES} samples, got {len(y)}.")
    return {"logmel": extract_log_mel(y), "mfcc": extract_mfcc(y)}


def save_preprocessing_config() -> None:
    """Write the settings used for feature extraction next to the model artifacts."""
    settings = {
        "sample_rate": config.SAMPLE_RATE,
        "clip_duration": config.CLIP_DURATION,
        "n_samples": config.N_SAMPLES,
        "trim_top_db": config.TRIM_TOP_DB,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "n_mels": config.N_MELS,
        "n_mfcc": config.N_MFCC,
        "n_frames": config.N_FRAMES,
        "logmel_ref": 1.0,
        "logmel_top_db": 80.0,
        "seed": config.SEED,
    }
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PREPROCESSING_CONFIG_PATH.write_text(json.dumps(settings, indent=2))

def fit_normalizer(X_train: np.ndarray) -> dict[str, np.ndarray]:
    """Per-band mean/std from TRAINING data only. X shape: (N, n_bands, n_frames)."""
    mean = X_train.mean(axis=(0, 2)).astype(np.float32)
    std = X_train.std(axis=(0, 2)).astype(np.float32)
    return {"mean": mean, "std": np.maximum(std, 1e-6)}


def apply_normalizer(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize and add a channel axis: (N, bands, frames) -> (N, bands, frames, 1)."""
    X = np.asarray(X, dtype=np.float32)
    X = (X - stats["mean"][None, :, None]) / stats["std"][None, :, None]
    return X[..., np.newaxis]


def save_normalizer(stats: dict[str, np.ndarray], path=config.NORMALIZATION_PATH) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stats, path)


def load_normalizer(path=config.NORMALIZATION_PATH) -> dict[str, np.ndarray]:
    return joblib.load(path)
