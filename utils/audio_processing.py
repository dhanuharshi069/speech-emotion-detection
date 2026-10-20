"""Audio loading, validation and standardization.

Used identically by the training pipeline and the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

import config


class AudioValidationError(ValueError):
    """Raised when an audio file cannot be used. The message is user-facing."""


def load_audio(source, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Decode audio from a path or file-like object to mono float32 at `sr`."""
    if isinstance(source, (str, Path)) and not Path(source).is_file():
        raise AudioValidationError(f"File not found: {source}")
    try:
        y, _ = librosa.load(source, sr=sr, mono=True)
    except Exception as exc:  # decoder errors vary by backend
        raise AudioValidationError(
            "Could not decode this audio file. Try a standard WAV file."
        ) from exc
    return y.astype(np.float32)


def validate_waveform(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> None:
    """Raise AudioValidationError if the waveform is unusable."""
    if y.size == 0:
        raise AudioValidationError("The audio file is empty.")
    if not np.all(np.isfinite(y)):
        raise AudioValidationError("The audio contains invalid (NaN or infinite) values.")
    seconds = len(y) / sr
    if seconds < config.MIN_AUDIO_SECONDS:
        raise AudioValidationError(
            f"The clip is too short ({seconds:.2f}s). Minimum is {config.MIN_AUDIO_SECONDS}s."
        )
    if seconds > config.MAX_AUDIO_SECONDS:
        raise AudioValidationError(
            f"The clip is too long ({seconds:.1f}s). Maximum is {config.MAX_AUDIO_SECONDS:.0f}s."
        )
    if float(np.max(np.abs(y))) < config.MIN_PEAK_AMPLITUDE:
        raise AudioValidationError("The recording appears to be silent.")


def trim_silence(y: np.ndarray) -> np.ndarray:
    """Trim leading and trailing silence. Falls back to the input if too much is removed."""
    trimmed, _ = librosa.effects.trim(y, top_db=config.TRIM_TOP_DB)
    if len(trimmed) < config.MIN_AUDIO_SECONDS * config.SAMPLE_RATE:
        return y
    return trimmed


def standardize_length(y: np.ndarray, n_samples: int = config.N_SAMPLES) -> np.ndarray:
    """Zero-pad at the end or truncate from the start to exactly n_samples."""
    if len(y) >= n_samples:
        return y[:n_samples].astype(np.float32)
    return np.pad(y, (0, n_samples - len(y))).astype(np.float32)


def prepare_waveform(source) -> tuple[np.ndarray, dict]:
    """Full pipeline: load, validate, trim, fix length.

    Returns the model-ready waveform and an info dict with
    original_seconds, trimmed_seconds and truncated.
    """
    y = load_audio(source)
    validate_waveform(y)
    trimmed = trim_silence(y)
    info = {
        "original_seconds": len(y) / config.SAMPLE_RATE,
        "trimmed_seconds": len(trimmed) / config.SAMPLE_RATE,
        "truncated": len(trimmed) > config.N_SAMPLES,
    }
    return standardize_length(trimmed), info

def augment_waveform(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random pitch shift, time stretch and additive noise. Training data only."""
    steps = rng.uniform(-config.PITCH_SHIFT_SEMITONES, config.PITCH_SHIFT_SEMITONES)
    rate = rng.uniform(*config.TIME_STRETCH_RANGE)
    out = librosa.effects.pitch_shift(y, sr=config.SAMPLE_RATE, n_steps=steps)
    out = librosa.effects.time_stretch(out, rate=rate)
    snr_db = rng.uniform(*config.NOISE_SNR_DB_RANGE)
    noise_power = (float(np.mean(out**2)) + 1e-12) / (10 ** (snr_db / 10))
    out = out + rng.normal(0.0, np.sqrt(noise_power), size=out.shape)
    return out.astype(np.float32)