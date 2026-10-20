import numpy as np
import pytest
import soundfile as sf

import config
from preprocessing import extract_features
from utils.audio_processing import (
    AudioValidationError, load_audio, prepare_waveform,
    standardize_length, validate_waveform,
)


def make_signal(seconds: float, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    rng = np.random.default_rng(config.SEED)
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def test_standardize_length_pads_and_truncates():
    assert len(standardize_length(make_signal(1.0))) == config.N_SAMPLES
    assert len(standardize_length(make_signal(5.0))) == config.N_SAMPLES


def test_feature_shapes():
    feats = extract_features(standardize_length(make_signal(2.0)))
    assert feats["logmel"].shape == (config.N_MELS, config.N_FRAMES)
    assert feats["mfcc"].shape == (config.N_MFCC, config.N_FRAMES)
    assert np.isfinite(feats["logmel"]).all() and np.isfinite(feats["mfcc"]).all()


def test_features_are_deterministic():
    y = standardize_length(make_signal(2.0))
    a, b = extract_features(y), extract_features(y)
    assert np.array_equal(a["logmel"], b["logmel"])


def test_extract_features_rejects_wrong_length():
    with pytest.raises(ValueError):
        extract_features(make_signal(1.0))


@pytest.mark.parametrize("y", [
    np.zeros(0, dtype=np.float32),
    np.zeros(config.SAMPLE_RATE, dtype=np.float32),            # silent
    make_signal(0.1),                                           # too short
    make_signal(config.MAX_AUDIO_SECONDS + 5),                  # too long
    np.full(config.SAMPLE_RATE, np.nan, dtype=np.float32),      # invalid
])
def test_validate_rejects_bad_audio(y):
    with pytest.raises(AudioValidationError):
        validate_waveform(y)


def test_load_missing_file():
    with pytest.raises(AudioValidationError):
        load_audio("does_not_exist.wav")


def test_load_invalid_file(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioValidationError):
        load_audio(bad)


def test_resamples_other_sample_rates(tmp_path):
    path = tmp_path / "clip16k.wav"
    sf.write(path, make_signal(2.0, sr=16000), 16000)
    y = load_audio(path)
    assert abs(len(y) - 2 * config.SAMPLE_RATE) <= 2


def test_prepare_waveform_output(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, make_signal(2.0), config.SAMPLE_RATE)
    y, info = prepare_waveform(path)
    assert len(y) == config.N_SAMPLES
    assert info["truncated"] is False