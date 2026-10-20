import numpy as np

import config
from utils.audio_processing import augment_waveform, standardize_length


def _clip():
    t = np.arange(int(2 * config.SAMPLE_RATE)) / config.SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_augment_is_finite_and_changes_signal():
    out = augment_waveform(_clip(), np.random.default_rng(0))
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert not np.array_equal(out[:1000], _clip()[:1000])


def test_augment_is_reproducible_with_same_seed():
    a = augment_waveform(_clip(), np.random.default_rng(1))
    b = augment_waveform(_clip(), np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_augmented_clip_standardizes_to_model_length():
    out = standardize_length(augment_waveform(_clip(), np.random.default_rng(2)))
    assert len(out) == config.N_SAMPLES