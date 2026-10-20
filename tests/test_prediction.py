import numpy as np
import pytest
import soundfile as sf

import config
from model import build_model
from predict import EmotionPredictor, ModelNotFoundError
from preprocessing import fit_normalizer, save_normalizer
from utils.audio_processing import AudioValidationError


def _signal(seconds: float, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    rng = np.random.default_rng(config.SEED)
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    """A predictor built from an untrained model, to test the inference path only."""
    d = tmp_path_factory.mktemp("model_files")
    build_model().save(d / "model.keras")
    rng = np.random.default_rng(config.SEED)
    stats = fit_normalizer(rng.normal(-40, 10, (20, config.N_MELS, config.N_FRAMES)).astype(np.float32))
    save_normalizer(stats, d / "norm.joblib")
    return EmotionPredictor(
        model_path=d / "model.keras", normalizer_path=d / "norm.joblib",
        calibration_path=None, model_config_path=None,
    )


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, _signal(2.0), config.SAMPLE_RATE)
    return path


def test_prediction_is_a_valid_distribution(predictor, wav):
    pred = predictor.predict_source(wav)
    assert list(pred.probabilities) == config.TARGET_EMOTIONS
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert pred.label == max(pred.probabilities, key=pred.probabilities.get)
    assert pred.confidence == pytest.approx(pred.probabilities[pred.label])
    assert pred.features["logmel"].shape == (config.N_MELS, config.N_FRAMES)
    assert len(pred.waveform) == config.N_SAMPLES


def test_prediction_is_deterministic(predictor, wav):
    a, b = predictor.predict_source(wav), predictor.predict_source(wav)
    assert a.probabilities == b.probabilities


def test_temperature_changes_confidence_not_label(predictor, wav):
    base = predictor.predict_source(wav)
    predictor.temperature = 3.0
    try:
        softened = predictor.predict_source(wav)
    finally:
        predictor.temperature = 1.0
    assert softened.label == base.label
    assert softened.confidence <= base.confidence + 1e-9


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        EmotionPredictor(
            model_path=tmp_path / "nope.keras", normalizer_path=tmp_path / "nope.joblib",
            calibration_path=None, model_config_path=None,
        )


def test_invalid_audio_raises(predictor, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioValidationError):
        predictor.predict_source(bad)


def test_silent_audio_raises(predictor, tmp_path):
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(config.SAMPLE_RATE, dtype=np.float32), config.SAMPLE_RATE)
    with pytest.raises(AudioValidationError):
        predictor.predict_source(silent)


def test_other_sample_rates_are_handled(predictor, tmp_path):
    path = tmp_path / "clip16k.wav"
    sf.write(path, _signal(2.0, sr=16000), 16000)
    assert predictor.predict_source(path).label in config.TARGET_EMOTIONS