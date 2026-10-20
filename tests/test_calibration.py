import numpy as np
import pytest

from utils.calibration import (
    apply_temperature, expected_calibration_error,
    fit_temperature, negative_log_likelihood,
)


def test_temperature_one_is_identity():
    p = np.array([[0.7, 0.1, 0.1, 0.1], [0.05, 0.15, 0.6, 0.2]])
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-9)


def test_rows_sum_to_one_and_argmax_preserved():
    p = np.array([[0.7, 0.1, 0.1, 0.1], [0.05, 0.15, 0.6, 0.2]])
    q = apply_temperature(p, 2.5)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert (q.argmax(axis=1) == p.argmax(axis=1)).all()
    assert q.max(axis=1)[0] < p.max(axis=1)[0]


def test_fit_temperature_softens_overconfident_predictions():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 4, n)
    pred = np.where(rng.random(n) < 0.5, y, rng.integers(0, 4, n))
    p = np.full((n, 4), 0.01)
    p[np.arange(n), pred] = 0.97
    t = fit_temperature(p, y)
    assert t > 1.0
    assert negative_log_likelihood(apply_temperature(p, t), y) < negative_log_likelihood(p, y)


def test_ece_is_zero_for_confident_correct_predictions():
    assert expected_calibration_error(np.eye(4), np.arange(4)) == pytest.approx(0.0)