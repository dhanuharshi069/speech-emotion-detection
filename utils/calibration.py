"""Temperature scaling for probability calibration."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-12


def apply_temperature(probs, temperature: float) -> np.ndarray:
    """softmax(log(p) / T). Equivalent to scaling the original logits by 1/T.

    T > 1 softens the probabilities, T < 1 sharpens them. The argmax never changes.
    """
    logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def negative_log_likelihood(probs, y) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1.0))))


def fit_temperature(probs, y) -> float:
    """Find the temperature that minimizes NLL on the given (validation) data."""
    result = minimize_scalar(
        lambda t: negative_log_likelihood(apply_temperature(probs, t), y),
        bounds=(0.05, 10.0), method="bounded",
    )
    return float(result.x)


def expected_calibration_error(probs, y, n_bins: int = 10) -> float:
    probs, y = np.asarray(probs), np.asarray(y)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)