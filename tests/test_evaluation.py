import numpy as np
import pytest

from evaluate import compute_metrics, wilson_interval


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert np.trace(np.array(m["confusion_matrix"])) == len(y)


def test_known_confusion_matrix_and_class_metrics():
    # label order: angry=0, happy=1, neutral=2, sad=3
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]
    assert m["confusion_matrix"][2] == [1, 0, 1, 0]
    assert m["per_class"]["happy"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["happy"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["angry"]["precision"] == pytest.approx(0.5)


def test_wilson_interval_properties():
    lo, hi = wilson_interval(8, 10)
    assert 0 <= lo < 0.8 < hi <= 1
    lo2, hi2 = wilson_interval(80, 100)
    assert (hi2 - lo2) < (hi - lo)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    