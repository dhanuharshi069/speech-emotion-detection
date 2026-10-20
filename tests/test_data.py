import numpy as np
import pytest

import config
from model import build_model
from preprocessing import apply_normalizer, fit_normalizer
from utils.data import assert_no_leakage, make_actor_split, split_indices

ACTORS = np.repeat(np.arange(1, 25), 28)


def test_split_is_disjoint_and_complete():
    split = make_actor_split(ACTORS)
    assert_no_leakage(split)
    assert sorted(split["train"] + split["val"] + split["test"]) == list(range(1, 25))
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (16, 4, 4)


def test_split_is_gender_balanced():
    split = make_actor_split(ACTORS)
    for name in ("val", "test"):
        assert sum(a % 2 for a in split[name]) == 2


def test_split_is_reproducible():
    assert make_actor_split(ACTORS) == make_actor_split(ACTORS)


def test_split_indices_cover_all_samples():
    idx = split_indices(ACTORS, make_actor_split(ACTORS))
    assert sum(len(v) for v in idx.values()) == len(ACTORS)


def test_leakage_is_detected():
    with pytest.raises(ValueError):
        assert_no_leakage({"train": [1, 2], "val": [2], "test": [3]})


def test_normalizer_shapes_and_statistics():
    rng = np.random.default_rng(config.SEED)
    X = rng.normal(5, 2, (50, config.N_MELS, config.N_FRAMES)).astype(np.float32)
    out = apply_normalizer(X, fit_normalizer(X))
    assert out.shape == (50, config.N_MELS, config.N_FRAMES, 1)
    assert abs(out.mean()) < 1e-3 and abs(out.std() - 1) < 1e-2


def test_model_outputs_probability_distribution():
    model = build_model()
    out = model.predict(np.zeros((2, config.N_MELS, config.N_FRAMES, 1), dtype=np.float32), verbose=0)
    assert out.shape == (2, len(config.TARGET_EMOTIONS))
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)