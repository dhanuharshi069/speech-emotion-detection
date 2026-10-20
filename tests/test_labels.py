import pytest

import config
from utils.dataset import parse_ravdess_filename

from utils.dataset import project_relative, resolve_audio_path


def test_label_mapping_is_consistent():
    assert config.TARGET_EMOTIONS == ["angry", "happy", "neutral", "sad"]
    for name, idx in config.LABEL_TO_ID.items():
        assert config.ID_TO_LABEL[idx] == name


def test_parse_valid_filename():
    info = parse_ravdess_filename("Actor_12/03-01-05-02-01-02-12.wav")
    assert info["emotion_raw"] == "angry"
    assert info["intensity"] == "strong"
    assert info["actor"] == 12
    assert info["supported"] is True


def test_calm_is_unsupported():
    info = parse_ravdess_filename("03-01-02-01-01-01-03.wav")
    assert info["emotion_raw"] == "calm"
    assert info["supported"] is False


@pytest.mark.parametrize("bad", ["song.wav", "03-01-05-01-02-01.wav", "03-01-99-01-01-01-01.wav"])
def test_parse_invalid_filename_raises(bad):
    with pytest.raises(ValueError):
        parse_ravdess_filename(bad)

def test_project_relative_round_trip():
    inside = config.PROJECT_ROOT / "data" / "raw" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    stored = project_relative(inside)
    assert stored == "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    assert resolve_audio_path(stored) == inside


def test_project_relative_keeps_outside_paths(tmp_path):
    outside = tmp_path / "x.wav"
    assert resolve_audio_path(project_relative(outside)) == outside