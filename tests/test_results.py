from utils.results import load_json, selected_epoch, split_summary


def test_selected_epoch_is_first_maximum():
    assert selected_epoch({"val_macro_f1": [0.2, 0.5, 0.5, 0.4]}) == 2


def test_selected_epoch_without_scores():
    assert selected_epoch({}) is None


def test_load_json_missing_file_returns_none(tmp_path):
    assert load_json(tmp_path / "nope.json") is None


def test_load_json_reads_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"a": 1}')
    assert load_json(path) == {"a": 1}


def test_split_summary_counts_clips(tmp_path):
    import json
    import pandas as pd

    pd.DataFrame({"actor": [1, 1, 2, 3, 3, 3]}).to_csv(tmp_path / "idx.csv", index=False)
    (tmp_path / "split.json").write_text(json.dumps({"train": [1, 2], "val": [3], "test": []}))
    out = split_summary(tmp_path / "idx.csv", tmp_path / "split.json")
    assert out["train"]["clips"] == 3 and out["val"]["clips"] == 3 and out["test"]["clips"] == 0


def test_split_summary_missing_files(tmp_path):
    assert split_summary(tmp_path / "a.csv", tmp_path / "b.json") is None