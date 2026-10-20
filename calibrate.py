"""Phase 4b: fit temperature scaling on VALIDATION data, report effect on val and test."""
import json
import sys

import numpy as np

import config
from preprocessing import apply_normalizer, load_normalizer
from utils.calibration import (
    apply_temperature, expected_calibration_error,
    fit_temperature, negative_log_likelihood,
)
from utils.data import load_features, split_indices

CALIBRATION_PATH = config.ARTIFACTS_DIR / "calibration.json"


def main() -> int:
    for p in (config.MODEL_PATH, config.NORMALIZATION_PATH, config.SPLIT_PATH, config.FEATURES_PATH):
        if not p.exists():
            print(f"Missing {p}. Run the earlier phases first.")
            return 1

    import keras

    split = json.loads(config.SPLIT_PATH.read_text())
    data = load_features()
    idx = split_indices(data["actor"], split)
    stats = load_normalizer()
    model = keras.models.load_model(config.MODEL_PATH)

    def predict(name):
        i = idx[name]
        X = apply_normalizer(data["X"][i], stats)
        return model.predict(X, verbose=0), data["y"][i]

    p_val, y_val = predict("val")
    p_test, y_test = predict("test")

    temperature = fit_temperature(p_val, y_val)  # fitted on validation only
    report = {"temperature": temperature, "fit_on": "validation"}

    print(f"Fitted temperature (validation only): {temperature:.3f}\n")
    print(f"{'split':6s} {'NLL before':>11s} {'NLL after':>10s} {'ECE before':>11s} "
          f"{'ECE after':>10s} {'mean conf before':>17s} {'after':>7s} {'accuracy':>9s}")
    for name, p, y in (("val", p_val, y_val), ("test", p_test, y_test)):
        q = apply_temperature(p, temperature)
        assert (q.argmax(axis=1) == p.argmax(axis=1)).all()
        row = {
            "accuracy": float((p.argmax(axis=1) == y).mean()),
            "nll_before": negative_log_likelihood(p, y),
            "nll_after": negative_log_likelihood(q, y),
            "ece_before": expected_calibration_error(p, y),
            "ece_after": expected_calibration_error(q, y),
            "mean_confidence_before": float(p.max(axis=1).mean()),
            "mean_confidence_after": float(q.max(axis=1).mean()),
        }
        report[name] = row
        print(f"{name:6s} {row['nll_before']:11.4f} {row['nll_after']:10.4f} "
              f"{row['ece_before']:11.4f} {row['ece_after']:10.4f} "
              f"{row['mean_confidence_before']:17.3f} {row['mean_confidence_after']:7.3f} "
              f"{row['accuracy']:9.4f}")

    CALIBRATION_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {CALIBRATION_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())