import numpy as np

import config
from utils import visualizations as viz


def test_probability_chart_order_and_highlight():
    fig = viz.probability_chart({l: 0.25 for l in config.TARGET_EMOTIONS}, "happy")
    bar = fig.data[0]
    assert list(bar.y) == [l.capitalize() for l in config.TARGET_EMOTIONS]
    assert bar.marker.color[config.TARGET_EMOTIONS.index("happy")] == viz.ACCENT


def test_confusion_matrix_figure_matches_input():
    cm = [[3, 1, 0, 0], [0, 4, 0, 0], [1, 0, 2, 1], [0, 0, 0, 4]]
    fig = viz.confusion_matrix_figure(cm, config.TARGET_EMOTIONS)
    assert np.allclose(np.asarray(fig.data[0].z).sum(axis=1), 1.0)
    assert np.array_equal(fig.data[0].customdata, cm)
    assert len(fig.layout.annotations) == 16


def test_history_figure_marks_selected_epoch():
    history = {"accuracy": [0.4, 0.5, 0.6], "val_accuracy": [0.3, 0.4, 0.5]}
    fig = viz.history_figure(history, "accuracy", "Accuracy", selected_epoch=2)
    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 1


def test_feature_figures_build():
    assert viz.waveform_figure(np.zeros(config.N_SAMPLES, dtype=np.float32)) is not None
    logmel = np.zeros((config.N_MELS, config.N_FRAMES), dtype=np.float32)
    mfcc = np.zeros((config.N_MFCC, config.N_FRAMES), dtype=np.float32)
    assert viz.mel_figure(logmel).data[0].z.shape == logmel.shape
    assert viz.mfcc_figure(mfcc).data[0].z.shape == mfcc.shape

def test_mfcc_display_is_scaled_per_coefficient():
    rng = np.random.default_rng(0)
    scale = np.arange(1, config.N_MFCC + 1)[:, None]
    mfcc = (rng.normal(0, 1, (config.N_MFCC, config.N_FRAMES)) * scale).astype(np.float32)
    z = np.asarray(viz.mfcc_figure(mfcc).data[0].z)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-3)
    assert np.allclose(z.std(axis=1), 1, atol=1e-3)