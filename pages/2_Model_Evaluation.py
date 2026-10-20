"""Model evaluation. Shows only results saved by train.py, evaluate.py and calibrate.py."""
from __future__ import annotations

import streamlit as st

import config
from utils import ui
from utils import visualizations as viz
from utils.results import load_results, selected_epoch

results = load_results()
metrics, history, calibration = results["metrics"], results["history"], results["calibration"]

if metrics is None:
    ui.page_header("Model Evaluation", "Test results are not available yet.")
    ui.show_note("Run python evaluate.py to generate the test results.", title="No evaluation results found")
    st.stop()

ui.page_header(
    "Model Evaluation",
    f"Results on {metrics['n_test']} clips from {len(metrics['test_actors'])} speakers. "
    "The split is by speaker, so none of these voices were seen during training or validation.",
)

# Headline numbers
st.markdown("## Test performance")
lo, hi = metrics["accuracy_ci95"]
left, right = st.columns(2, gap="large")
with left:
    ui.show_html(ui.kv_table([
        ("Accuracy", f"{metrics['accuracy']:.1%}"),
        ("95% interval", f"{lo:.1%} to {hi:.1%}"),
        ("Macro precision", f"{metrics['macro_precision']:.3f}"),
        ("Macro recall", f"{metrics['macro_recall']:.3f}"),
        ("Macro F1", f"{metrics['macro_f1']:.3f}"),
        ("Weighted F1", f"{metrics['weighted_f1']:.3f}"),
        ("Test clips", str(metrics["n_test"])),
        ("Test speakers", ", ".join(str(a) for a in metrics["test_actors"])),
    ]))
with right:
    ui.show_note(
        "The interval treats clips as independent. Clips from the same speaker are correlated "
        "and there are only a few test speakers, so the real uncertainty is larger.",
        title="How to read this",
    )

# Confusion matrix and per-class results
st.markdown("## Confusion matrix and per-class results")
left, right = st.columns(2, gap="large")
with left:
    ui.show_figure(viz.confusion_matrix_figure(metrics["confusion_matrix"], metrics["labels"]))
    st.caption("Rows are true emotions, columns are predicted emotions. Cell color shows the share of each true class.")
with right:
    per_class = metrics["per_class"]
    ui.show_html(ui.data_table(
        ["Emotion", "Precision", "Recall", "F1", "Support"],
        [[name.capitalize(), f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", str(m["support"])]
         for name, m in per_class.items()],
    ))
    smallest = min(per_class, key=lambda k: per_class[k]["support"])
    st.caption(f"{smallest.capitalize()} has the fewest test clips ({per_class[smallest]['support']}), "
               "so its figures are the least stable.")

# Per-speaker accuracy
st.markdown("## Accuracy by speaker")
speaker_acc = metrics["per_actor_accuracy"]
ui.show_html(ui.data_table(
    ["Speaker", "Accuracy"], [[f"Actor {a}", f"{v:.1%}"] for a, v in speaker_acc.items()],
))
st.caption(f"Accuracy by speaker ranges from {min(speaker_acc.values()):.1%} to "
           f"{max(speaker_acc.values()):.1%}, so results on new voices will vary.")

# Training history
st.markdown("## Training history")
if history:
    epoch = selected_epoch(history)
    left, right = st.columns(2, gap="large")
    with left:
        ui.show_figure(viz.history_figure(history, "accuracy", "Accuracy", epoch))
    with right:
        ui.show_figure(viz.history_figure(history, "loss", "Loss", epoch))
    rows = [("Epochs run", str(len(history["loss"])))]
    if epoch:
        rows += [
            ("Selected epoch", str(epoch)),
            ("Validation accuracy there", f"{history['val_accuracy'][epoch - 1]:.1%}"),
            ("Validation macro-F1 there", f"{history['val_macro_f1'][epoch - 1]:.3f}"),
        ]
    ui.show_html(ui.kv_table(rows))
    st.caption(
        "Training values are measured with dropout and augmented data, so they are not directly "
        "comparable to validation values. The dashed line marks the epoch with the highest "
        "validation macro-F1, whose weights were kept. That choice was made on validation data, "
        "so validation figures are optimistic. The test results above are the unbiased estimate."
    )
else:
    ui.show_note("Run python train.py to generate the training history.", title="No training history found")

# Calibration
if calibration:
    st.markdown("## Confidence calibration")
    names = {"val": "Validation", "test": "Test"}
    rows = []
    for key in ("val", "test"):
        c = calibration.get(key)
        if c:
            rows.append([
                names[key],
                f"{c['nll_before']:.3f} to {c['nll_after']:.3f}",
                f"{c['ece_before']:.3f} to {c['ece_after']:.3f}",
                f"{c['mean_confidence_before']:.3f} to {c['mean_confidence_after']:.3f}",
                f"{c['accuracy']:.3f}",
            ])
    ui.show_html(ui.data_table(["Split", "Log loss", "ECE", "Mean confidence", "Accuracy"], rows))
    st.caption(
        f"Each pair shows values before to after temperature scaling. The temperature "
        f"({calibration['temperature']:.2f}) was fitted on the validation set only. Scaling changes "
        "the displayed confidence, not the predicted emotion. Test values are the independent check."
    )

# Primary action
st.markdown("## Try the model")
if st.button("Analyze an audio clip", type="primary", key="eval_to_prediction"):
    st.switch_page("pages/1_Prediction.py")