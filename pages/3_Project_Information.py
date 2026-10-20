"""Project information. Facts come from config.py and the saved result files."""
from __future__ import annotations

import streamlit as st

import config
from predict import get_dataset_info
from utils import ui
from utils.results import load_json, split_summary

ui.page_header(
    "Project Information",
    "How the model was built, what it was trained on, and where its limits are.",
)

metrics = load_json(config.EVALUATION_METRICS_PATH)
dataset = get_dataset_info()
splits = split_summary()

st.markdown("## Problem statement")
st.markdown(
    "The way something is said carries information about emotion, through pitch, loudness, "
    "rhythm and tone. This project predicts which of four emotions, angry, happy, neutral or "
    "sad, is expressed in a short speech clip. The result describes the emotion expressed in the "
    "recording. It is not a measure of how the speaker actually feels."
)

st.markdown("## Objectives")
st.markdown(
    "- Build a reproducible pipeline from raw audio to a trained model, with a fixed random seed.\n"
    "- Evaluate on speakers the model never saw, and report only measured results.\n"
    "- Offer an interface to upload a clip, see the predicted emotion with approximate confidence, "
    "and inspect the input features."
)

st.markdown("## Dataset")
st.markdown(
    "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains acted "
    "speech by 24 professional actors, 12 female and 12 male, speaking two sentences in a neutral "
    "North American accent. This project uses the speech recordings for four of its eight "
    "emotions. Calm, fearful, disgust and surprised are excluded. Neutral has only one intensity "
    "level in the dataset, so it has half as many recordings as the other classes."
)
rows = [("Source", "RAVDESS, speech only (Livingstone and Russo, 2018)"), ("License", "CC BY-NC-SA 4.0")]
if dataset:
    rows += [
        ("Recordings used", str(dataset["recordings"])),
        ("Actors", str(dataset["actors"])),
        ("Per class", ", ".join(f"{k} {v}" for k, v in dataset["class_counts"].items())),
    ]
ui.show_html(ui.kv_table(rows))
if splits:
    st.markdown("### Split by speaker")
    names = {"train": "Train", "val": "Validation", "test": "Test"}
    ui.show_html(ui.data_table(
        ["Split", "Speakers", "Clips", "Actor IDs"],
        [[names.get(k, k), str(len(v["actors"])), str(v["clips"]), ", ".join(map(str, v["actors"]))]
         for k, v in splits.items()],
    ))
    st.caption("Validation and test speakers are balanced by gender. No speaker appears in more than one split.")

st.markdown("## Feature extraction")
st.markdown(
    "Each clip is resampled to mono, trimmed of leading and trailing silence, then padded or cut "
    "to a fixed length. The model input is a log-Mel spectrogram, normalized per frequency band "
    "using statistics from the training speakers only. MFCCs are computed for inspection but are "
    "not used by the model."
)
ui.show_html(ui.kv_table([
    ("Sample rate", f"{config.SAMPLE_RATE} Hz, mono"),
    ("Silence trimming", f"{config.TRIM_TOP_DB} dB below peak"),
    ("Clip length", f"{config.CLIP_DURATION:g} s, padded or cut"),
    ("Log-Mel spectrogram", f"{config.N_MELS} bands, FFT {config.N_FFT}, hop {config.HOP_LENGTH}"),
    ("Model input", f"{config.N_MELS} x {config.N_FRAMES}, normalized per band"),
    ("MFCCs", f"{config.N_MFCC} coefficients, inspection only"),
]))

st.markdown("## Model architecture")
st.markdown(
    "Three convolutional blocks learn local patterns in the spectrogram. A bidirectional LSTM "
    "then reads the result over time, and a small dense layer outputs probabilities for the four "
    "emotions."
)
ui.show_html(ui.kv_table([
    ("Input", f"{config.N_MELS} x {config.N_FRAMES} x 1 log-Mel spectrogram"),
    ("Convolution", "3 blocks of 32, 64 and 128 filters (3x3), batch norm, ReLU, max pooling, dropout"),
    ("Channel reduction", "1x1 convolution to 32 channels"),
    ("Sequence layer", "Bidirectional LSTM, 64 units per direction"),
    ("Classifier", "Dense 64, dropout, softmax over 4 classes"),
    ("Parameters", "270,340"),
]))

st.markdown("## Training workflow")
low, high = config.TIME_STRETCH_RANGE
st.markdown(
    "1. `inspect_dataset.py` indexes the recordings and keeps the four target emotions.\n"
    "2. `build_features.py` extracts and caches log-Mel and MFCC features.\n"
    f"3. `augment_features.py` makes {config.AUG_COPIES} augmented copies of each training clip, "
    f"using pitch shifts up to {config.PITCH_SHIFT_SEMITONES:g} semitones, time stretching from "
    f"{low:g} to {high:g} and added noise. Validation and test clips are never augmented.\n"
    f"4. `train.py` splits by speaker and trains with Adam (learning rate {config.LEARNING_RATE:g}), "
    f"batch size {config.BATCH_SIZE}, class weights for the smaller neutral class, and random "
    "frequency and time masking. The epoch with the best validation macro-F1 is kept, and "
    f"training stops after {config.EARLY_STOP_PATIENCE} epochs without improvement.\n"
    "5. `evaluate.py` evaluates the final model once on the held-out speakers.\n"
    "6. `calibrate.py` fits a temperature on the validation set so that displayed confidence is "
    "less overconfident."
)

st.markdown("## Limitations and future improvements")
limits = [
    "The data is acted speech from 24 actors with a North American accent, recorded in a studio. "
    "Performance on spontaneous speech, other languages, other accents or noisy recordings has "
    "not been measured and may be much lower.",
]
if metrics:
    acc = metrics["per_actor_accuracy"]
    limits.append(
        f"The test set has {metrics['n_test']} clips from {len(metrics['test_actors'])} speakers, "
        f"so the estimates are uncertain. Accuracy by speaker ranges from {min(acc.values()):.1%} "
        f"to {max(acc.values()):.1%}."
    )
neutral_total = dataset["class_counts"].get("neutral") if dataset else None
limits += [
    "Neutral has the least data" + (f" ({neutral_total} recordings in total)" if neutral_total else "")
    + ", and its results are the least stable.",
    "The training epoch and the confidence temperature were chosen using four validation "
    "speakers, so validation figures are optimistic and the calibration is approximate.",
    "Four categories simplify real emotion. The labels reflect what actors were asked to express, "
    "not what they felt.",
    f"Only the first {config.CLIP_DURATION:g} seconds of a clip after trimming silence are analyzed.",
    "Spectrogram and MFCC plots show the input, not the reasons for a prediction.",
]
st.markdown("\n".join(f"- {item}" for item in limits))

st.markdown("### Possible improvements")
st.markdown(
    "- Cross-validation by speaker for more stable performance estimates.\n"
    "- More varied data: other corpora, languages and recording conditions.\n"
    "- Fine-tuning a pretrained speech model instead of training from scratch.\n"
    "- Explanation methods such as Grad-CAM or occlusion, checked for whether they reflect what "
    "the model actually uses.\n"
    "- Windowed analysis of longer recordings."
)

if st.button("Analyze an audio clip", type="primary", key="info_to_prediction"):
    st.switch_page("pages/1_Prediction.py")