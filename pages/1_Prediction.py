"""Prediction page: upload, validate, analyze and inspect a speech clip."""
from __future__ import annotations

import html
import io

import streamlit as st

import config
from predict import EmotionPredictor, ModelNotFoundError, Prediction
from utils import ui
from utils import visualizations as viz
from utils.audio_processing import AudioValidationError

ui.page_header(
    "Prediction",
    "Upload a short speech clip to see which emotion the model detects in the way it is spoken.",
)


@st.cache_resource(show_spinner="Loading model")
def load_predictor() -> EmotionPredictor:
    return EmotionPredictor()


def get_predictor() -> EmotionPredictor:
    try:
        return load_predictor()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.code("python train.py", language="bash")
    except Exception as exc:  # corrupt or incompatible model file
        st.error(f"The model could not be loaded: {exc}")
    st.stop()


def analyze(predictor: EmotionPredictor, audio: dict) -> None:
    try:
        with st.spinner("Analyzing audio"):
            prediction = predictor.predict_source(io.BytesIO(audio["bytes"]))
    except AudioValidationError as exc:
        st.session_state["error"] = str(exc)
        st.session_state.pop("result", None)
    except Exception as exc:
        st.session_state["error"] = f"Analysis failed: {exc}"
        st.session_state.pop("result", None)
    else:
        st.session_state["result"] = {"digest": audio["digest"], "prediction": prediction}
        st.session_state.pop("error", None)


def render_result(pred: Prediction) -> None:
    st.markdown("## Result")
    left, right = st.columns([2, 3], gap="large")
    with left:
        ui.show_html(ui.result_card(pred.label, pred.confidence))
    with right:
        ui.show_figure(viz.probability_chart(pred.probabilities, pred.label))

    if pred.confidence < config.LOW_CONFIDENCE_THRESHOLD:
        ui.show_note(
            "The probabilities are spread across several emotions, so treat this result as uncertain.",
            title="Low confidence",
        )
    if pred.info["truncated"]:
        st.warning(
            f"After trimming silence this clip is longer than {config.CLIP_DURATION:g} seconds. "
            f"Only the first {config.CLIP_DURATION:g} seconds were analyzed."
        )
    ui.show_note(
        "This is a prediction of the emotion expressed in the voice. "
        "It does not measure how the speaker actually feels."
    )

    st.markdown("## Feature views")
    st.caption("These views show the processed clip the model analyzed: silence trimmed, fixed length.")
    ui.show_figure(viz.waveform_figure(pred.waveform))
    if st.checkbox("Show Mel spectrogram", key="show_mel"):
        ui.show_figure(viz.mel_figure(pred.features["logmel"]))
    if st.checkbox("Show MFCCs", key="show_mfcc"):
        ui.show_figure(viz.mfcc_figure(pred.features["mfcc"]))
    with st.expander("What the model analyzes"):
        for paragraph in ui.FEATURE_EXPLANATION:
            st.markdown(paragraph)

    st.markdown("## Processing details")
    info = pred.info
    ui.show_html(ui.kv_table([
        ("Original duration", f"{info['original_seconds']:.2f} s"),
        ("After silence trimming", f"{info['trimmed_seconds']:.2f} s"),
        ("Analyzed length", f"{config.CLIP_DURATION:.1f} s at {config.SAMPLE_RATE} Hz"),
        ("Feature shape", f"{config.N_MELS} Mel bands x {config.N_FRAMES} frames"),
        ("Confidence scaling",
         "Off" if pred.temperature == 1.0 else f"Temperature {pred.temperature:.2f}"),
    ]))
    if pred.temperature != 1.0:
        st.caption(
            "Confidence scores are rescaled with a temperature fitted on four validation "
            "speakers. Treat them as approximate."
        )


predictor = get_predictor()
auto_run = st.session_state.pop("auto_run", False)

uploader_n = st.session_state.get("uploader_n", 0)
uploaded = st.file_uploader("Audio file", type=config.UPLOAD_TYPES, key=f"pred_uploader_{uploader_n}")
st.caption(ui.upload_hint())
if uploaded is not None:
    ui.remember_audio(uploaded.name, uploaded.getvalue())

audio = st.session_state.get("audio")
if audio is None:
    ui.show_note("Choose a WAV, FLAC, OGG or MP3 file above to begin.", title="No audio selected")
    st.stop()

st.markdown("## Selected file")
ui.show_html(f'<div class="filename">{html.escape(audio["name"])}</div>')
st.audio(audio["bytes"], format=ui.audio_mime(audio["name"]))

analyze_col, remove_col, _ = st.columns([2, 2, 6])
run = analyze_col.button("Analyze audio", type="primary", key="analyze")
if remove_col.button("Remove file", key="remove"):
    ui.clear_audio()
    st.session_state["uploader_n"] = uploader_n + 1
    st.rerun()

if run or auto_run:
    analyze(predictor, audio)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

result = st.session_state.get("result")
if result and result["digest"] == audio["digest"]:
    render_result(result["prediction"])
elif not st.session_state.get("error"):
    ui.show_note("Press Analyze audio to run the model on this file.")