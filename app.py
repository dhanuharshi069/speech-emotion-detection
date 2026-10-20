"""Home dashboard and navigation. Run with: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

import config
from predict import get_dataset_info, get_model_status
from utils import ui

st.set_page_config(page_title="Emotion Detection from Speech", layout="wide")
ui.inject_theme()


def home() -> None:
    ui.page_header(
        "Emotion Detection from Speech",
        "Upload a short speech recording and see which of four emotions (angry, happy, neutral "
        "or sad) is expressed in the voice. A CNN-BiLSTM trained on the RAVDESS dataset makes "
        "the prediction.",
    )
    status = get_model_status()
    dataset = get_dataset_info()

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("## Analyze a recording")
        uploaded = st.file_uploader("Audio file", type=config.UPLOAD_TYPES, key="home_uploader")
        st.caption(ui.upload_hint())
        if uploaded is not None:
            data = uploaded.getvalue()
            ui.remember_audio(uploaded.name, data)
            st.audio(data, format=ui.audio_mime(uploaded.name))
        if not status["ready"]:
            st.error("Model files not found: " + ", ".join(status["missing"]) + ". Run `python train.py` first.")
        if st.button("Analyze audio", type="primary", key="home_analyze", disabled=not status["ready"]):
            if uploaded is None:
                ui.show_note("Choose an audio file above, then press Analyze audio.", title="No audio selected")
            else:
                st.session_state["auto_run"] = True
                st.switch_page("pages/1_Prediction.py")
        ui.show_note(
            "The result is a prediction of the emotion expressed in the voice. "
            "It does not measure how the speaker actually feels."
        )

    with right:
        cfg, metrics, cal = status["config"], status["metrics"], status["calibration"]
        st.markdown("## Model")
        ui.show_html(ui.kv_table([
            ("Status", "Ready" if status["ready"] else "Not found"),
            ("Architecture", cfg["architecture"] if cfg else "Not available"),
            ("Classes", ", ".join(config.TARGET_EMOTIONS)),
            ("Input", f"{config.N_MELS} Mel bands x {config.N_FRAMES} frames"),
            ("Test accuracy", f"{metrics['accuracy']:.1%}" if metrics else "Not evaluated"),
            ("Test macro-F1", f"{metrics['macro_f1']:.3f}" if metrics else "Not evaluated"),
            ("Test set", f"{metrics['n_test']} clips, {len(metrics['test_actors'])} unseen speakers"
             if metrics else "Not evaluated"),
            ("Confidence scaling",
             f"Temperature {cal['temperature']:.2f}" if cal and config.USE_CALIBRATION else "Off"),
        ]))
        st.markdown("## Dataset")
        if dataset:
            per_class = ", ".join(f"{k} {v}" for k, v in dataset["class_counts"].items())
            ui.show_html(ui.kv_table([
                ("Source", "RAVDESS, speech only"),
                ("Recordings used", str(dataset["recordings"])),
                ("Actors", str(dataset["actors"])),
                ("Per class", per_class),
            ]))
        else:
            ui.show_note("Run python inspect_dataset.py to index the dataset.", title="Dataset not indexed")

    st.markdown("## The four emotions")
    ui.show_html(ui.emotion_table())
    st.caption("These are general tendencies in how emotion is voiced. They are not rules the model is known to follow.")


pages = [st.Page(home, title="Home", default=True)]
for rel_path, title in (
    ("pages/1_Prediction.py", "Prediction"),
    ("pages/2_Model_Evaluation.py", "Model Evaluation"),
    ("pages/3_Project_Information.py", "Project Information"),
):
    if (config.PROJECT_ROOT / rel_path).exists():
        pages.append(st.Page(rel_path, title=title))
st.navigation(pages).run()