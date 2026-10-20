"""Shared styling and small UI helpers."""
from __future__ import annotations

import hashlib
import html
import inspect

import streamlit as st

import config

EMOTION_NOTES = {
    "angry": "Often louder, with more energy at higher frequencies and abrupt changes.",
    "happy": "Often higher in pitch, with more pitch variation and a brighter tone.",
    "neutral": "Often steady in energy and pitch, with little variation.",
    "sad": "Often quieter and lower in pitch, with slower and flatter delivery.",
}
if set(EMOTION_NOTES) != set(config.TARGET_EMOTIONS):
    raise RuntimeError("EMOTION_NOTES must match config.TARGET_EMOTIONS.")

_MIME = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg", "mp3": "audio/mpeg"}
_PLOTLY_CONFIG = {"displayModeBar": False}

_MONO = "'DM Mono', ui-monospace, Menlo, Consolas, monospace"
_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
:root {{
  --ink: #20211F; --muted: #686963; --border: #DEDED7; --bg: #F7F6F2;
  --surface: #FFFFFF; --warm: #EFEEE8; --accent: #356B52; --accent-dark: #2B5842;
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp table, .stApp small {{
  font-family: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{ color: var(--ink); }}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1120px !important; margin: 0 auto; padding: 3rem 2rem 4rem !important;
}}
.stApp h1 {{ font-size: 2.25rem; font-weight: 600; line-height: 1.15; letter-spacing: -0.01em; margin: 0 0 0.75rem; padding: 0; }}
.stApp h2 {{ font-size: 1.375rem; font-weight: 600; line-height: 1.25; margin: 2rem 0 0.75rem; padding: 0; }}
.stApp h3 {{ font-size: 1.0625rem; font-weight: 600; line-height: 1.3; }}
.lede {{ color: var(--muted); font-size: 1.0625rem; line-height: 1.6; max-width: 46rem; margin: 0 0 1.5rem; }}
.eyebrow {{ font-family: {_MONO}; font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.25rem; }}
.kv {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
.kv td {{ padding: 0.5rem 0; border-bottom: 1px solid var(--border); vertical-align: top; font-family: {_MONO}; }}
.kv td.k {{ color: var(--muted); width: 42%; padding-right: 1rem; }}
.emotions {{ width: 100%; border-collapse: collapse; }}
.emotions td {{ padding: 0.75rem 0; border-bottom: 1px solid var(--border); vertical-align: top; }}
.emotions td.name {{ width: 9rem; font-weight: 600; }}
.emotions td.desc {{ color: var(--muted); }}
.result {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; }}
.result .emotion {{ font-size: 2.5rem; font-weight: 600; line-height: 1.1; color: var(--accent); margin: 0.25rem 0 0.5rem; }}
.result .conf {{ font-family: {_MONO}; font-size: 0.9375rem; }}
.note {{ background: var(--warm); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.9375rem; line-height: 1.5; margin: 0.75rem 0; }}
.note strong {{ font-weight: 600; }}
.filename {{ font-weight: 500; margin-bottom: 0.5rem; word-break: break-all; }}
.stApp [data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {{
  background-color: var(--accent); color: #FFFFFF; border: 1px solid var(--accent); border-radius: 6px; font-weight: 500;
}}
.stApp [data-testid="stBaseButton-primary"]:hover, .stApp button[kind="primary"]:hover {{
  background-color: var(--accent-dark); border-color: var(--accent-dark); color: #FFFFFF;
}}
.stApp [data-testid="stBaseButton-primary"]:disabled, .stApp button[kind="primary"]:disabled {{
  background-color: var(--warm); border-color: var(--border); color: var(--muted); opacity: 1;
}}
.stApp [data-testid="stBaseButton-secondary"], .stApp button[kind="secondary"] {{
  background-color: var(--surface); color: var(--ink); border: 1px solid var(--border); border-radius: 6px; font-weight: 500;
}}
.stApp [data-testid="stBaseButton-secondary"]:hover, .stApp button[kind="secondary"]:hover {{
  background-color: var(--warm); border-color: #C9CAC2; color: var(--ink);
}}
.stApp :focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }}
.stApp [data-testid="stCaptionContainer"] {{ color: var(--muted); }}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; opacity: 1 !important; }}
.kv, .data, .emotions {{ border: 0 !important; }}
.kv td, .data th, .data td, .emotions td {{ border-width: 0 0 1px 0 !important; border-style: solid !important; border-color: var(--border) !important; background: transparent !important; }}
[data-testid="stPlotlyChart"] {{ border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: 1.5rem 1rem 3rem !important; }}
  .stApp h1 {{ font-size: 1.75rem; }}
  .result .emotion {{ font-size: 2rem; }}
  .emotions td.name {{ width: 6rem; }}
}}
"""


def inject_theme() -> None:
    css = f"""<style>{_CSS}
.table-wrap {{ overflow-x: auto; }}
.data {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
.data th {{ text-align: left; font-family: {_MONO}; font-weight: 500; font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); padding: 0.5rem 1rem 0.5rem 0; border-bottom: 1px solid var(--border); white-space: nowrap; }}
.data td {{ padding: 0.5rem 1rem 0.5rem 0; border-bottom: 1px solid var(--border); font-family: {_MONO}; white-space: nowrap; }}
.data td:first-child {{ font-family: 'DM Sans', system-ui, sans-serif; font-weight: 500; }}</style>"""
    if hasattr(st, "html"):
        st.html(css)
    else:
        st.markdown(css, unsafe_allow_html=True)


def show_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header(title: str, lede: str) -> None:
    show_html(f'<h1>{html.escape(title)}</h1><p class="lede">{html.escape(lede)}</p>')


def show_note(text: str, title: str | None = None) -> None:
    heading = f"<strong>{html.escape(title)}</strong><br>" if title else ""
    show_html(f'<div class="note">{heading}{html.escape(text)}</div>')


def kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<tr><td class="k">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k, v in rows
    )
    return f'<table class="kv">{body}</table>'


def emotion_table() -> str:
    body = "".join(
        f'<tr><td class="name">{html.escape(label.capitalize())}</td>'
        f'<td class="desc">{html.escape(EMOTION_NOTES[label])}</td></tr>'
        for label in config.TARGET_EMOTIONS
    )
    return f'<table class="emotions">{body}</table>'


def result_card(label: str, confidence: float) -> str:
    return (
        '<div class="result"><div class="eyebrow">Predicted expressed emotion</div>'
        f'<div class="emotion">{html.escape(label.capitalize())}</div>'
        f'<div class="conf">Confidence {confidence:.1%}</div></div>'
    )


def upload_hint() -> str:
    return (
        f"WAV is the most reliable format. FLAC, OGG and MP3 are accepted if they can be decoded. "
        f"Clips of {config.MIN_AUDIO_SECONDS:g} to {config.MAX_AUDIO_SECONDS:g} seconds, up to "
        f"{config.MAX_UPLOAD_MB} MB. The model analyzes up to {config.CLIP_DURATION:g} seconds "
        f"after trimming silence."
    )


def audio_mime(filename: str) -> str:
    return _MIME.get(filename.rsplit(".", 1)[-1].lower(), "audio/wav")


def remember_audio(name: str, data: bytes) -> None:
    """Store the selected audio. A different file clears any previous result."""
    digest = hashlib.sha1(data).hexdigest()
    current = st.session_state.get("audio")
    if current is None or current["digest"] != digest:
        st.session_state["audio"] = {"name": name, "bytes": data, "digest": digest}
        st.session_state.pop("result", None)
        st.session_state.pop("error", None)


def clear_audio() -> None:
    for key in ("audio", "result", "error"):
        st.session_state.pop(key, None)


def show_figure(fig) -> None:
    """Full-width Plotly chart that works across Streamlit versions."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, theme=None, width="stretch", config=_PLOTLY_CONFIG)
    else:
        st.plotly_chart(fig, theme=None, use_container_width=True, config=_PLOTLY_CONFIG)

def data_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (f'<div class="table-wrap"><table class="data"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


FEATURE_EXPLANATION = (
    "The model receives a log-Mel spectrogram: how energy is spread across 64 frequency bands "
    "over about three seconds. Patterns in this input, such as loudness, pitch movement, "
    "brightness and timing, are what it can learn from.",
    "MFCCs summarize the overall shape of the spectrum. They are shown here for inspection "
    "only. The model does not use them as input.",
    "These plots show the input, not the reason for a prediction. This app does not include a "
    "method that identifies which parts of the audio drove the model's decision, so no such "
    "conclusion should be drawn from them.",
)

