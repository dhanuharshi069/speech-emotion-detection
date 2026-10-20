"""Plotly figures in the project palette."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import config

INK, MUTED, BORDER, SURFACE = "#20211F", "#686963", "#DEDED7", "#FFFFFF"
ACCENT, OTHER_BAR, GRID = "#356B52", "#A5A69E", "#EEEEE8"
FONT = "DM Sans, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "DM Mono, ui-monospace, Menlo, Consolas, monospace"


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK),
    )
    fig.update_xaxes(linecolor=BORDER, gridcolor=GRID, zeroline=False, automargin=True,
                     tickfont=dict(family=MONO, size=12, color=MUTED))
    fig.update_yaxes(linecolor=BORDER, gridcolor=GRID, zeroline=False, automargin=True,
                     tickfont=dict(family=MONO, size=12, color=MUTED))
    return fig


def probability_chart(probabilities: dict, predicted: str) -> go.Figure:
    labels = list(config.TARGET_EMOTIONS)
    values = [probabilities[l] for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=[l.capitalize() for l in labels], orientation="h",
        marker_color=[ACCENT if l == predicted else OTHER_BAR for l in labels],
        text=[f"{v:.1%}" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(family=MONO, size=13, color=INK),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    _style(fig, 230)
    fig.update_xaxes(range=[0, 1.15], tickvals=[0, 0.25, 0.5, 0.75, 1.0], tickformat=".0%")
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     tickfont=dict(family=FONT, size=14, color=INK))
    return fig


def waveform_figure(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> go.Figure:
    step = max(1, len(y) // 3000)
    t = np.arange(len(y))[::step] / sr
    fig = go.Figure(go.Scatter(
        x=t, y=y[::step], mode="lines", line=dict(color=ACCENT, width=1),
        hovertemplate="%{x:.2f} s<extra></extra>",
    ))
    _style(fig, 200)
    fig.update_xaxes(title_text="Time (s)", range=[0, len(y) / sr])
    fig.update_yaxes(title_text="Amplitude", zeroline=True, zerolinecolor=BORDER)
    return fig


def mel_figure(logmel: np.ndarray, sr: int = config.SAMPLE_RATE,
               hop_length: int = config.HOP_LENGTH) -> go.Figure:
    n_bands, n_frames = logmel.shape
    fig = go.Figure(go.Heatmap(
        z=logmel, x=np.arange(n_frames) * hop_length / sr, y=np.arange(n_bands),
        colorscale=[[0.0, "#F7F6F2"], [1.0, ACCENT]],
        colorbar=dict(title=dict(text="dB", font=dict(family=MONO, size=12)),
                      thickness=12, outlinewidth=0, tickfont=dict(family=MONO, size=11)),
        hovertemplate="%{x:.2f} s, band %{y}: %{z:.0f} dB<extra></extra>",
    ))
    _style(fig, 260)
    fig.update_xaxes(title_text="Time (s)", showgrid=False)
    fig.update_yaxes(title_text="Mel band (low to high)", showgrid=False)
    return fig

def confusion_matrix_figure(cm, labels) -> go.Figure:
    cm = np.asarray(cm)
    share = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    names = [l.capitalize() for l in labels]
    fig = go.Figure(go.Heatmap(
        z=share, x=names, y=names, zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
        colorscale=[[0.0, "#FFFFFF"], [1.0, ACCENT]], customdata=cm,
        hovertemplate="True %{y}, predicted %{x}: %{customdata} clips (%{z:.0%} of true)<extra></extra>",
    ))
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(
                x=names[j], y=names[i], text=str(int(cm[i, j])), showarrow=False,
                font=dict(family=MONO, size=14, color="#FFFFFF" if share[i, j] > 0.5 else INK),
            )
    _style(fig, 340)
    fig.update_xaxes(title_text="Predicted", showgrid=False, tickfont=dict(family=FONT, size=13, color=INK))
    fig.update_yaxes(title_text="True", autorange="reversed", showgrid=False,
                     tickfont=dict(family=FONT, size=13, color=INK))
    return fig


def history_figure(history: dict, key: str, axis_title: str, selected_epoch: int | None = None) -> go.Figure:
    train, val = history.get(key, []), history.get(f"val_{key}", [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(train) + 1)), y=train, name="Training",
                             mode="lines", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=list(range(1, len(val) + 1)), y=val, name="Validation",
                             mode="lines", line=dict(color=INK, width=2, dash="dot")))
    if selected_epoch:
        fig.add_vline(x=selected_epoch, line_width=1, line_dash="dash", line_color=MUTED)
    _style(fig, 300)
    fig.update_layout(
        showlegend=True, margin=dict(l=8, r=8, t=32, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(family=FONT, size=13)),
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text=axis_title)
    return fig


def mfcc_figure(mfcc: np.ndarray, sr: int = config.SAMPLE_RATE,
                hop_length: int = config.HOP_LENGTH) -> go.Figure:
    n_coef, n_frames = mfcc.shape
    fig = go.Figure(go.Heatmap(
        z=mfcc, x=np.arange(n_frames) * hop_length / sr, y=np.arange(n_coef),
        colorscale=[[0.0, "#F7F6F2"], [1.0, ACCENT]],
        colorbar=dict(title=dict(text="value", font=dict(family=MONO, size=12)),
                      thickness=12, outlinewidth=0, tickfont=dict(family=MONO, size=11)),
        hovertemplate="%{x:.2f} s, coefficient %{y}: %{z:.1f}<extra></extra>",
    ))
    _style(fig, 260)
    fig.update_xaxes(title_text="Time (s)", showgrid=False)
    fig.update_yaxes(title_text="MFCC coefficient", showgrid=False)
    return fig