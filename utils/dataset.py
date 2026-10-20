"""RAVDESS filename parsing and dataset indexing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config

_INTENSITY = {"01": "normal", "02": "strong"}


def parse_ravdess_filename(path: str | Path) -> dict:
    """Parse a RAVDESS filename such as 03-01-05-01-02-01-12.wav.

    Fields: modality, vocal channel, emotion, intensity, statement,
    repetition, actor. Raises ValueError for names that do not match.
    """
    path = Path(path)
    parts = path.stem.split("-")
    if len(parts) != 7 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Not a RAVDESS filename: {path.name}")

    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if emotion not in config.RAVDESS_EMOTION_CODES:
        raise ValueError(f"Unknown emotion code '{emotion}' in {path.name}")

    emotion_name = config.RAVDESS_EMOTION_CODES[emotion]
    return {
        "path": str(path),
        "filename": path.name,
        "modality": modality,
        "vocal_channel": channel,
        "emotion_code": emotion,
        "emotion_raw": emotion_name,
        "intensity": _INTENSITY.get(intensity, intensity),
        "statement": int(statement),
        "repetition": int(repetition),
        "actor": int(actor),
        "supported": emotion_name in config.LABEL_TO_ID,
    }


def build_dataset_index(data_dir: str | Path = config.DATA_DIR) -> pd.DataFrame:
    """Scan data_dir for speech .wav files and return one row per recording.

    Includes unsupported emotions (supported=False) so they can be counted
    before being filtered. Files that do not follow the RAVDESS naming
    scheme, or that are not audio-only speech, are skipped.
    """
    data_dir = Path(data_dir)
    rows = []
    for wav in sorted(data_dir.rglob("*.wav")):
        try:
            info = parse_ravdess_filename(wav)
        except ValueError:
            continue
        if (
            info["modality"] != config.AUDIO_ONLY_MODALITY
            or info["vocal_channel"] != config.SPEECH_VOCAL_CHANNEL
        ):
            continue
        info["path"] = project_relative(wav)
        rows.append(info)

    columns = [
        "path", "filename", "modality", "vocal_channel", "emotion_code",
        "emotion_raw", "intensity", "statement", "repetition", "actor", "supported",
    ]
    return pd.DataFrame(rows, columns=columns)


def filter_supported(index: pd.DataFrame) -> pd.DataFrame:
    """Keep only the four target emotions and add integer labels."""
    df = index[index["supported"]].copy()
    df["emotion"] = df["emotion_raw"]
    df["label_id"] = df["emotion"].map(config.LABEL_TO_ID).astype(int)
    return df.reset_index(drop=True)

def project_relative(path: str | Path) -> str:
    """Store dataset paths relative to the project root so the index is portable."""
    try:
        return Path(path).resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)  # dataset stored outside the project folder


def resolve_audio_path(path: str | Path) -> Path:
    """Inverse of project_relative: turn a stored path into a usable one."""
    path = Path(path)
    return path if path.is_absolute() else config.PROJECT_ROOT / path