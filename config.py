"""Central configuration. Every other module imports labels and paths from here."""
from pathlib import Path

SEED = 42

# All paths are resolved from the project root, so scripts work from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_INDEX_PATH = ARTIFACTS_DIR / "dataset_index.csv"
LABEL_MAPPING_PATH = ARTIFACTS_DIR / "label_mapping.json"

# RAVDESS filename field 3 (emotion code) -> emotion name.
RAVDESS_EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# The four supported classes. This order defines the model output index
# and must be reused by training, evaluation, the probability chart and the UI.
TARGET_EMOTIONS = ["angry", "happy", "neutral", "sad"]
LABEL_TO_ID = {name: i for i, name in enumerate(TARGET_EMOTIONS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

# RAVDESS filename fields we accept: audio-only (03), speech (01).
AUDIO_ONLY_MODALITY = "03"
SPEECH_VOCAL_CHANNEL = "01"

# ---------------------------------------------------------------------------
# Audio and feature settings (shared by training and inference)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
CLIP_DURATION = 3.0                      # seconds, fixed model input length
N_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

MIN_AUDIO_SECONDS = 0.5                  # reject clips shorter than this
MAX_AUDIO_SECONDS = 30.0                 # reject uploads longer than this
MIN_PEAK_AMPLITUDE = 1e-4                # below this the file is treated as silent
TRIM_TOP_DB = 30

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
N_MFCC = 40
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH   # time steps per feature matrix

FEATURES_PATH = ARTIFACTS_DIR / "features.npz"
PREPROCESSING_CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"

# ---------------------------------------------------------------------------
# Splitting and training
# ---------------------------------------------------------------------------
VAL_ACTORS_PER_GENDER = 2
TEST_ACTORS_PER_GENDER = 2

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 12
DROPOUT = 0.4
FREQ_MASK_MAX = 8      # max Mel bands masked per sample (training only)
TIME_MASK_MAX = 30     # max frames masked per sample (training only)
CONV_DROPOUT = 0.15


MODEL_PATH = MODELS_DIR / "emotion_model.keras"
NORMALIZATION_PATH = ARTIFACTS_DIR / "normalization.joblib"
SPLIT_PATH = ARTIFACTS_DIR / "split.json"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"

AUG_COPIES = 4
PITCH_SHIFT_SEMITONES = 1.5
TIME_STRETCH_RANGE = (0.9, 1.1)
NOISE_SNR_DB_RANGE = (20.0, 35.0)
AUG_FEATURES_PATH = ARTIFACTS_DIR / "features_aug.npz"

# ---------------------------------------------------------------------------
# Inference and web app
# ---------------------------------------------------------------------------
CALIBRATION_PATH = ARTIFACTS_DIR / "calibration.json"
EVALUATION_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
USE_CALIBRATION = True
LOW_CONFIDENCE_THRESHOLD = 0.5
UPLOAD_TYPES = ["wav", "flac", "ogg", "mp3"]
MAX_UPLOAD_MB = 10  # keep in sync with maxUploadSize in .streamlit/config.toml
