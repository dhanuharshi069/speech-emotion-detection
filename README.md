# Emotion Detection from Speech Using Deep Learning

Classifies the emotion expressed in a short speech clip as angry, happy, neutral or sad, using a
CNN-BiLSTM trained on the RAVDESS dataset. A Streamlit app lets you upload a clip, see the
prediction with approximate confidence, inspect the input features and review the evaluation.

The output is a prediction of **expressed vocal emotion**. It is not a measure of how the speaker
actually feels.

## Results

One evaluation of the final model on four speakers never used for training or validation
(actors 1, 12, 15 and 24, 112 clips). Numbers come from `artifacts/evaluation_metrics.json`.

| Metric | Value |
|---|---|
| Accuracy | 70.5% (95% interval 61.5% to 78.2%) |
| Macro precision / recall / F1 | 0.758 / 0.680 / 0.691 |
| Weighted F1 | 0.708 |

| Emotion | Precision | Recall | F1 | Test clips |
|---|---|---|---|---|
| Angry | 1.000 | 0.719 | 0.836 | 32 |
| Happy | 0.818 | 0.562 | 0.667 | 32 |
| Neutral | 0.667 | 0.500 | 0.571 | 16 |
| Sad | 0.545 | 0.938 | 0.690 | 32 |

Read these with care:

- The interval treats clips as independent. Clips from one speaker are correlated, and there are
  only four test speakers, so the real uncertainty is larger.
- Accuracy by speaker ranged from 53.6% to 78.6%.
- Neutral has the fewest clips, so its figures are the least stable.
- The training epoch was chosen on validation data, so validation figures are optimistic. The test
  figures above are the independent estimate.
- Raw confidence was overconfident (mean confidence about 0.84 against accuracy of about 0.69 to
  0.71). Temperature scaling (T = 2.60, fitted on validation data) lowered test log loss from
  0.828 to 0.747 but left expected calibration error essentially unchanged (0.142 to 0.141).
  Displayed confidence is approximate.
- Exact numbers can differ slightly across machines and library versions if you retrain.

Tested on Windows with Python 3.13, TensorFlow 2.20 and librosa 0.11.

## Method

- **Data:** RAVDESS speech, 24 actors, four classes (angry, happy, neutral, sad), 672 recordings.
- **Features:** 22,050 Hz mono, silence trimmed, fixed 3.0 s, 64-band log-Mel spectrogram
  (64 x 259), normalized per band with statistics from the training speakers only.
- **Model:** three conv blocks, a 1x1 channel reduction, a bidirectional LSTM and a dense softmax
  head (270,340 parameters).
- **Split:** by speaker, 16 train, 4 validation, 4 test, with validation and test balanced by gender.
- **Training:** training speakers only are augmented (pitch shift, time stretch, noise), plus
  random frequency and time masking, class weights, and epoch selection by validation macro-F1.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dataset

Download `Audio_Speech_Actors_01-24.zip` from https://zenodo.org/records/1188976 and extract it
so that files look like `data/raw/Actor_01/03-01-01-01-01-01-01.wav`. See `data/README.md`.
RAVDESS is licensed CC BY-NC-SA 4.0. Check its terms before sharing models or using them
commercially.

## Run the pipeline

```bash
python inspect_dataset.py     # index RAVDESS and keep the four target emotions
python build_features.py      # cache log-Mel and MFCC features
python augment_features.py    # augment the training speakers only
python train.py               # train, select the epoch on validation macro-F1
python evaluate.py            # one evaluation on held-out speakers
python calibrate.py           # fit the confidence temperature on validation data
python verify_inference.py    # check the app's inference path reproduces the evaluation
streamlit run app.py
```

`train.py` overwrites the model and artifact files. Run `evaluate.py` once, on a model you will
not change afterwards. `diagnose.py` prints validation-only diagnostics for model development.

## Tests

```bash
pytest
```

The tests cover label mapping, preprocessing, the actor-wise split, augmentation, evaluation
metrics, calibration, inference and a source-level audit of the interface design rules.

## Project structure

```text
app.py                     Home page and navigation
pages/                     Prediction, Model Evaluation, Project Information
config.py                  Paths, labels and all settings
preprocessing.py           Feature extraction and normalization
predict.py                 Inference and status helpers
model.py                   CNN-BiLSTM definition
train.py, evaluate.py      Training and held-out evaluation
calibrate.py               Temperature scaling
inspect_dataset.py, build_features.py, augment_features.py
diagnose.py, verify_inference.py
utils/                     Audio, data, calibration, results, UI and charts
models/                    Trained model
artifacts/                 Label mapping, normalization, metrics, history, calibration
tests/                     Automated tests
```

## Design

Warm ivory background, forest green accent, DM Sans for text and DM Mono for technical labels.
`tests/test_design_rules.py` checks the source for the prohibited patterns that can be detected
in code. Layout, spacing and icon use were reviewed by hand.

## Limitations

- Acted studio speech from 24 actors with a North American accent. Performance on spontaneous
  speech, other languages or accents, or noisy recordings has not been measured.
- Small evaluation: four test speakers, 112 clips.
- Four coarse categories. The labels reflect what actors were asked to express.
- Only the first 3 seconds after trimming silence are analyzed.
- Spectrogram and MFCC plots show the input, not why the model made a prediction.

## Citation

Livingstone, S. R., and Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional
Speech and Song (RAVDESS). PLOS ONE.