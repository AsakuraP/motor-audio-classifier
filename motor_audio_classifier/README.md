# Motor Audio Fault Classifier

A clean, modular Python project converted from the exploratory notebook
`motor(1).ipynb`. It classifies motor audio recordings as **OK** (healthy) or
**NG** (faulty) using a 2D CNN operating on Mel-spectrogram segments.

## How it works

1. Every `.wav` file is resampled to 44.1 kHz and trimmed/padded to 4 seconds.
2. A log-Mel spectrogram (96 mel bands) is computed.
3. The spectrogram is split into fixed 96-frame segments; each segment becomes
   one training sample.
4. A small CNN is trained to predict the binary label (derived from the file
   name prefix: files starting with `NG` are faulty, everything else is OK).
5. Training statistics are used to normalise the data; the same statistics are
   reused at inference time.

## Project layout

```
motor_audio_classifier/
├── config.py              # All hyperparameters and paths in one place
├── train.py               # Entry point: train the model
├── predict.py             # Entry point: run inference on new audio
├── requirements.txt
├── README.md
├── data/
│   ├── audio.py           # Loading + Mel-spectrogram feature extraction
│   └── dataset.py         # Dataset assembly, split and normalisation
├── models/
│   └── cnn.py             # CNN definition / compilation
├── training/
│   └── trainer.py         # Training loop + GPU setup
└── utils/
    └── visualize.py       # Accuracy/loss curves, confusion matrix, report
```

## Setup

```bash
cd motor_audio_classifier
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data location

By default the project scans the **parent folder of this project** for `*.wav`
files (where the original `sounds/`, `sounds1/` and `data/` folders live). To
point it somewhere else, set the `MOTOR_DATA_ROOT` environment variable or pass
`--data`:

```bash
export MOTOR_DATA_ROOT=/path/to/audio
python train.py
# or
python train.py --data /path/to/audio
```

## Train

```bash
python train.py                 # uses config defaults
python train.py --epochs 50 --batch 16
```

Outputs (under `models/checkpoints/`):
- `motor_cnn.h5` — model weights
- `norm_stats.json` — mean/std used for normalisation
- `training_history.png`, `confusion_matrix.png` — figures

## Predict

```bash
python predict.py path/to/file.wav
python predict.py path/to/folder --threshold 0.5
```

Each file's prediction is the majority vote across its segments, and the mean
confidence is reported.

## Configuration

Edit `config.py` to change the sample rate, Mel parameters, segment size,
labelling prefixes, training hyperparameters, or output paths.
