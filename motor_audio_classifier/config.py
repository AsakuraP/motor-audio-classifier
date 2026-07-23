"""Global configuration for the motor audio fault classifier.

All tunable hyperparameters and paths live here so the rest of the
codebase can stay clean and the behaviour of the project is easy to
inspect and modify in a single place.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Absolute path of this file's directory (the project root).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Root directory that is scanned recursively for *.wav files.
# Defaults to the parent folder of this project (where the original
# sounds/, sounds1/ and data/ folders live). Override with the
# MOTOR_DATA_ROOT environment variable when your data lives elsewhere.
DATA_ROOT = os.environ.get(
    "MOTOR_DATA_ROOT",
    os.path.dirname(PROJECT_ROOT),
)

# Where trained artefacts are written.
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
MODEL_PATH = os.path.join(MODEL_DIR, "motor_cnn.keras")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "norm_stats.json")

# ---------------------------------------------------------------------------
# Audio / signal processing
# ---------------------------------------------------------------------------
TARGET_SR = 44100                 # sample rate all audio is resampled to
DURATION = 4.0                    # seconds kept from each recording
TARGET_SAMPLES = int(TARGET_SR * DURATION)
WINDOW_DURATION = 0.5             # informational: ~1.11s window per segment
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_DURATION)

N_MELS = 96                       # mel bands
N_FFT = 2048                      # STFT window size
HOP_LENGTH = 512                  # STFT hop
MEL_HOP = N_MELS                  # segment width in frames (= 96)

# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
# A file whose name starts with "NG" (case insensitive) is a faulty motor,
# everything else is treated as a healthy ("OK") motor.
OK_PREFIX = "OK"
NG_PREFIX = "NG"
LABEL_OK = 0
LABEL_NG = 1
CLASS_NAMES = {LABEL_OK: "OK", LABEL_NG: "NG"}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 5e-4
VAL_SPLIT = 0.2                   # share of data used for testing
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Regularization (anti-overfitting)
# ---------------------------------------------------------------------------
L2_REG = 1e-4                     # L2 weight decay on conv/dense layers
CONV_DROPOUT = 0.25               # dropout after each conv block
DENSE_DROPOUT = 0.3               # dropout before the final classifier
EARLY_STOPPING_PATIENCE = 10       # stop when val_auc stops improving

# ---------------------------------------------------------------------------
# Asymmetric cost
# ---------------------------------------------------------------------------
# NG (faulty) MUST NOT be missed. We therefore:
#   * up-weight the NG class in the loss so NG->OK mistakes are penalized more,
#   * lower the decision threshold so the model leans toward flagging NG,
#   * tune that threshold on the validation set to guarantee a minimum NG recall.
# Trade-off allowed by the user: it is OK to call an OK motor NG (false alarm),
# but never OK to call an NG motor OK (missed fault).
NG_CLASS_WEIGHT = 2.0             # relative penalty for the NG class
TARGET_RECALL_NG = 0.90           # minimum required recall of the NG class
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.json")
# File-level NG-fraction threshold (tuned on the held-out set so the rule is
# "flag a file NG when at least this fraction of its segments are NG").
NG_FRACTION_PATH = os.path.join(MODEL_DIR, "ng_fraction.json")

# File-level decision rule (used in predict.py and the file-level eval).
# A file is called NG only when at least this fraction of its segments exceed
# the decision threshold. This is far less prone to OK false alarms than the
# per-segment / "any-segment" rule, while NG files (mostly NG segments) are
# still caught. Raise toward 1.0 for even fewer false alarms; lower toward 0.0
# for a more conservative (max-probability-style) behaviour.
MIN_NG_FRACTION = 0.5            # fraction of NG segments required to flag a file

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
# Set to True to enable mixed precision / GPU memory growth hints.
ENABLE_GPU = True
