"""Run inference on new audio files using a trained model.

Each input file is turned into segments; we report the per-file prediction
as the majority vote across its segments (and the mean confidence).

Usage:
    python predict.py path/to/file.wav
    python predict.py path/to/folder --threshold 0.5
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from config import (  # noqa: E402
    CLASS_NAMES,
    MODEL_PATH,
    NORM_STATS_PATH,
    MIN_NG_FRACTION,
)
from data.audio import list_audio_files, extract_segments  # noqa: E402
from data.dataset import (  # noqa: E402
    load_norm_stats,
    load_threshold,
    load_ng_fraction,
)
from models.cnn import build_model  # noqa: E402


def load_model(model_path=MODEL_PATH, norm_stats_path=NORM_STATS_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with `python train.py`."
        )
    model = build_model()
    model.load_weights(model_path)
    stats = load_norm_stats(norm_stats_path)
    return model, stats


def predict_file(model, path, stats, threshold, min_ng_fraction=MIN_NG_FRACTION):
    segs, _ = extract_segments(path)
    if len(segs) == 0:
        return None, None, None, None
    X = segs[..., np.newaxis]
    mean, std = stats["mean"], stats["std"]
    X = (X - mean) / (std + 1e-8)

    probs = model.predict(X, verbose=0).ravel()
    mean_prob = float(probs.mean())
    max_prob = float(probs.max())
    # File-level decision: flag NG only when at least min_ng_fraction of the
    # file's segments exceed the threshold. A single spurious segment therefore
    # will not trigger a false alarm, while a faulty motor (most segments NG)
    # is still reliably caught.
    ng_frac = float(np.mean(probs >= threshold))
    label = int(ng_frac >= min_ng_fraction)
    return label, mean_prob, max_prob, ng_frac


def main():
    parser = argparse.ArgumentParser(description="Predict motor fault from audio")
    parser.add_argument("input", help="A .wav file or a folder of .wav files")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the decision threshold (default: tuned value)")
    args = parser.parse_args()

    model, stats = load_model()

    threshold = args.threshold
    if threshold is None:
        threshold = load_threshold()
    if threshold is None:
        threshold = 0.5
        print("[info] no tuned threshold found; using 0.5")

    # File-level rule: flag NG when at least this fraction of a file's segments
    # exceed the threshold. Prefer the value tuned on the held-out set.
    min_ng_frac = load_ng_fraction()
    if min_ng_frac is None:
        min_ng_frac = MIN_NG_FRACTION
        print(f"[info] no tuned NG fraction found; using {min_ng_frac:.2f}")
    else:
        print(f"[info] using tuned NG fraction = {min_ng_frac:.3f}")

    if os.path.isdir(args.input):
        targets = list_audio_files(args.input)
    else:
        targets = [args.input]

    for fp in targets:
        label, mean_prob, max_prob, ng_frac = predict_file(
            model, fp, stats, threshold, min_ng_frac)
        if label is None:
            print(f"{fp}: no segments extracted")
            continue
        verdict = CLASS_NAMES[label]
        print(f"{fp}: {verdict} (mean_conf={mean_prob:.3f}, "
              f"max_conf={max_prob:.3f}, ng_frac={ng_frac:.3f}, "
              f"min_ng_frac={min_ng_frac:.2f}, threshold={threshold:.3f})")


if __name__ == "__main__":
    main()
