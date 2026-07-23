"""Assemble the full dataset and expose train/test splits.

A single recording produces many short segments; each segment inherits the
label of its source file. We split at the *segment* level (stratified) which
keeps the original notebook behaviour where every slice is an independent
sample.
"""

import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE,
    VAL_SPLIT,
    RANDOM_SEED,
    N_MELS,
    MEL_HOP,
    NORM_STATS_PATH,
    THRESHOLD_PATH,
    NG_FRACTION_PATH,
)
from .audio import list_audio_files, extract_segments


def build_dataset(root):
    """Return (X, y, file_ids).

    ``X`` has shape (n_segments, n_mels, mel_hop, 1); ``file_ids`` is a 1-D array
    mapping every segment back to the source file (so segments can later be
    aggregated per file for a file-level decision / evaluation).
    """
    files = list_audio_files(root)
    if not files:
        raise RuntimeError(f"No .wav files found under {root!r}.")

    X_parts, y_parts, id_parts = [], [], []
    skipped = 0
    for file_idx, fp in enumerate(files):
        try:
            segs, label = extract_segments(fp)
        except Exception as exc:  # corrupt / unreadable file: skip, don't crash
            print(f"[warn] skipping {fp}: {exc}")
            skipped += 1
            continue
        if len(segs) == 0:
            skipped += 1
            continue
        X_parts.append(segs)
        y_parts.append(np.full((len(segs),), label))
        id_parts.append(np.full((len(segs),), file_idx))

    if not X_parts:
        raise RuntimeError(
            f"No usable segments extracted from {len(files)} file(s) "
            f"under {root!r} ({skipped} skipped)."
        )
    print(f"[data] found {len(files)} file(s), {skipped} skipped, "
          f"{sum(len(p) for p in X_parts)} segments")

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    file_ids = np.concatenate(id_parts)
    X = X[..., np.newaxis]  # add channel dimension for the CNN
    return X, y, file_ids


def normalize(X_train, X_test):
    """Standardise using training statistics; return (Xtr, Xte, stats)."""
    mean = float(X_train.mean())
    std = float(X_train.std())
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    stats = {"mean": mean, "std": std}
    return X_train, X_test, stats


def save_norm_stats(stats, path=NORM_STATS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)


def load_norm_stats(path=NORM_STATS_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_threshold(value, path=THRESHOLD_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(value)}, f)


def load_threshold(path=THRESHOLD_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["threshold"])


def save_ng_fraction(value, path=NG_FRACTION_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"min_ng_fraction": float(value)}, f)


def load_ng_fraction(path=NG_FRACTION_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["min_ng_fraction"])


def get_data_loaders(root, norm_stats=None):
    """Build train/test arrays (already normalised) plus the norm stats.

    Returns (X_train, X_test, y_train, y_test, stats).
    """
    X, y, file_ids = build_dataset(root)
    X_train, X_test, y_train, y_test, _, fid_test = train_test_split(
        X, y, file_ids,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    if norm_stats is None:
        X_train, X_test, stats = normalize(X_train, X_test)
        save_norm_stats(stats)
    else:
        mean, std = norm_stats["mean"], norm_stats["std"]
        X_train = (X_train - mean) / (std + 1e-8)
        X_test = (X_test - mean) / (std + 1e-8)
        stats = norm_stats
    return X_train, X_test, y_train, y_test, stats, fid_test
