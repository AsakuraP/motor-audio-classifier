"""Audio loading and Mel-spectrogram feature extraction.

Each recording is:
  1. resampled and trimmed/padded to a fixed length,
  2. converted to a log-Mel spectrogram,
  3. split into fixed-width segments that become the CNN's input windows.

All spectrograms are forced to ``float32`` so they match the model's dtype,
and ``build_segments`` always returns a 3-D array ``(n_segments, n_mels,
mel_hop)`` — including an empty ``(0, n_mels, mel_hop)`` when no segment can
be formed — so concatenation downstream never fails on a shape/dtype mismatch.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import librosa

from config import (
    TARGET_SR,
    TARGET_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    MEL_HOP,
    OK_PREFIX,
    NG_PREFIX,
    LABEL_OK,
    LABEL_NG,
)


# Directories we should never descend into (virtualenvs, caches, VCS).
SKIP_DIRS = {".venv", "venv", "env", "node_modules", "__pycache__"}


def list_audio_files(root: str) -> List[str]:
    """Recursively collect all *.wav files under *root*.

    Hidden directories and known dependency/cache folders (e.g. ``.venv``)
    are pruned so we don't accidentally try to load library test fixtures.
    """
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place: avoid descending into venvs / caches / dot dirs.
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            if name.lower().endswith(".wav"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def label_from_filename(path: str) -> int:
    """Return the integer label derived from a file name prefix."""
    name = os.path.basename(path).upper()
    if name.startswith(NG_PREFIX):
        return LABEL_NG
    return LABEL_OK


def load_audio(path: str) -> Tuple[npt.NDArray[np.float32], int]:
    """Load a wav file, resample and trim/pad it to TARGET_SAMPLES."""
    signal, sr = librosa.load(path, sr=TARGET_SR)
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) >= TARGET_SAMPLES:
        signal = signal[:TARGET_SAMPLES]
    else:
        signal = np.pad(signal, (0, TARGET_SAMPLES - len(signal)))
    return signal, int(sr)


def to_log_melspectrogram(signal: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
    """Compute a log-Mel spectrogram (dB scale) for a signal (float32)."""
    S = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def build_segments(
    spectrogram: npt.NDArray[np.float32],
    mel_hop: int = MEL_HOP,
    keep_last: bool = True,
) -> npt.NDArray[np.float32]:
    """Split a ``(n_mels, n_frames)`` spectrogram into fixed-width segments.

    Full ``mel_hop``-wide windows are taken first. When ``keep_last`` is True
    the trailing partial window is zero-padded to ``mel_hop`` so that short
    recordings still contribute at least one segment instead of being silently
    dropped. The result is always a 3-D ``float32`` array
    ``(n_segments, n_mels, mel_hop)`` (possibly empty in shape).
    """
    n_frames = spectrogram.shape[1]
    segments: List[npt.NDArray[np.float32]] = []

    start = 0
    while start + mel_hop <= n_frames:
        segments.append(spectrogram[:, start:start + mel_hop])
        start += mel_hop

    if keep_last and start < n_frames:
        tail = spectrogram[:, start:]
        pad_width = mel_hop - tail.shape[1]
        tail = np.pad(tail, ((0, 0), (0, pad_width)))
        segments.append(tail)

    if len(segments) == 0:
        # No segment could be formed (e.g. zero-frame spectrogram): return a
        # correctly shaped, correctly typed empty array instead of (0,).
        return np.zeros((0, spectrogram.shape[0], mel_hop), dtype=np.float32)

    return np.asarray(segments, dtype=np.float32)


def extract_segments(path: str) -> Tuple[npt.NDArray[np.float32], int]:
    """Return ``(segments, label)`` for a single audio file.

    *segments* has shape ``(n_segments, n_mels, mel_hop)`` and dtype float32.
    """
    signal, sr = load_audio(path)
    spec = to_log_melspectrogram(signal, sr)
    segs = build_segments(spec)
    label = label_from_filename(path)
    return segs, label
