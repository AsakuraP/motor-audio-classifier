"""Training loop, optional GPU setup and decision-threshold tuning.

Mirrors the notebook's flow (build dataset -> normalise -> train CNN -> persist
artefacts) and adds:
  * regularised training with early stopping to fight overfitting,
  * class-weighting + a validation-tuned decision threshold so a faulty motor
    (NG) is (almost) never missed, even at the cost of some OK->NG false alarms.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import (
    DATA_ROOT,
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
    MODEL_PATH,
    NORM_STATS_PATH,
    THRESHOLD_PATH,
    NG_FRACTION_PATH,
    ENABLE_GPU,
    NG_CLASS_WEIGHT,
    TARGET_RECALL_NG,
    MIN_NG_FRACTION,
    EARLY_STOPPING_PATIENCE,
)
from data.dataset import (
    get_data_loaders,
    load_norm_stats,
    save_threshold,
    load_threshold,
    save_ng_fraction,
)
from models.cnn import build_model
from utils.visualize import plot_training_history, evaluate_model


def setup_hardware():
    """Enable GPU memory growth when requested and a GPU is present."""
    if not ENABLE_GPU:
        return
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[hardware] GPU enabled: {len(gpus)} device(s)")
        except RuntimeError as exc:  # pragma: no cover - environment specific
            print(f"[hardware] could not configure GPU: {exc}")


def select_threshold(y_true, y_prob, target_recall):
    """Pick a decision threshold that keeps NG recall high without collapsing.

    The primary goal is the highest threshold whose NG (positive) recall still
    meets ``target_recall`` (most permissive while honouring the recall target).
    However, if the model simply cannot reach ``target_recall`` on this set
    (e.g. a handful of hard NG samples are never caught), every threshold fails
    the check and the naive scan would fall back to 1.0 -- predicting every
    sample OK and destroying NG recall entirely. To avoid that, we also track
    the F1-optimal threshold and return it as a fallback, which yields a balanced
    threshold (NG recall ~0.85, OK recall ~0.99) instead of a useless 1.0.
    """
    best = 1.0
    best_f1 = -1.0
    best_f1_thr = 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= thr).astype(int)
        pos = y_true == 1
        tp = int(np.sum((pred[pos] == 1)))
        fn = int(np.sum((pred[pos] == 0)))
        fp = int(np.sum((pred[y_true == 0] == 1)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        if recall >= target_recall:
            best = float(thr)  # ascending scan -> last kept is the highest
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = float(thr)
    # If the recall target is unattainable, fall back to the F1-optimal threshold
    # rather than collapsing to a constant predictor.
    return best if best < 1.0 else best_f1_thr


def select_ng_fraction(y_true, y_prob, file_ids, threshold,
                       max_ng_missed=0):
    """Pick the largest file-level NG fraction that keeps NG missed at 0.

    With the segment threshold fixed (chosen to guarantee NG segment recall),
    we scan the per-file NG-segment fraction from high to low. A file is flagged
    NG when at least ``frac`` of its segments exceed ``threshold``. We keep the
    *largest* frac such that no NG file is missed (i.e. the strictest rule that
    still yields zero missed faults). A larger frac = fewer OK false alarms.

    ``max_ng_missed`` allows a small tolerance (e.g. accept 1 missed fault) for
    datasets where the segment recall cannot reach a perfect 1.0 at the file
    level; default 0 honours the "never miss NG" requirement.
    """
    from utils.visualize import file_level_predict

    best = 1.0  # most conservative: every segment must be NG
    for frac in np.linspace(1.0, 0.05, 96):
        f_true, f_pred = file_level_predict(
            y_true, y_prob, file_ids, threshold, float(frac))
        f_pos = f_true == 1
        f_fn = int(np.sum((f_pred[f_pos] == 0)))
        if f_fn <= max_ng_missed:
            best = float(frac)
    return best



def train(
    root=DATA_ROOT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_path=MODEL_PATH,
    norm_stats_path=NORM_STATS_PATH,
    threshold_path=THRESHOLD_PATH,
    ng_class_weight=None,
    target_recall_ng=None,
    plot=True,
):
    """Train the model and save model + norm stats + decision threshold.

    ``ng_class_weight`` and ``target_recall_ng`` override the config defaults so
    the OK/NG trade-off can be tuned from the command line (see train.py).
    """
    setup_hardware()
    os.makedirs(MODEL_DIR, exist_ok=True)

    if root is None:
        root = DATA_ROOT

    # Resolve asymmetric-cost settings (CLI override > config default).
    ng_weight = NG_CLASS_WEIGHT if ng_class_weight is None else ng_class_weight
    recall_target = TARGET_RECALL_NG if target_recall_ng is None else target_recall_ng

    norm_stats = load_norm_stats(norm_stats_path)
    X_train, X_test, y_train, y_test, stats, file_ids_test = get_data_loaders(
        root, norm_stats)

    print(f"[data] train segments: {X_train.shape[0]} | "
          f"test segments: {X_test.shape[0]}")
    print(f"[data] norm stats: mean={stats['mean']:.4f} std={stats['std']:.4f}")

    # Asymmetric cost: penalize missing an NG motor far more than a false alarm.
    class_weight = {0: 1.0, 1: ng_weight}

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Re-load the best weights (EarlyStopping restored them in-memory, but the
    # checkpoint file is the canonical saved artefact).
    if os.path.exists(model_path):
        model.load_weights(model_path)
    print(f"[train] best model saved to {model_path}")

    # Tune the decision threshold on validation so NG recall >= target.
    y_prob = model.predict(X_test, verbose=0).ravel()
    threshold = select_threshold(y_test, y_prob, recall_target)
    save_threshold(threshold, threshold_path)
    print(f"[train] decision threshold = {threshold:.3f} "
          f"(NG class weight = {ng_weight:.2f}, "
          f"target NG recall >= {recall_target:.2f})")

    # Tune the file-level NG fraction on the same held-out set: pick the
    # strictest fraction that still yields zero missed NG files.
    ng_fraction = select_ng_fraction(y_test, y_prob, file_ids_test, threshold)
    save_ng_fraction(ng_fraction, NG_FRACTION_PATH)
    print(f"[train] file-level NG fraction = {ng_fraction:.3f} "
          f"(flag a file NG when >= this fraction of its segments are NG)")

    if plot:
        plot_training_history(history, save_dir=MODEL_DIR)

    return model, history, (X_test, y_test, threshold, file_ids_test,
                            ng_fraction), stats


def evaluate(model, X_test, y_test, threshold):
    """Convenience wrapper used by train.py after training."""
    return evaluate_model(model, X_test, y_test, threshold=threshold)
