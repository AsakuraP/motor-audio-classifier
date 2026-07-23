"""Plotting and evaluation helpers (mirrors the notebook's visualisations)."""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from config import CLASS_NAMES, MODEL_DIR, MIN_NG_FRACTION


def plot_training_history(history, save_dir=MODEL_DIR):
    """Save accuracy and loss curves from a training History object."""
    os.makedirs(save_dir, exist_ok=True)
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["accuracy"], label="train")
    plt.plot(epochs, history_dict["val_accuracy"], label="val")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict["loss"], label="train")
    plt.plot(epochs, history_dict["val_loss"], label="val")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] training history saved to {path}")


def file_level_predict(y_true, y_prob, file_ids, threshold, min_ng_fraction):
    """Aggregate per-segment predictions into one verdict per source file.

    A file is flagged NG only when at least ``min_ng_fraction`` of its segments
    exceed ``threshold``. All segments of a single file share the same true
    label, so the file's true class is just that shared label.
    """
    file_ids = np.asarray(file_ids).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_true = np.asarray(y_true).ravel()

    file_true, file_pred = [], []
    for fid in np.unique(file_ids):
        mask = file_ids == fid
        true_label = int(np.max(y_true[mask]))  # segments of a file share label
        ng_frac = float(np.mean(y_prob[mask] >= threshold))
        file_true.append(true_label)
        file_pred.append(int(ng_frac >= min_ng_fraction))
    return np.array(file_true), np.array(file_pred)


def evaluate_model(model, X_test, y_test, threshold=0.5, file_ids=None,
                   min_ng_fraction=None, save_dir=MODEL_DIR):
    """Print a classification report (using *threshold*) and save a figure.

    Two confusion matrices are reported:
      * a per-segment one (every slice is an independent sample), and
      * a per-FILE one (the unit that actually matters in production).

    The NG class (faulty, label 1) must (almost) never be missed, so the
    NG-called-OK cells are the numbers we keep at (or near) zero.
    """
    os.makedirs(save_dir, exist_ok=True)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int).ravel()

    # ----- per-segment confusion matrix -----------------------------------
    print(f"\n=== Per-segment evaluation (threshold={threshold:.3f}) ===")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=[CLASS_NAMES[0], CLASS_NAMES[1]]))

    pos = y_test == 1
    tp = int(np.sum((y_pred[pos] == 1)))
    fn = int(np.sum((y_pred[pos] == 0)))
    ng_recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    print(f"NG recall (faults caught): {ng_recall:.4f}  |  "
          f"NG missed (faults called OK): {fn}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("        " + "".join(f"{c:>10s}" for c in labels))
    for i, true_label in enumerate(labels):
        print(f"{true_label:>8s}" + "".join(
            f"{cm[i, j]:>10d}" for j in range(len(labels))))
    print(f"  -> NG missed (true NG, pred OK): {cm[1, 0]}")
    print(f"  -> OK false-alarm (true OK, pred NG): {cm[0, 1]}")

    # ----- per-FILE confusion matrix --------------------------------------
    if file_ids is not None:
        frac = MIN_NG_FRACTION if min_ng_fraction is None else min_ng_fraction
        f_true, f_pred = file_level_predict(
            y_test, y_prob, file_ids, threshold, frac)
        print(f"\n=== Per-FILE evaluation (file flagged NG when "
              f">= {frac:.2f} of segments are NG) ===")
        print("\nFile classification report:")
        print(classification_report(f_true, f_pred,
                                    target_names=[CLASS_NAMES[0], CLASS_NAMES[1]]))

        f_pos = f_true == 1
        f_tp = int(np.sum((f_pred[f_pos] == 1)))
        f_fn = int(np.sum((f_pred[f_pos] == 0)))
        f_recall = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 1.0
        print(f"NG recall (faults caught): {f_recall:.4f}  |  "
              f"NG missed (faults called OK): {f_fn}")

        cm_f = confusion_matrix(f_true, f_pred, labels=[0, 1])
        print("\nFile confusion matrix (rows = true, cols = predicted):")
        print("        " + "".join(f"{c:>10s}" for c in labels))
        for i, true_label in enumerate(labels):
            print(f"{true_label:>8s}" + "".join(
                f"{cm_f[i, j]:>10d}" for j in range(len(labels))))
        print(f"  -> NG missed (true NG, pred OK): {cm_f[1, 0]}")
        print(f"  -> OK false-alarm (true OK, pred NG): {cm_f[0, 1]}")
        cm = cm_f  # save the file-level matrix for the figure

    # ----- figure ---------------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xticks([0, 1], [CLASS_NAMES[0], CLASS_NAMES[1]])
    plt.yticks([0, 1], [CLASS_NAMES[0], CLASS_NAMES[1]])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="red" if i != j else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] confusion matrix saved to {path}")
    return y_pred, cm
