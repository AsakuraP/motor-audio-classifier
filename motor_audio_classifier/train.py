"""Train the motor audio fault classifier.

Usage:
    python train.py                 # uses config.DATA_ROOT
    python train.py --data /path    # scan a specific folder
    python train.py --epochs 50 --batch 16
"""

import argparse
import os
import sys

# Make the project root importable when run as a script.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from training.trainer import train  # noqa: E402
from utils.visualize import evaluate_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train motor fault CNN")
    parser.add_argument("--data", default=None,
                        help="Root folder to scan for *.wav files")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--ng-weight", type=float, default=None,
                        help="Class weight for the NG class (higher = fewer "
                             "NG missed, more OK false alarms). Default 4.0")
    parser.add_argument("--recall-ng", type=float, default=None,
                        help="Minimum required NG recall when tuning the "
                             "decision threshold. Default 0.90")
    args = parser.parse_args()

    model, _, (X_test, y_test, threshold, file_ids_test, ng_fraction), _ = train(
        root=args.data or config.DATA_ROOT,
        epochs=args.epochs or config.EPOCHS,
        batch_size=args.batch or config.BATCH_SIZE,
        ng_class_weight=args.ng_weight,
        target_recall_ng=args.recall_ng,
    )

    # Evaluate on the held-out test set using the tuned decision threshold.
    evaluate_model(model, X_test, y_test, threshold=threshold,
                   file_ids=file_ids_test, min_ng_fraction=ng_fraction)


if __name__ == "__main__":
    main()
