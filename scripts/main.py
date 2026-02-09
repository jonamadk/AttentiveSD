#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list, env: dict) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def get_data_sizes(dataset: str, organism: str, site: str) -> tuple:
    """Get train and test data sizes from preprocessed files."""
    processed_dir = ROOT / "data" / "processed" / dataset / organism
    train_path = processed_dir / f"train_{site}_{organism}.npz"
    test_path = processed_dir / f"test_{site}_{organism}.npz"

    train_size = 0
    test_size = 0

    if train_path.exists():
        data = np.load(train_path)
        train_size = data["x"].shape[0]

    if test_path.exists():
        data = np.load(test_path)
        test_size = data["x"].shape[0]

    return train_size, test_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AttentiveSD pipeline: preprocess -> train+eval")
    parser.add_argument("--dataset", default="balanced")
    parser.add_argument("--organism", default="hs")
    parser.add_argument("--site", default="donor")
    parser.add_argument(
        "--mode", choices=["cnn", "cnn_attention", "cnn_attention_rope", "all"], default=None)
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(
        f"Dataset: {args.dataset}, Organism: {args.organism}, Site: {args.site}")
    print(f"{'='*80}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    py = sys.executable

    # Preprocess train split
    run(
        [
            py,
            str(ROOT / "scripts" / "preprocess_data.py"),
            "--dataset",
            args.dataset,
            "--organism",
            args.organism,
            "--site",
            args.site,
            "--split",
            "train",
        ],
        env,
    )

    # Preprocess test split
    run(
        [
            py,
            str(ROOT / "scripts" / "preprocess_data.py"),
            "--dataset",
            args.dataset,
            "--organism",
            args.organism,
            "--site",
            args.site,
            "--split",
            "test",
        ],
        env,
    )

    # Get and display data sizes
    train_size, test_size = get_data_sizes(
        args.dataset, args.organism, args.site)
    print(f"\n{'='*80}")
    print(f"Dataset Information:")
    print(f"  Training samples: {train_size:,}")
    print(f"  Test samples: {test_size:,}")
    print(f"{'='*80}")

    # Ask for number of training samples
    max_samples = None
    max_test_samples = None
    print("\nHow many training samples do you want to use?")
    print(
        f"  Enter a number (1-{train_size:,}) or press Enter to use all samples")

    while True:
        samples_input = input(f"Training samples [{train_size:,}]: ").strip()
        if not samples_input:
            # Use all samples
            print(f"✓ Using all {train_size:,} training samples")
            print(f"✓ Using all {test_size:,} test samples")
            break
        try:
            max_samples = int(samples_input)
            if 1 <= max_samples <= train_size:
                # Calculate test samples as 20% of training samples
                max_test_samples = max(1, int(max_samples * 0.2))
                print(f"✓ Will use {max_samples:,} training samples")
                print(
                    f"✓ Will use {max_test_samples:,} test samples (20% of training)")
                break
            else:
                print(f"Please enter a number between 1 and {train_size:,}")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter for all samples.")

    # Ask for model mode if not provided
    if args.mode is None:
        print("\nSelect model architecture:")
        print("  1) cnn                - CNN-only (no attention)")
        print("  2) cnn_attention      - CNN + Multi-head Attention (no RoPE)")
        print("  3) cnn_attention_rope - CNN + Multi-head Attention + RoPE")
        print("  4) all                - Train and evaluate all models sequentially")

        while True:
            choice = input("\nEnter choice (1/2/3/4): ").strip()
            if choice == "1":
                args.mode = "cnn"
                break
            elif choice == "2":
                args.mode = "cnn_attention"
                break
            elif choice == "3":
                args.mode = "cnn_attention_rope"
                break
            elif choice == "4":
                args.mode = "all"
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")

    print(f"\n{'='*80}")
    print(f"Running pipeline with mode: {args.mode}")
    print(f"{'='*80}\n")

    # Train and evaluate - either single mode or all modes
    if args.mode == "all":
        modes = ["cnn", "cnn_attention", "cnn_attention_rope"]
        for i, mode in enumerate(modes, 1):
            print(f"\n{'='*80}")
            print(f"Training model {i}/3: {mode}")
            print(f"{'='*80}\n")
            cmd = [
                py,
                str(ROOT / "scripts" / "train.py"),
                "--dataset",
                args.dataset,
                "--organism",
                args.organism,
                "--site",
                args.site,
                "--mode",
                mode,
            ]
            if max_samples is not None:
                cmd.extend(["--max-samples", str(max_samples)])
            if max_test_samples is not None:
                cmd.extend(["--max-test-samples", str(max_test_samples)])
            run(cmd, env)
    else:
        # Train and evaluate (evaluation happens automatically after training)
        cmd = [
            py,
            str(ROOT / "scripts" / "train.py"),
            "--dataset",
            args.dataset,
            "--organism",
            args.organism,
            "--site",
            args.site,
            "--mode",
            args.mode,
        ]
        if max_samples is not None:
            cmd.extend(["--max-samples", str(max_samples)])
        if max_test_samples is not None:
            cmd.extend(["--max-test-samples", str(max_test_samples)])
        run(cmd, env)

    print("\n" + "="*80)
    if args.mode == "all":
        print("Pipeline complete! Check outputs/ for model-specific folders:")
        print("  outputs/cnn/")
        print("  outputs/cnn_attention/")
        print("  outputs/cnn_attention_rope/")
    else:
        print(f"Pipeline complete! Check outputs/{args.mode}/ for:")
    print("\nEach model folder contains:")
    print("  - checkpoint_best.pt (trained model)")
    print("  - training_log.txt (complete training log with timestamps)")
    print("  - training_history.json (epoch-wise metrics)")
    print("  - classification_report.txt (detailed metrics)")
    print("  - confusion_matrix.png (confusion matrix heatmap)")
    print("  - training_curves.png (train/val loss, accuracy, F1)")
    print("  - roc_curve.png (ROC curve for test set)")
    print("  - precision_recall_curve.png (PR curve for test set)")
    print("  - train_val_test_comparison.png (train/val/test comparison)")
    print("="*80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
