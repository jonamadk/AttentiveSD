import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    # Handle edge case where only one class is predicted
    if len(np.unique(y_pred)) == 1:
        print(f"WARNING: Model predicting only class {y_pred[0]}!")
        print(f"Unique predictions: {np.unique(y_pred)}")
        print(f"Prediction distribution: {np.bincount(y_pred)}")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    """Plot training curves from training history."""
    if not history_path.exists():
        print(
            f"Training history not found at {history_path}, skipping curves plot")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"],
                 label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, history["train_f1"], label="Train F1", marker="o")
    axes[2].plot(epochs, history["val_f1"], label="Val F1", marker="s")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Training and Validation F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Generate and save classification report."""
    report = classification_report(
        y_true, y_pred,
        target_names=["Negative", "Positive"],
        digits=4
    )
    with open(output_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"Classification report saved to {output_path}")


def evaluate_model(checkpoint_path: Path, data_module, config: dict, output_dir: Path) -> Dict[str, float]:
    """Evaluate model on test set and generate all reports."""
    from attentivesd.models.model import HybridSpliceModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridSpliceModel(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Assume checkpoint is the state dict itself
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Get test loader
    if not hasattr(data_module, 'test_loader') or data_module.test_loader is None:
        data_module.get_loaders(
            batch_size=config["train"].get("batch_size", 32),
            num_workers=config["train"].get("num_workers", 0)
        )
    test_loader = data_module.test_loader

    all_preds = []
    all_probs = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            # Debug: Check logits distribution
            if len(all_logits) == 0:
                print(f"\nDEBUG - First batch logits stats:")
                print(f"  Shape: {logits.shape}")
                print(
                    f"  Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
                print(
                    f"  Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")

            # For binary classification with single output
            # Logits shape is [batch_size] for single output
            if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[-1] == 1):
                if logits.dim() == 2:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
            else:
                # For 2-class output
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs > 0.5).long()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    logits_all = np.concatenate(all_logits)

    # Debug output
    print(f"\n{'='*80}")
    print("Evaluation Debug Information")
    print(f"{'='*80}")
    print(f"True labels distribution: {np.bincount(y_true.astype(int))}")
    print(f"Predicted labels distribution: {np.bincount(y_pred.astype(int))}")
    print(f"Probability range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    print(f"Logits range: [{logits_all.min():.4f}, {logits_all.max():.4f}]")
    print(f"{'='*80}\n")

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\n" + "="*80)
    print("Test Set Evaluation Metrics")
    print("="*80)
    for key, value in metrics.items():
        print(f"{key:>12s}: {value:.4f}")
    print("="*80 + "\n")

    # Generate visualizations and reports
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    save_classification_report(
        y_true, y_pred, output_dir / "classification_report.txt")
    plot_training_curves(output_dir / "training_history.json",
                         output_dir / "training_curves.png")

    return metrics
