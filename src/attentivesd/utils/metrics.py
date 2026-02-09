import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        average_precision_score,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute binary classification metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    targets = targets.float()

    accuracy = (preds == targets).float().mean().item()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"accuracy": accuracy, "f1": f1}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    if HAS_SKLEARN:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        }
    else:
        # Fallback implementation
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        accuracy = (y_pred == y_true).mean()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": 0.0,
        }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save confusion matrix."""
    if not HAS_PLOT:
        print("matplotlib/seaborn not available, skipping confusion matrix plot")
        return

    if HAS_SKLEARN:
        cm = confusion_matrix(y_true, y_pred)
    else:
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        cm = np.array([[tn, fp], [fn, tp]])

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


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Generate and save classification report."""
    if HAS_SKLEARN:
        report = classification_report(
            y_true, y_pred,
            target_names=["Negative", "Positive"],
            digits=4
        )
    else:
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        neg_prec = tn / (tn + fn + 1e-8)
        neg_rec = tn / (tn + fp + 1e-8)
        neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec + 1e-8)

        pos_prec = tp / (tp + fp + 1e-8)
        pos_rec = tp / (tp + fn + 1e-8)
        pos_f1 = 2 * pos_prec * pos_rec / (pos_prec + pos_rec + 1e-8)

        report = f"""              precision    recall  f1-score   support

    Negative     {neg_prec:.4f}    {neg_rec:.4f}    {neg_f1:.4f}      {tn+fp}
    Positive     {pos_prec:.4f}    {pos_rec:.4f}    {pos_f1:.4f}      {tp+fn}

    accuracy                         {(tp+tn)/(tp+tn+fp+fn):.4f}      {tp+tn+fp+fn}
"""

    with open(output_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"Classification report saved to {output_path}")


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    """Plot training curves from training history."""
    if not history_path.exists():
        print(
            f"Training history not found at {history_path}, skipping curves plot")
        return

    if not HAS_PLOT:
        print("matplotlib not available, skipping training curves plot")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = history.get("epochs", list(
        range(1, len(history["train_loss"]) + 1)))

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


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    """Plot ROC curve for test set."""
    if not HAS_PLOT:
        print("matplotlib not available, skipping ROC curve plot")
        return

    if not HAS_SKLEARN:
        print("sklearn not available, skipping ROC curve plot")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    """Plot Precision-Recall curve for test set."""
    if not HAS_PLOT:
        print("matplotlib not available, skipping PR curve plot")
        return

    if not HAS_SKLEARN:
        print("sklearn not available, skipping PR curve plot")
        return

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Test Set',
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Precision-Recall curve saved to {output_path}")


def plot_train_val_test_comparison(history_path: Path, test_metrics: Dict[str, float], output_path: Path) -> None:
    """Plot comparison of final train/val/test metrics."""
    if not HAS_PLOT:
        print("matplotlib not available, skipping comparison plot")
        return

    if not history_path.exists():
        print(
            f"Training history not found at {history_path}, skipping comparison plot")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    # Get final train and val metrics
    train_acc = history["train_acc"][-1] if history["train_acc"] else 0
    val_acc = history["val_acc"][-1] if history["val_acc"] else 0
    train_f1 = history["train_f1"][-1] if history["train_f1"] else 0
    val_f1 = history["val_f1"][-1] if history["val_f1"] else 0

    test_acc = test_metrics.get("accuracy", 0)
    test_precision = test_metrics.get("precision", 0)
    test_recall = test_metrics.get("recall", 0)
    test_f1 = test_metrics.get("f1", 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    categories = ['Train', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    axes[0].bar(categories, accuracies, color=colors,
                alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.4f}',
                     ha='center', fontsize=10, fontweight='bold')

    # F1 Score comparison
    f1_scores = [train_f1, val_f1, test_f1]

    axes[1].bar(categories, f1_scores, color=colors,
                alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.02, f'{v:.4f}',
                     ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Train/Val/Test comparison saved to {output_path}")


def evaluate_model(checkpoint_path: Path, data_module, config: dict, output_dir: Path) -> Dict[str, float]:
    """Evaluate model on test set and generate all reports."""
    from attentivesd.models.model import HybridSpliceModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridSpliceModel(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Get test loader
    _, _, test_loader = data_module.get_loaders(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"]
    )

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

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

    # Test set specific plots
    plot_roc_curve(y_true, y_prob, output_dir / "roc_curve.png")
    plot_precision_recall_curve(
        y_true, y_prob, output_dir / "precision_recall_curve.png")
    plot_train_val_test_comparison(output_dir / "training_history.json",
                                   metrics, output_dir / "train_val_test_comparison.png")

    return metrics
