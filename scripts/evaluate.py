#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import yaml

from attentivesd.data.cnnsplice import CnnSpliceDataModule
from attentivesd.models.model import HybridSpliceModel
from attentivesd.utils.seed import set_seed

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.organism:
        config["data"]["organism"] = args.organism
    if args.site:
        config["data"]["site"] = args.site
    if args.mode:
        config["model"]["mode"] = args.mode
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    return config


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    targets = targets.float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    accuracy = (preds == targets).float().mean().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def plot_confusion_matrix(metrics: dict, save_path: Path = None) -> None:
    """Create and optionally save a confusion matrix visualization."""
    import numpy as np

    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar=True, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()
    else:
        # Text-based confusion matrix
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Neg    Pos")
        print(f"True  Neg      {metrics['tn']:6d} {metrics['fp']:6d}")
        print(f"      Pos      {metrics['fn']:6d} {metrics['tp']:6d}")


def generate_classification_report(metrics: dict, save_path: Path = None) -> None:
    """Generate and save a classification report."""
    # Calculate per-class metrics
    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]

    # Negative class (class 0)
    neg_precision = tn / (tn + fn + 1e-8)
    neg_recall = tn / (tn + fp + 1e-8)
    neg_f1 = 2 * neg_precision * neg_recall / \
        (neg_precision + neg_recall + 1e-8)
    neg_support = tn + fp

    # Positive class (class 1)
    pos_precision = metrics["precision"]
    pos_recall = metrics["recall"]
    pos_f1 = metrics["f1"]
    pos_support = tp + fn

    # Macro and weighted averages
    macro_precision = (neg_precision + pos_precision) / 2
    macro_recall = (neg_recall + pos_recall) / 2
    macro_f1 = (neg_f1 + pos_f1) / 2

    total_support = neg_support + pos_support
    weighted_precision = (neg_precision * neg_support +
                          pos_precision * pos_support) / total_support
    weighted_recall = (neg_recall * neg_support +
                       pos_recall * pos_support) / total_support
    weighted_f1 = (neg_f1 * neg_support + pos_f1 * pos_support) / total_support

    # Format the report
    report = "\n" + "="*70 + "\n"
    report += "                    CLASSIFICATION REPORT\n"
    report += "="*70 + "\n\n"
    report += f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
    report += "-"*70 + "\n"
    report += f"{'Negative (0)':<15} {neg_precision:<12.4f} {neg_recall:<12.4f} {neg_f1:<12.4f} {neg_support:<10}\n"
    report += f"{'Positive (1)':<15} {pos_precision:<12.4f} {pos_recall:<12.4f} {pos_f1:<12.4f} {pos_support:<10}\n"
    report += "-"*70 + "\n"
    report += f"{'Accuracy':<15} {'':<12} {'':<12} {metrics['accuracy']:<12.4f} {total_support:<10}\n"
    report += f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {total_support:<10}\n"
    report += f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {total_support:<10}\n"
    report += "="*70 + "\n"

    print(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")


def plot_training_curves(history_path: Path, save_path: Path = None) -> None:
    """Plot training and validation loss/accuracy curves."""
    if not history_path.exists():
        print(f"Training history not found at {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = history.get("epochs", [])
    if not epochs:
        print("No training history available")
        return

    if HAS_PLOT:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss plot
        axes[0].plot(epochs, history["train_loss"], 'b-',
                     label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history["val_loss"], 'r-',
                     label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss',
                          fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot(epochs, history["train_acc"], 'b-',
                     label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, history["val_acc"], 'r-',
                     label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy',
                          fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # F1 plot
        axes[2].plot(epochs, history["train_f1"], 'b-',
                     label='Train F1', linewidth=2)
        axes[2].plot(epochs, history["val_f1"], 'r-',
                     label='Val F1', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('F1 Score', fontsize=12)
        axes[2].set_title('Training and Validation F1 Score',
                          fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()
        plt.close()
    else:
        # Text summary
        print("\nTraining Summary:")
        print(f"Epochs: {len(epochs)}")
        print(
            f"Final Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")
        print(
            f"Final Train Acc: {history['train_acc'][-1]:.4f}, Val Acc: {history['val_acc'][-1]:.4f}")
        print(
            f"Final Train F1: {history['train_f1'][-1]:.4f}, Val F1: {history['val_f1'][-1]:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset", choices=["balanced", "imbalanced"], default=None)
    parser.add_argument(
        "--organism", choices=["hs", "at", "d_mel", "c_elegans", "oriza"], default=None
    )
    parser.add_argument("--site", choices=["donor", "acceptor"], default=None)
    parser.add_argument(
        "--mode", choices=["cnn", "cnn_attention", "attention"], default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = apply_overrides(load_config(Path(args.config)), args)
    set_seed(config["seed"])

    data_module = CnnSpliceDataModule(config["data"])
    model = HybridSpliceModel(config["model"])

    state_dict = load_checkpoint(Path(args.checkpoint))
    model.load_state_dict(state_dict)

    device = resolve_device(config["train"]["device"])
    model.to(device)
    model.eval()

    _, _, test_loader = data_module.get_loaders(
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
    )

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(logits, targets)

    print(
        "test_acc {accuracy:.4f} test_f1 {f1:.4f} test_precision {precision:.4f} test_recall {recall:.4f}".format(
            **metrics
        )
    )
    print(
        "confusion_matrix tn={tn} fp={fp} fn={fn} tp={tp}".format(
            **metrics
        )
    )

    # Generate outputs
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    report_path = output_dir / "classification_report.txt"
    generate_classification_report(metrics, save_path=report_path)

    # Confusion matrix visualization
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics, save_path=cm_path)

    # Plot training curves
    history_path = output_dir / "training_history.json"
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history_path, save_path=curves_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
