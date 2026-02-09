#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import yaml

from attentivesd.data.cnnsplice import CnnSpliceDataModule
from attentivesd.models.cnn_attention import HybridSpliceModel
from attentivesd.utils.seed import set_seed


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["balanced", "imbalanced"], default=None)
    parser.add_argument(
        "--organism", choices=["hs", "at", "d_mel", "c_elegans", "oriza"], default=None
    )
    parser.add_argument("--site", choices=["donor", "acceptor"], default=None)
    parser.add_argument("--mode", choices=["cnn", "cnn_attention", "attention"], default=None)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
