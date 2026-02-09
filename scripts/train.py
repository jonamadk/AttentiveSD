#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml

from attentivesd.data.cnnsplice import CnnSpliceDataModule
from attentivesd.models.cnn_attention import HybridSpliceModel
from attentivesd.training.trainer import Trainer
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
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["balanced", "imbalanced"], default=None)
    parser.add_argument(
        "--organism", choices=["hs", "at", "d_mel", "c_elegans", "oriza"], default=None
    )
    parser.add_argument("--site", choices=["donor", "acceptor"], default=None)
    parser.add_argument("--mode", choices=["cnn", "cnn_attention", "attention"], default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    config = apply_overrides(config, args)

    set_seed(config["seed"])

    data_module = CnnSpliceDataModule(config["data"])
    model = HybridSpliceModel(config["model"])

    trainer = Trainer(model=model, data_module=data_module, config=config["train"])
    trainer.fit()
    trainer.test()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
