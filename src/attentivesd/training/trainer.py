from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import AdamW

from attentivesd.utils.metrics import binary_metrics


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    device: str
    log_every: int
    output_dir: str
    save_best: bool
    save_last: bool


class Trainer:
    def __init__(self, model: nn.Module, data_module, config: dict):
        self.model = model
        self.data_module = data_module
        self.config = TrainConfig(**config)
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_f1 = -1.0

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_loaders(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _run_epoch(self, loader, train: bool) -> Tuple[float, dict]:
        epoch_loss = 0.0
        all_logits = []
        all_targets = []
        self.model.train(train)

        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

            if train and step % self.config.log_every == 0:
                print(f"step {step} loss {loss.item():.4f}")

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        metrics = binary_metrics(logits, targets)
        return epoch_loss / max(1, len(loader)), metrics

    def fit(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_metrics = self._run_epoch(self.train_loader, train=True)
            val_loss, val_metrics = self._run_epoch(self.val_loader, train=False)
            print(
                f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
                f"train_acc {train_metrics['accuracy']:.4f} val_acc {val_metrics['accuracy']:.4f} "
                f"train_f1 {train_metrics['f1']:.4f} val_f1 {val_metrics['f1']:.4f}"
            )

            if self.config.save_last:
                last_path = self.output_dir / "checkpoint_last.pt"
                torch.save({"state_dict": self.model.state_dict(), "epoch": epoch}, last_path)

            if self.config.save_best and val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                best_path = self.output_dir / "checkpoint_best.pt"
                torch.save({"state_dict": self.model.state_dict(), "epoch": epoch}, best_path)

    def test(self) -> None:
        test_loss, test_metrics = self._run_epoch(self.test_loader, train=False)
        print(
            f"test_loss {test_loss:.4f} test_acc {test_metrics['accuracy']:.4f} "
            f"test_f1 {test_metrics['f1']:.4f}"
        )
