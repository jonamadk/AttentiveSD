from dataclasses import dataclass
import json
from datetime import datetime
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
    early_stopping_patience: int = 5
    grad_clip: float = 1.0
    scheduler: str = "plateau"  # plateau, cosine, or none


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
        self.start_epoch = 1
        self.history = {"train_loss": [], "train_acc": [], "train_f1": [],
                        "val_loss": [], "val_acc": [], "val_f1": [], "epochs": []}

        # Setup logging
        self.log_file = self.output_dir / "training_log.txt"
        with open(self.log_file, "w") as f:
            f.write(
                f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Batch size: {self.config.batch_size}\n")
            f.write(f"Learning rate: {self.config.lr}\n")
            f.write(f"Epochs: {self.config.epochs}\n")
            f.write(f"Weight decay: {self.config.weight_decay}\n")
            f.write("="*80 + "\n\n")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(self.model.parameters(
        ), lr=self.config.lr, weight_decay=self.config.weight_decay)

        # Setup learning rate scheduler
        if self.config.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
        elif self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs, eta_min=1e-6
            )
        else:
            self.scheduler = None

        self.patience_counter = 0

        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_loaders(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        state_dict = payload.get("state_dict") if isinstance(
            payload, dict) else None
        if state_dict is None:
            state_dict = payload
        self.model.load_state_dict(state_dict)

        if isinstance(payload, dict):
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
            if "best_f1" in payload:
                self.best_f1 = float(payload["best_f1"])
            if "epoch" in payload:
                self.start_epoch = int(payload["epoch"]) + 1
            if "history" in payload:
                self.history = payload["history"]

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _log(self, message: str, file_only: bool = False) -> None:
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"

        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

        if not file_only:
            print(message)

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
                # Gradient clipping to prevent exploding gradients
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()

            epoch_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

            if train and step % self.config.log_every == 0:
                self._log(f"step {step} loss {loss.item():.4f}")

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        metrics = binary_metrics(logits, targets)
        return epoch_loss / max(1, len(loader)), metrics

    def fit(self) -> None:
        self._log(
            f"Starting training from epoch {self.start_epoch} to {self.config.epochs}")
        self._log("="*80)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss, train_metrics = self._run_epoch(
                self.train_loader, train=True)
            val_loss, val_metrics = self._run_epoch(
                self.val_loader, train=False)

            epoch_msg = (
                f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
                f"train_acc {train_metrics['accuracy']:.4f} val_acc {val_metrics['accuracy']:.4f} "
                f"train_f1 {train_metrics['f1']:.4f} val_f1 {val_metrics['f1']:.4f}"
            )

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            lr_msg = f" lr {current_lr:.6f}"
            self._log(epoch_msg + lr_msg)

            # Save history
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])

            # Update learning rate scheduler
            if self.scheduler is not None:
                if self.config.scheduler == "plateau":
                    self.scheduler.step(val_metrics["f1"])
                else:
                    self.scheduler.step()

            if self.config.save_last:
                last_path = self.output_dir / "checkpoint_last.pt"
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "best_f1": self.best_f1,
                        "history": self.history,
                    },
                    last_path,
                )

            if self.config.save_best and val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self.patience_counter = 0  # Reset patience counter
                best_path = self.output_dir / "checkpoint_best.pt"
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "best_f1": self.best_f1,
                        "history": self.history,
                    },
                    best_path,
                )
                self._log(
                    f"New best model saved with val_f1: {self.best_f1:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    self._log(
                        f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break

        # Save history to JSON
        self._log("="*80)
        self._log("Training completed!")
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        self._log(f"Training history saved to {history_path}")
        self._log(f"Training log saved to {self.log_file}")

    def test(self) -> None:
        self._log("Running test evaluation...")
        test_loss, test_metrics = self._run_epoch(
            self.test_loader, train=False)
        test_msg = (
            f"test_loss {test_loss:.4f} test_acc {test_metrics['accuracy']:.4f} "
            f"test_f1 {test_metrics['f1']:.4f}"
        )
        self._log(test_msg)
