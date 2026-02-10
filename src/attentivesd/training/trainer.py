import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from attentivesd.utils.metrics import binary_metrics


class Trainer:
    def __init__(self, model: nn.Module, data_module, config: dict):
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training config
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("lr", 1e-3)
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = self.output_dir / "training_log.txt"
        self._setup_logging()

        # Optimizer and criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Use standard BCE loss without pos_weight - relies on balanced sampling
        self.criterion = nn.BCEWithLogitsLoss()

        self.log_message("=" * 80)
        self.log_message(f"Model: {config.get('mode', 'unknown')}")
        self.log_message(f"Output directory: {self.output_dir}")
        self.log_message("=" * 80)
        self.log_message("")
        self.log_message(
            "Using standard BCE loss (no pos_weight) - relying on balanced data sampling")
        self.log_message("")

        # Learning rate scheduler
        scheduler_type = config.get("scheduler", "plateau")
        if scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=3, verbose=False
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, verbose=False
            )
        else:
            self.scheduler = None

        # Early stopping
        self.patience = config.get("early_stopping_patience", 10)
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0
        self.early_stop = False

        # Gradient clipping
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
        }

        # Current epoch for resume
        self.start_epoch = 1

    def _setup_logging(self):
        """Setup logging to file and console."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(self.log_file, mode="w"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def log_message(self, message: str):
        """Log message to both file and console."""
        self.logger.info(message)

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
        for step, (x, y) in enumerate(pbar, 1):
            x = x.to(self.device)
            y = y.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(x).squeeze()
            loss = self.criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if step % 50 == 0:
                self.log_message(f"step {step} loss {loss.item():.4f}")

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        metrics = binary_metrics(np.array(all_labels), np.array(all_preds))
        return avg_loss, metrics["accuracy"], metrics["f1"]

    def _validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device).float()

                logits = self.model(x).squeeze()
                loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        metrics = binary_metrics(np.array(all_labels), np.array(all_preds))
        return avg_loss, metrics["accuracy"], metrics["f1"]

    def fit(self):
        """Train the model."""
        train_loader, val_loader, test_loader = self.data_module.get_loaders(
            batch_size=self.batch_size,
            num_workers=self.config.get("num_workers", 0)
        )

        self.log_message(
            f"Starting training from epoch {self.start_epoch} to {self.epochs}")
        self.log_message("=" * 80)

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss, train_acc, train_f1 = self._train_epoch(
                train_loader, epoch)
            val_loss, val_acc, val_f1 = self._validate(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.log_message(
                f"epoch {epoch} "
                f"train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
                f"train_acc {train_acc:.4f} val_acc {val_acc:.4f} "
                f"train_f1 {train_f1:.4f} val_f1 {val_f1:.4f} "
                f"lr {current_lr:.6f}"
            )

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                self.save_checkpoint(
                    self.output_dir / "checkpoint_best.pt", epoch)
                self.log_message(
                    f"New best model saved with val_f1: {val_f1:.4f}")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                self.log_message(
                    f"Early stopping triggered after {epoch} epochs")
                self.early_stop = True
                break

        # Save last checkpoint
        self.save_checkpoint(self.output_dir / "checkpoint_last.pt", epoch)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        self.log_message("=" * 80)
        self.log_message("Training completed!")
        self.log_message(f"Training history saved to {history_path}")
        self.log_message(f"Training log saved to {self.log_file}")

    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_f1": self.best_val_f1,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: Path):
        """Load model checkpoint for resume."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_f1 = checkpoint["best_val_f1"]
        self.history = checkpoint["history"]
        self.start_epoch = checkpoint["epoch"] + 1
        self.log_message(
            f"Resumed from epoch {checkpoint['epoch']}, best val_f1: {self.best_val_f1:.4f}")
