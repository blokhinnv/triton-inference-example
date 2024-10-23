from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for CIFAR10 classification.

    The network consists of three convolutional layers followed by
    three fully connected layers, with batch normalization and dropout.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class CIFAR10Module(pl.LightningModule):
    """
    PyTorch Lightning module for training CIFAR10 classifier.

    Args:
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for optimization
    """

    def __init__(
        self, learning_rate: float = 0.001, weight_decay: float = 1e-4
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Initialize metrics
        num_classes = 10
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch

        Returns:
            Training loss
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Update and log metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing validation metrics
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update all validation metrics
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(preds, y)

        # Log metrics
        # Using metrics in combination with self.log, then setting on_epoch=True will also internally
        # call . compute() at the end of the epoch.
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)

        return {"val_loss": loss, "val_preds": preds, "val_targets": y}

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
