# data_module.py
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms  # type: ignore


class CIFAR10DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for CIFAR10 dataset.

    Args:
        data_dir: Directory to store the dataset
        batch_size: Number of samples per batch
        num_workers: Number of workers for DataLoader
        val_split: Fraction of training data to use for validation
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Define transforms with data augmentation
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self) -> None:
        """Download the CIFAR10 dataset."""
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """
        Set up the dataset splits.

        Args:
            stage: Current stage ('fit' or 'test')
        """
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.train_transform
            )
            train_size = int((1 - self.val_split) * len(cifar_full))
            val_size = len(cifar_full) - train_size
            self.train_data, self.val_data = random_split(
                cifar_full,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.test_data = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
