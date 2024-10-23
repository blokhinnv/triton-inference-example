# train.py
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data import CIFAR10DataModule  # type: ignore
from model import CIFAR10Module  # type: ignore


def main(args: Namespace) -> None:
    """Main training routine."""
    # Set up data module
    data_module = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Set up model
    model = CIFAR10Module(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="./checkpoints",
            monitor='val_acc',
            mode='max',
            save_top_k=3,
            filename='cifar10-{epoch:02d}-{val_acc:.2f}'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ]
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=callbacks,
        deterministic=True
    )
    
    # Train the model
    trainer.fit(model, data_module)   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    
    args = parser.parse_args()
    main(args)