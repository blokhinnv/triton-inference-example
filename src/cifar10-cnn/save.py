import os
from pathlib import Path

import torch
from model import CIFAR10Module


def main(lightning_checkpoint_path: Path, traced_checkpoint_path: Path) -> None:
    model = CIFAR10Module.load_from_checkpoint(lightning_checkpoint_path).model
    model.eval()
    model.to(device="cpu")
    example = torch.rand(1, 3, 32, 32)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(traced_checkpoint_path)
    print("Output shape: ", model(example).shape)


if __name__ == "__main__":
    # Get the latest checkpoint
    checkpoints = sorted(Path("./checkpoints/").iterdir(), key=os.path.getmtime)
    lightning_checkpoint_path = Path(checkpoints[-1])

    # Trace the model to use pytorch_libtorch backend for triton
    Path("./traced").mkdir(exist_ok=True)
    traced_checkpoint_path = Path(f"./traced/{lightning_checkpoint_path.stem}.pt")
    main(lightning_checkpoint_path, traced_checkpoint_path)
