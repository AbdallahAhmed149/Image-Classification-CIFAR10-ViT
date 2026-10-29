"""Central configuration for the CIFAR10 Vision Transformer project.

Every hyperparameter lives here, so main.py / predict.py / wandb all read
from ONE place (same idea as AEConfig in the VAE project).
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class ViTConfig:
    # ---- Data ----
    data_dir: str = "./data"
    batch_size: int = 32
    val_size: int = 5000
    num_workers: int = 2
    img_size: int = 32
    in_channels: int = 3
    num_classes: int = 10
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2023, 0.1994, 0.2010)

    # ---- Model ----
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.0

    # ---- Training ----
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.01

    # ---- Outputs ----
    models_dir: str = str(BASE_DIR / "models")
    metrics_dir: str = str(BASE_DIR / "metrics")

    # ---- Weights & Biases ----
    use_wandb: bool = True
    wandb_project: str = "vit-cifar10"
    wandb_entity: Optional[str] = None  # your W&B username/team (None = default)

    # ---- Misc ----
    seed: int = 42
    classes: tuple = (
        "plane", "car", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    @property
    def run_name(self) -> str:
        """Used in file names so every experiment is identifiable,
        e.g. ViT_CIFAR10_P4_E10_B32"""
        return f"ViT_CIFAR10_P{self.patch_size}_E{self.epochs}_B{self.batch_size}"

    def to_dict(self) -> dict:
        return asdict(self)
