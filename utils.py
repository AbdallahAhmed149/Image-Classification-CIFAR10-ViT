"""Persistence helpers: checkpoints -> models/, metrics + plots -> metrics/."""
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ViTConfig


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: ViTConfig):
    Path(cfg.models_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.metrics_dir).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ models/
def checkpoint_path(cfg: ViTConfig) -> Path:
    return Path(cfg.models_dir) / f"{cfg.run_name}.pth"


def save_checkpoint(model, cfg: ViTConfig, epoch: int, val_acc: float, path=None):
    """Saves weights AND the config used to build the model, so the model
    can be rebuilt later without guessing hyperparameters."""
    path = Path(path) if path else checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )
    return path


def load_checkpoint(path, device="cpu"):
    """Returns (model, cfg, checkpoint_dict) rebuilt from a saved file."""
    from model import build_model  # local import to avoid circular imports

    ckpt = torch.load(path, map_location=device)
    saved = ckpt["config"]
    # tuples become lists after saving -> convert back where needed
    for key in ("mean", "std", "classes"):
        if key in saved:
            saved[key] = tuple(saved[key])
    cfg = ViTConfig(**saved)
    cfg.device = device

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


# ---------------------------------------------------------------- metrics/
def save_metrics(cfg: ViTConfig, history: dict, num_params: int,
                 best_epoch: int, best_val_acc: float, final: dict,
                 train_time_sec: float):
    """One JSON file with EVERYTHING about the run."""
    path = Path(cfg.metrics_dir) / f"{cfg.run_name}_metrics.json"
    payload = {
        "run_name": cfg.run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": cfg.device,
        "num_parameters": num_params,
        "train_time_sec": round(train_time_sec, 1),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final": final,
        "config": cfg.to_dict(),
        "history": history,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def plot_history(cfg: ViTConfig, history: dict):
    """Loss + accuracy curves (train vs val)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(cfg.run_name)
    fig.tight_layout()
    path = Path(cfg.metrics_dir) / f"training_curves_{cfg.run_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------- evaluation plots (metrics/)
def per_class_accuracy(cfg: ViTConfig, preds, labels) -> dict:
    preds, labels = np.asarray(preds), np.asarray(labels)
    result = {}
    for i, name in enumerate(cfg.classes):
        mask = labels == i
        result[name] = round(float((preds[mask] == i).mean() * 100), 2)
    return result


def plot_confusion_matrix(cfg: ViTConfig, preds, labels):
    preds, labels = np.asarray(preds), np.asarray(labels)
    n = len(cfg.classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cfg.classes, rotation=45, ha="right")
    ax.set_yticklabels(cfg.classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {cfg.run_name}")
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    path = Path(cfg.metrics_dir) / f"confusion_matrix_{cfg.run_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@torch.no_grad()
def plot_sample_predictions(cfg: ViTConfig, model, loader, n=16):
    """Grid of test images with predicted vs true label (green=correct, red=wrong)."""
    device = torch.device(cfg.device)
    model.eval()
    inputs, labels = next(iter(loader))
    inputs, labels = inputs[:n], labels[:n]
    preds = model(inputs.to(device)).argmax(1).cpu()

    mean = torch.tensor(cfg.mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.std).view(1, 3, 1, 1)
    imgs = (inputs * std + mean).clamp(0, 1)

    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for ax, img, p, t in zip(axes.flat, imgs, preds, labels):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ok = p.item() == t.item()
        ax.set_title(f"{cfg.classes[p]} ({cfg.classes[t]})", fontsize=8,
                     color="green" if ok else "red")
        ax.axis("off")
    for ax in list(axes.flat)[len(imgs):]:
        ax.axis("off")
    fig.suptitle("Predicted (True)")
    fig.tight_layout()
    path = Path(cfg.metrics_dir) / f"sample_predictions_{cfg.run_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
