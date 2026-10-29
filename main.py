"""Training entry point.

Usage:
    python main.py
    python main.py --epochs 50 --lr 0.0003 --batch_size 64
    python main.py --no_wandb
"""
import argparse

import torch

from config import ViTConfig
from dataset import get_dataloaders
from engine import evaluate, fit
from model import build_model, count_parameters
from utils import (
    checkpoint_path, ensure_dirs, load_checkpoint, per_class_accuracy,
    plot_confusion_matrix, plot_history, plot_sample_predictions,
    save_metrics, set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train a ViT on CIFAR10")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--patch_size", type=int)
    p.add_argument("--embed_dim", type=int)
    p.add_argument("--depth", type=int)
    p.add_argument("--num_heads", type=int)
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity", type=str)
    p.add_argument("--no_wandb", action="store_true", help="disable W&B tracking")
    return p.parse_args()


def build_config(args) -> ViTConfig:
    cfg = ViTConfig()
    # Only override what was actually passed on the command line
    for key in ("epochs", "batch_size", "lr", "patch_size", "embed_dim",
                "depth", "num_heads", "wandb_project", "wandb_entity"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    if args.no_wandb:
        cfg.use_wandb = False
    return cfg


def main():
    cfg = build_config(parse_args())
    set_seed(cfg.seed)
    ensure_dirs(cfg)
    device = torch.device(cfg.device)
    print(f"Run: {cfg.run_name} | device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    num_params = count_parameters(model)
    print(f"Parameters: {num_params / 1e6:.2f}M")

    # ---------------- W&B init ----------------
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
            config=cfg.to_dict(),
            job_type="train",
        )
        wandb.config.update({"num_parameters": num_params})
        wandb.watch(model, log="gradients", log_freq=100)

    # ---------------- Train ----------------
    print("Starting Training...")
    result = fit(model, train_loader, val_loader, cfg)

    # ---------------- Final evaluation on the BEST checkpoint ----------------
    print("Starting Testing (best checkpoint, test set touched ONCE)...")
    best_model, _, ckpt = load_checkpoint(checkpoint_path(cfg), cfg.device)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(
        best_model, test_loader, criterion, device, return_preds=True
    )
    print(f"Accuracy: {test_acc:.2f}% (best epoch {ckpt['epoch']})")

    class_acc = per_class_accuracy(cfg, preds, labels)

    # ---------------- Save everything to metrics/ ----------------
    final = {"test_loss": test_loss, "test_acc": test_acc, "per_class_acc": class_acc}
    metrics_path = save_metrics(
        cfg, result["history"], num_params,
        result["best_epoch"], result["best_val_acc"], final, result["train_time_sec"],
    )
    curves_path = plot_history(cfg, result["history"])
    cm_path = plot_confusion_matrix(cfg, preds, labels)
    samples_path = plot_sample_predictions(cfg, best_model, test_loader)
    print(f"Metrics saved to: {metrics_path}")

    # ---------------- Push results to W&B ----------------
    if cfg.use_wandb:
        import wandb
        wandb.summary["best_val_acc"] = result["best_val_acc"]
        wandb.summary["best_epoch"] = result["best_epoch"]
        wandb.summary["test_acc"] = test_acc
        wandb.summary["train_time_sec"] = result["train_time_sec"]
        wandb.log({
            "plots/training_curves": wandb.Image(str(curves_path)),
            "plots/confusion_matrix": wandb.Image(str(cm_path)),
            "plots/sample_predictions": wandb.Image(str(samples_path)),
            "per_class_acc": wandb.Table(
                columns=["class", "accuracy"],
                data=[[k, v] for k, v in class_acc.items()],
            ),
        })

        # Model Artifact = versioned model file, visible in the W&B UI
        artifact = wandb.Artifact(
            name=f"vit-cifar10-{cfg.patch_size}p",
            type="model",
            description="Vision Transformer (from scratch) trained on CIFAR10",
            metadata={
                "test_acc": test_acc,
                "best_val_acc": result["best_val_acc"],
                "best_epoch": result["best_epoch"],
                "num_parameters": num_params,
                **cfg.to_dict(),
            },
        )
        artifact.add_file(str(checkpoint_path(cfg)))
        artifact.add_file(str(metrics_path))
        wandb.log_artifact(artifact, aliases=["latest", "best"])
        wandb.finish()


if __name__ == "__main__":
    main()
