"""Training / evaluation loops + Weights & Biases logging."""
import time

import torch
from torch import nn, optim

from config import ViTConfig
from utils import save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch, cfg: ViTConfig, log_every=10):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += inputs.size(0)

        if batch_idx % log_every == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss {loss.item():.4f}")
            if cfg.use_wandb:
                import wandb
                wandb.log({"batch/train_loss": loss.item()})

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(1)

        running_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

        if return_preds:
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss, acc = running_loss / total, 100.0 * correct / total
    if return_preds:
        return avg_loss, acc, torch.cat(all_preds), torch.cat(all_labels)
    return avg_loss, acc


def fit(model, train_loader, val_loader, cfg: ViTConfig):
    """Full training run. The best model (by VALIDATION accuracy) is saved to
    models/ automatically. The test set is never used here."""
    device = torch.device(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # T_max = epochs so the cosine schedule completes exactly at the last epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_epoch = 0.0, 0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"== Epoch {epoch}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}% | lr {current_lr:.2e}"
        )

        if cfg.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "lr": current_lr,
            })

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            save_checkpoint(model, cfg, epoch, val_acc)
            print(f"   -> new best val acc ({best_acc:.2f}%) saved to models/")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "train_time_sec": time.time() - start,
    }
