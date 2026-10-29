"""Load a trained checkpoint from models/ and evaluate / predict.

Usage:
    python predict.py                          # full test-set evaluation + plots
    python predict.py --ckpt models/xxx.pth    # specific checkpoint
    python predict.py --image path/to/img.jpg  # classify one image
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import ViTConfig
from dataset import get_dataloaders
from engine import evaluate
from utils import (
    checkpoint_path, ensure_dirs, load_checkpoint, per_class_accuracy,
    plot_confusion_matrix, plot_sample_predictions,
)


def parse_args():
    p = argparse.ArgumentParser(description="ViT inference")
    p.add_argument("--ckpt", type=str, help="path to a .pth checkpoint")
    p.add_argument("--image", type=str, help="classify a single image file")
    return p.parse_args()


@torch.no_grad()
def predict_image(model, cfg: ViTConfig, image_path: str, topk=3):
    tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(cfg.device)
    probs = model(x).softmax(dim=1)[0]
    top_p, top_i = probs.topk(topk)
    return [(cfg.classes[i], float(p) * 100) for p, i in zip(top_p, top_i)]


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt) if args.ckpt else checkpoint_path(ViTConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg, ckpt = load_checkpoint(ckpt_path, device)
    ensure_dirs(cfg)
    print(f"Loaded {ckpt_path.name} (epoch {ckpt['epoch']}, saved val acc {ckpt['val_acc']:.2f}%)")

    if args.image:
        for name, prob in predict_image(model, cfg, args.image):
            print(f"{name}: {prob:.2f}%")
        return

    _, _, test_loader = get_dataloaders(cfg)
    loss, acc, preds, labels = evaluate(
        model, test_loader, torch.nn.CrossEntropyLoss(), torch.device(device),
        return_preds=True,
    )
    print(f"Test loss {loss:.4f} | Accuracy {acc:.2f}%")
    for name, a in per_class_accuracy(cfg, preds, labels).items():
        print(f"  {name:>6}: {a:.2f}%")

    print("Saved:", plot_confusion_matrix(cfg, preds, labels))
    print("Saved:", plot_sample_predictions(cfg, model, test_loader))


if __name__ == "__main__":
    main()
