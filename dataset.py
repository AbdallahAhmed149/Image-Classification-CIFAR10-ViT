"""CIFAR10 download, augmentation, train/val split and DataLoaders."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import ViTConfig


def get_transforms(cfg: ViTConfig):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=cfg.img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    return tf_train, tf_eval


def get_dataloaders(cfg: ViTConfig):
    """Returns (train_loader, val_loader, test_loader).

    train/val come from the 50k CIFAR10 train set (45k/5k by default).
    The test set (10k) is kept untouched for the final evaluation.
    """
    tf_train, tf_eval = get_transforms(cfg)

    # Two views of the same training images: augmented (train) / clean (val)
    full_train_aug = datasets.CIFAR10(cfg.data_dir, train=True, transform=tf_train, download=True)
    full_train_clean = datasets.CIFAR10(cfg.data_dir, train=True, transform=tf_eval, download=True)
    test_data = datasets.CIFAR10(cfg.data_dir, train=False, transform=tf_eval, download=True)

    # Fixed, seeded split -> same train/val images in every run
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(full_train_aug), generator=generator).tolist()
    val_idx, train_idx = perm[:cfg.val_size], perm[cfg.val_size:]

    train_data = Subset(full_train_aug, train_idx)
    val_data = Subset(full_train_clean, val_idx)

    pin = cfg.device == "cuda"
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=pin)
    train_loader = DataLoader(train_data, shuffle=True, **kw)
    val_loader = DataLoader(val_data, shuffle=False, **kw)
    test_loader = DataLoader(test_data, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
