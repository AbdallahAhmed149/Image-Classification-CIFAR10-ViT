# CIFAR-10 Image Classification with Vision Transformer

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-5B8FF9)](https://www.cs.toronto.edu/~kriz/cifar.html)

A production-oriented, from-scratch **Vision Transformer (ViT)** for classifying 32×32 RGB images across the 10 CIFAR-10 categories. The repository includes deterministic data splitting, configurable training, validation-based checkpoint selection, held-out test evaluation, per-class metrics, diagnostic plots, and optional Weights & Biases experiment tracking.

## Abstract / Overview

This project applies a Transformer encoder to small-image classification without relying on a pre-trained backbone. Each image is divided into non-overlapping 4×4 patches, projected into a 256-dimensional token space, augmented with a learnable classification token and positional embeddings, and processed by six Transformer blocks with eight attention heads. The final `[CLS]` representation is mapped to the ten CIFAR-10 classes.

The training pipeline is designed for reproducibility and trustworthy evaluation:

- The 50,000-image CIFAR-10 training split is partitioned into **45,000 training** and **5,000 validation** examples using a fixed seed.
- The **10,000-image test split remains untouched** until final evaluation.
- The best checkpoint is selected by validation accuracy, then evaluated once on the test set.
- Metrics, training curves, a confusion matrix, sample predictions, and the exact run configuration are persisted locally.

The latest recorded 30-epoch run achieved **77.78% test accuracy** and **78.46% best validation accuracy** at epoch 27. It used the default ViT configuration, completed in **1,448.7 seconds (24.1 minutes)** on CUDA, and contained **4,771,082 trainable parameters**.

## Architecture & Pipeline

```text
CIFAR-10 download
       │
       ├── Train transforms: random horizontal flip + padded random crop + normalization
       ├── Validation transforms: normalization only
       └── Test transforms: normalization only
              │
       Seeded 45k / 5k train-validation split
              │
       32×32 image → 4×4 patch projection → 64 patch tokens
              │
       [CLS] token + positional embeddings
              │
       6 × Pre-LN Transformer blocks
       (multi-head self-attention + GELU MLP)
              │
       LayerNorm → [CLS] representation → 10-class linear head
              │
       Validation checkpoint selection → final test evaluation
              │
       models/*.pth + metrics/*.json + diagnostic PNGs
```

### Training process

1. Download CIFAR-10 automatically into `./data`.
2. Apply augmentation only to the training view.
3. Optimize cross-entropy with **AdamW** and cosine-annealed learning rate.
4. Log batch and epoch metrics locally and optionally to W&B.
5. Save the best validation checkpoint together with the configuration required to rebuild the model.

### Inference process

`predict.py` restores the saved architecture and weights from a checkpoint. It can classify one RGB image with top-3 probabilities or evaluate a checkpoint across the complete test set and regenerate diagnostic plots.

## Performance Metrics

| Metric | Value / Behavior |
|---|---:|
| Test accuracy | **77.78%** |
| Test loss | **0.6666** |
| Best validation accuracy | **78.46%** |
| Best validation epoch | **27** |
| Model parameters | **4,771,082** |
| Training time | **1,448.7 sec (24.1 min)** |
| Classification classes | **10** |
| Input resolution | **32×32 RGB** |
| Training data | **45,000 images** |
| Validation data | **5,000 images** |
| Held-out test data | **10,000 images** |
| Patch size | **4×4** |
| Tokens per image | **64 patches + 1 `[CLS]` token** |
| Transformer depth | **6 blocks** |
| Attention heads | **8** |
| Embedding dimension | **256** |
| Default batch size | **32** |
| Default training duration | **30 epochs** |
| Default optimizer | **AdamW**, learning rate `3e-4`, weight decay `0.01` |
| Learning-rate schedule | **CosineAnnealingLR** |
| Reproducibility seed | **42** |
| Operational latency / throughput | Not benchmarked in the repository |

### Per-class test accuracy

| Class | Accuracy |
|---|---:|
| Plane | 78.8% |
| Car | 90.4% |
| Bird | 73.0% |
| Cat | 60.0% |
| Deer | 70.1% |
| Dog | 69.3% |
| Frog | 80.0% |
| Horse | 84.6% |
| Ship | 85.8% |
| Truck | 85.8% |

These values come from `metrics/ViT_CIFAR10_P4_E30_B32_metrics.json`. Accuracy and training time are hardware-, seed-, and configuration-dependent; use that JSON file as the authoritative record for future runs.

## Tech Stack

- **Python 3.9+**
- **PyTorch** and **torchvision** for modeling, optimization, and CIFAR-10
- **NumPy** for metric calculations
- **Pillow** for single-image inference
- **Matplotlib** for training curves, confusion matrices, and sample predictions
- **Weights & Biases** for optional experiment tracking and model artifacts

## Repository Structure

| Path | Responsibility |
|---|---|
| `config.py` | Centralized dataclass configuration and run naming |
| `dataset.py` | CIFAR-10 download, transforms, seeded split, and data loaders |
| `model.py` | Patch embedding, attention, Transformer blocks, and ViT classifier |
| `engine.py` | Training and evaluation loops, checkpoint selection, and W&B logging |
| `main.py` | End-to-end training and final test evaluation entry point |
| `predict.py` | Checkpoint-based test evaluation and single-image inference |
| `utils.py` | Checkpoint persistence, metrics, plots, and reproducibility helpers |
| `models/` | Generated model checkpoints |
| `metrics/` | Generated JSON metrics and diagnostic images |

## Getting Started

### Prerequisites

- Python 3.9 or newer
- `pip`
- Optional: CUDA-enabled PyTorch installation and an NVIDIA-compatible environment
- Optional: a W&B account if experiment tracking is enabled

### Local setup

```bash
git clone https://github.com/AbdallahAhmed149/Image-Classification-CIFAR10-ViT.git
cd Image-Classification-CIFAR10-ViT

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first training or inference run downloads CIFAR-10 into `./data`. Generated checkpoints and reports are written to `models/` and `metrics/`.

### Environment variables

W&B is enabled by default in `config.py`. Authenticate before training if you want remote tracking:

```bash
export WANDB_API_KEY="your_api_key"
wandb login
```

Alternatively, disable W&B entirely for an offline/local run:

```bash
python main.py --no_wandb
```

The repository currently does not include a `Dockerfile` or container orchestration configuration; local virtual-environment execution is the supported setup path.

## Training & Inference

### Train with defaults

```bash
python main.py
```

### Reproduce a controlled run

```bash
python main.py \
  --epochs 30 \
  --batch_size 32 \
  --lr 0.0003 \
  --patch_size 4 \
  --embed_dim 256 \
  --depth 6 \
  --num_heads 8 \
  --no_wandb
```

The command prints the selected device and parameter count, saves the best validation checkpoint as `models/ViT_CIFAR10_P4_E30_B32.pth`, and writes the run record and plots under `metrics/`.

### Evaluate the default checkpoint

```bash
python predict.py
```

### Evaluate a specific checkpoint

```bash
python predict.py --ckpt models/ViT_CIFAR10_P4_E30_B32.pth
```

### Classify a single image

```bash
python predict.py \
  --ckpt models/ViT_CIFAR10_P4_E30_B32.pth \
  --image path/to/image.jpg
```

The image is converted to RGB, resized to 32×32, normalized with CIFAR-10 statistics, and reported with the top three class probabilities.

## Reproducibility and Experiment Outputs

- Set the seed with `ViTConfig.seed` (default: `42`).
- Keep the generated checkpoint and matching metrics JSON together; the checkpoint stores the full model configuration.
- Compare `best_val_acc`, `final.test_acc`, `num_parameters`, `train_time_sec`, and the saved `history` fields between runs.
- The test set is intentionally excluded from model selection to reduce evaluation leakage.

## Contributing

Contributions are welcome. For substantial changes:

1. Open an issue describing the proposed change or experiment.
2. Keep configuration changes explicit and reproducible.
3. Preserve the train/validation/test separation.
4. Add or update tests and documentation where behavior changes.
5. Open a focused pull request with the command used to validate the change and any before/after metrics.

## License

This project is released under the [MIT License](LICENSE).
