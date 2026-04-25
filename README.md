# 🧠 CIFAR-10 Image Classification with Vision Transformer (ViT)

Image classification on the **CIFAR-10** dataset using a **Vision Transformer (ViT)** implemented in **PyTorch**. 🚀

---

## ✨ Features

- ✅ End-to-end **ViT** implementation (patch embedding + Transformer encoder + classification head)
- 🧩 **Patch Embedding** via `Conv2d` projection
- 🧠 `nn.TransformerEncoder`-based backbone
- 🔁 Training loop + evaluation loop included
- 📉 **CosineAnnealingLR** learning-rate scheduler
- 🧼 CIFAR-10 standard normalization + simple data augmentation

---

## 📁 Project Structure

- `Image_Classification_Vision_Transformer.py` — main training + testing script
- `README.md` — project documentation

---

## 🧰 Requirements

- Python 3.9+ (recommended)
- PyTorch
- torchvision

Install dependencies:

```bash
pip install torch torchvision
```

---

## ▶️ How to Run

Train and evaluate the model:

```bash
python Image_Classification_Vision_Transformer.py
```

Notes:
- The script will automatically download **CIFAR-10** into `./data` on first run. 📦
- It will train for **10 epochs** by default.

---

## ⚙️ Configuration (Key Hyperparameters)

These are defined at the top of `Image_Classification_Vision_Transformer.py`:

- **batch size**: `32`
- **learning rate**: `3e-4`
- **epochs**: `10`
- **patch size**: the model is instantiated with `patch_size=4`
- **optimizer**: Adam + `weight_decay=0.01`
- **scheduler**: Cosine Annealing (`T_max=50`)

---

## 🧠 Model Overview

The model follows the standard ViT recipe:

1. Split the image into fixed-size patches (implemented using a `Conv2d` projection)
2. Add a learnable **[CLS] token**
3. Add learnable **positional embeddings**
4. Pass tokens through a stack of Transformer encoder blocks
5. Classify using the final representation of the **CLS token**

---

## 📊 Results

- Reported in the script: **~47% accuracy** on CIFAR-10.

This is a solid starting point, and there are many ways to improve performance (see below). 📈

---

## 🔥 Ideas to Improve Accuracy

If you want higher accuracy, consider:

- 🏋️ Train longer (e.g., 50–200 epochs)
- 📈 Use warmup + cosine decay (common for ViT training)
- 🧪 Tune hyperparameters (LR, weight decay, depth/heads/embedding dim)
- 🧰 Add regularization (label smoothing, dropout, stochastic depth)
- 🧠 Use stronger augmentation (RandAugment, MixUp, CutMix)
- 💾 Save best checkpoint and evaluate the best model

---

## 📝 Notes

- The code is written to be easy to read and modify for learning / experimentation.
- ViTs can be data-hungry; CIFAR-10 is small, so careful tuning and augmentation help a lot.

---

## 🤝 Contributing

Contributions are welcome!

- Open an issue to propose improvements
- Submit a PR for fixes, refactors, or new features

---

## 📜 License

No license file is currently included in the repository.

If you plan to share or reuse this work broadly, consider adding a license (e.g., MIT, Apache-2.0). 🧾
