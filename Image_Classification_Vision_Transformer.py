# ( Building Vision Transformer (ViT) for Image Classification )

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 3e-4
patch_size = 8

tf_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = datasets.CIFAR10(root="./data", train=True, transform=tf_train, download=True)
test_data = datasets.CIFAR10(root="./data", train=False, transform=tf_test, download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck",)

# The ViT Model (Optimized for CIFAR)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_size=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Project patches to vectors
        self.proj = nn.Conv2d(
            in_channels, embed_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (Batch, Embed_Dim, Grid_H, Grid_W)
        x = x.flatten(2)  # (Batch, Embed_Dim, N_Patches)
        x = x.transpose(1, 2)  # (Batch, N_Patches, Embed_Dim)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10,
                embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels=3, embed_size=embed_dim)

        # Special Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))

        # Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Output Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add Position Embedding
        x = x + self.pos_embed

        # Pass through Transformer
        x = self.encoder(x)

        # Classification on CLS token only
        cls_output = x[:, 0]
        x = self.head(self.norm(cls_output))
        return x

model = VisionTransformer(patch_size=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)
schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) # Adjusts LR smoothly

# Check number of parameters (ViTs are heavy!)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Training
print(f"Starting Training...")
epochs = 10

for epoch in range(epochs):
    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}")
    schedular.step()

# Testing
print("Starting Testing...")
total = 0
correct = 0

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        
        total += inputs.size(0)
        correct += (preds == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy}") # accuracy = 47%
