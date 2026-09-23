"""
NeuroLens - brain tumour classification from MRI
Expects Training/ and Testing/ folders next to this script,
each containing: glioma, meningioma, notumor, pituitary
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

IMG = 224
BATCH = 32
EPOCHS_HEAD = 4
EPOCHS_FT = 20
VAL_FRACTION = 0.15

# ---------- dataset location ----------
ROOT = Path(__file__).resolve().parent
if not (ROOT / "Training").is_dir():
    raise SystemExit(f"No Training/ folder found in {ROOT}")
print("dataset:", ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

# ---------- transforms ----------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG, IMG)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def main():
    # ---------- data ----------
    train_base = datasets.ImageFolder(ROOT / "Training", transform=train_tf)
    val_base = datasets.ImageFolder(ROOT / "Training", transform=eval_tf)
    classes = train_base.classes
    print("classes:", classes)

    n_val = int(VAL_FRACTION * len(train_base))
    n_train = len(train_base) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = random_split(range(len(train_base)), [n_train, n_val], generator=gen)

    train_ds = torch.utils.data.Subset(train_base, list(train_idx))
    val_ds = torch.utils.data.Subset(val_base, list(val_idx))
    test_ds = datasets.ImageFolder(ROOT / "Testing", transform=eval_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH, num_workers=0)
    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    # ---------- model ----------
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    def run_epoch(dl, optimizer=None):
        training = optimizer is not None
        model.train() if training else model.eval()
        loss_sum = correct = total = 0
        with torch.set_grad_enabled(training):
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return loss_sum / total, correct / total

    def train_phase(name, epochs, optimizer, best):
        for ep in range(1, epochs + 1):
            tl, ta = run_epoch(train_dl, optimizer)
            vl, va = run_epoch(val_dl)
            flag = ""
            if va > best:
                best = va
                torch.save({"state_dict": model.state_dict(), "classes": classes}, ROOT / "best1.pt")
                flag = "  <- saved"
            print(f"[{name}] epoch {ep}/{epochs}  "
                  f"train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}{flag}")
        return best

    # phase 1 - train the new head only
    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    best = train_phase("head", EPOCHS_HEAD, opt, 0.0)

    # phase 2 - unfreeze last blocks, fine-tune
    for p in model.features[-3:].parameters():
        p.requires_grad = True
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    best = train_phase("fine", EPOCHS_FT, opt, best)

    # ---------- final test ----------
    ckpt = torch.load(ROOT / "best1.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    tl, ta = run_epoch(test_dl)
    print(f"\nbest val acc  {best:.4f}")
    print(f"TEST  loss {tl:.3f}  acc {ta:.4f}")
    print("saved:", ROOT / "best1.pt")


if __name__ == "__main__":
    main()