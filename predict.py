import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

ROOT = Path(__file__).resolve().parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ckpt = torch.load(ROOT / "best1.pt", map_location=DEVICE)
classes = ckpt["classes"]

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(path):
    img = Image.open(path)
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    order = probs.argsort(descending=True)
    print(f"\n{Path(path).name}")
    for i in order:
        print(f"  {classes[i]:12s} {probs[i]*100:5.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python predict.py <image> [image2 ...]")
    for p in sys.argv[1:]:
        predict(p)