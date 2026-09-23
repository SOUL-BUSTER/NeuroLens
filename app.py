from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
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

PRETTY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumour",
    "tggi": "Pituitary tumour",
}


def classify(img):
    if img is None:
        return {}, ""
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu()

    scores = {PRETTY.get(c, c): float(probs[i]) for i, c in enumerate(classes)}
    top = max(scores, key=scores.get)
    conf = scores[top]

    if conf < 0.60:
        note = f"Low confidence ({conf*100:.1f}%). The model is unsure — treat this output as unreliable."
    else:
        note = f"Top prediction: **{top}** at {conf*100:.1f}% confidence."
    return scores, note


with gr.Blocks(title="NeuroLens") as demo:
    gr.Markdown(
        "# NeuroLens\n"
        "Brain tumour classification from MRI. "
        "Paste an image with **Cmd+V**, drag one in, or click to browse.\n\n"
        "*Research and coursework demo only. Not a diagnostic tool.*"
    )

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", sources=["upload", "clipboard"], label="MRI slice", height=360)
            btn = gr.Button("Classify", variant="primary")
            clr = gr.Button("Clear")
        with gr.Column():
            out = gr.Label(num_top_classes=4, label="Prediction")
            msg = gr.Markdown()

    btn.click(classify, inputs=inp, outputs=[out, msg])
    inp.change(classify, inputs=inp, outputs=[out, msg])
    clr.click(lambda: (None, {}, ""), outputs=[inp, out, msg])

if __name__ == "__main__":
    demo.launch(inbrowser=True)