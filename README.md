# NeuroLens

Brain tumor classification from MRI using transfer learning, with an honest look at what the model actually learned.

A PyTorch model built on EfficientNet-B0 that sorts brain MRI slices into four classes — glioma, meningioma, pituitary tumor, and no tumor — and serves predictions through a browser interface.

**This is a university coursework project (AI 1). It is not a medical device and must not be used for diagnosis.**

---

## Results

| Split | Images | Accuracy | Loss |
|---|---|---|---|
| Validation | 840 | 99.05% | 0.033 |
| Test | 1,600 | 95.06% | 0.337 |

Trained for 4 head epochs and 20 fine-tuning epochs; the best checkpoint came from fine-tuning epoch 19.

The 4-point gap between validation and test is the most interesting result here, not the accuracy itself.

The validation set is a random 15% slice of the `Training/` folder, so it shares patients, slices, and scanner sources with the data the model trained on. The test set is a separate collection. That overlap inflates the validation score, which is why **95.06% is the number worth reporting**. Many published results on this dataset quote a figure closer to the validation number without making the distinction.

The loss tells a sharper story than the accuracy does. Test loss is ten times validation loss (0.337 vs 0.033) while accuracy differs by only four points, which means the test images the model gets wrong, it gets wrong with high confidence. Those cases are the target of the planned error analysis.

Validation accuracy plateaued around epoch 6 of fine-tuning (0.986) and gained only 0.4 points over the following fourteen epochs, so roughly half the training budget could be cut with little cost.

---

## Quick start

```bash
git clone https://github.com/<your-username>/neurolens.git
cd neurolens

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset and place `Training/` and `Testing/` in the project root:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -q
unzip -q brain-tumor-mri-dataset.zip
```

Expected layout:

```
neurolens/
├── Training/
│   ├── glioma/  meningioma/  notumor/  pituitary/
├── Testing/
│   ├── glioma/  meningioma/  notumor/  pituitary/
├── train.py          # training, both phases
├── predict.py        # command-line inference
├── app.py            # Gradio web interface
└── best1.pt          # trained checkpoint (weights + class names)
```

Then train and launch:

```bash
python train.py       # ~45 min on Apple Silicon (MPS), longer on CPU
python app.py         # opens the web interface at http://127.0.0.1:7860
```

---

## Usage

**Train.** `python train.py` runs both phases and writes `best1.pt` whenever validation accuracy improves. The checkpoint stores the class names alongside the weights so inference can never mismatch the label order.

**Predict from the command line.**

```bash
python predict.py Testing/glioma/Te-gl_0010.jpg
```

```
Te-gl_0010.jpg
  glioma         94.2%
  meningioma      3.1%
  pituitary       1.9%
  notumor         0.8%
```

(Replace this with real output from your own run.)

**Web interface.** `python app.py` opens a Gradio page. Paste an image from the clipboard, drag one in, or browse for a file. It shows all four probabilities and warns when top confidence falls below 60%.

---

## How it works

**Preprocessing.** The dataset mixes image sizes and color modes, so everything is converted to 3-channel grayscale, resized to 224×224, and normalized with ImageNet statistics. Training images also get random rotation (±15°), horizontal flips, and small shifts and zooms. Validation and test images get no augmentation, so evaluation stays deterministic.

**Model.** EfficientNet-B0 pretrained on ImageNet, with the classifier replaced by a `Linear(1280, 4)` layer.

**Two-phase training.**

1. *Head* — the backbone is frozen and only the new output layer trains at `lr=1e-3`. This stops gradients from the randomly initialized head from disturbing the pretrained filters.
2. *Fine-tune* — the last blocks unfreeze and train at `lr=1e-4`, letting the late features adapt to MRI texture.

**Evaluation.** The `Testing/` folder is used exactly once, after loading the best checkpoint. Epoch selection is done on validation only.

---

## Math

The five equations behind the code:

| Step | Equation |
|---|---|
| Normalization | x′ = (x − μ) / σ |
| Convolution | y(i,j) = Σₘ Σₙ x(i+m, j+n) · w(m,n) + b |
| Softmax | pₖ = exp(zₖ) / Σⱼ exp(zⱼ) |
| Cross-entropy loss | L = −ln(pₜ) |
| Gradient descent | w ← w − η · ∂L/∂w |

The training code uses Adam, which adapts the step size per weight, but the update rule above is the idea underneath it.

---

## Limitations

- **No patient-level split.** The dataset provides no patient IDs, so adjacent slices from one patient may land in both training and validation. This is the likeliest cause of the validation/test gap.
- **No "unknown" class.** The model always returns one of four labels. Feed it a photo of a cat and it will confidently call it a tumor type. The 60% confidence warning in the app is a partial guard, not a fix.
- **Possible scanner shortcuts.** The dataset is assembled from several sources, and the classes are not evenly drawn from them. The model may be picking up on acquisition style rather than pathology. Grad-CAM is the planned check.
- **Not clinically validated.** No external validation set, no radiologist review, no regulatory assessment.

---

## Roadmap

- [ ] Grad-CAM heatmaps to show which regions drive each prediction
- [ ] Confusion matrix, per-class precision/recall/F1, ROC curves
- [ ] k-NN baseline on the extracted 1280-d features, for comparison against the neural classifier
- [ ] Full-network fine-tuning with a cosine LR schedule and label smoothing
- [ ] Error analysis of misclassified images

---

## Requirements

```
torch
torchvision
gradio
pillow
matplotlib
```

Tested on macOS with Apple Silicon (PyTorch MPS backend). CUDA and CPU also work — the device is selected automatically.

---

## Acknowledgments

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle).

Key references:

1. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019.
2. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," ICLR, 2015.
3. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV, 2017.
4. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS, 2019.

## License

MIT
