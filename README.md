# GCMFF-Net: Gated Cross-Modal Feature Fusion for Moringa Leaf Disease Classification

A lightweight, explainable hybrid deep-learning framework for classifying **Moringa oleifera** leaf diseases from field-collected images. GCMFF-Net fuses a CNN image branch (with Squeeze-and-Excitation attention) and a 252-dimensional handcrafted texture/frequency feature branch through a **bidirectional gated cross-modal fusion** mechanism, preceded by HSV-based disease-region segmentation and followed by Grad-CAM / Grad-CAM++ / Score-CAM explainability.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Highlights

- **Primary dataset** — *MoringaLeafNet* (v4): 2,817 expert-annotated, field-collected images across 4 classes (Healthy, Yellow, Cercospora Leaf Spot, Bacterial Leaf Spot).
- **HSV segmentation** isolates disease-relevant tissue and generates semantic disease overlays before feature extraction.
- **Two complementary branches** — a 3-block SE-attention CNN and a 252-D handcrafted descriptor (statistical + GLCM + GLDM + FFT + DWT).
- **Bidirectional gated cross-modal fusion** lets each modality dynamically re-weight the other, per sample.
- **Lightweight** — ~0.3M trainable parameters (≈98% fewer than InceptionV3) yet stronger on the hardest class.
- **Built-in interpretability** — Grad-CAM, Grad-CAM++, Score-CAM (plus LIME/SHAP utilities).

---

## Results

Macro-averaged performance on the held-out test set (20% stratified split):

| Model | Accuracy | Precision | Recall | F1 | Bacterial Leaf Spot Acc. |
|---|---|---|---|---|---|
| Baseline CNN | 75.63% | 0.75 | 0.75 | 0.75 | 71.57% |
| TinyCNN | 74.29% | 0.76 | 0.73 | 0.73 | 68.80% |
| InceptionV3 (ImageNet) | 77.98% | 0.78 | 0.78 | 0.77 | 57.14% |
| NASNetMobile (ImageNet) | 70.61% | 0.80 | 0.76 | 0.76 | 50.87% |
| **GCMFF-Net (proposed)** | **90.06%** | **0.90** | **0.90** | **0.90** | **89.94%** |

The largest gain is on **Bacterial Leaf Spot** — the most visually ambiguous class — where GCMFF-Net improves accuracy by **+32.80 points** over InceptionV3 and **+39.07 points** over NASNetMobile.

> Per-class F1 stays in a tight 0.88–0.91 band, indicating balanced behaviour rather than majority-class bias.

---

## Repository structure

```
.
├── moringa-segmentation_ImageCentricCNN.ipynb   # Step 1: HSV segmentation + image-only CNN baseline
├── moringa-glcm.ipynb                           # Step 2: handcrafted features, feature engineering, CNN+XAI
├── moringa-glcm-votingensemble__2_.ipynb        # Step 3: full GCMFF-Net + ensembles + XAI (main notebook)
├── README.md
└── requirements.txt
```

| Notebook | What it does |
|---|---|
| `moringa-segmentation_ImageCentricCNN.ipynb` | Converts each augmented image to HSV, thresholds green/yellow/brown regions, refines with morphological ops, crops to the largest contour (256×256), and writes three outputs per image: `cropped_leaves/`, `masks/`, `overlays/`. Also trains a transfer-learning **image-only CNN** as a baseline. |
| `moringa-glcm.ipynb` | Extracts the **252-D handcrafted feature vector** from the segmented overlays, runs eight feature-engineering strategies (PCA, KPCA-RBF, sparse/stacked autoencoders, ANOVA, Chi², Random-Forest importance, full vector), evaluates each with an ANN, then trains a CNN on overlays and generates **XAI** maps. |
| `moringa-glcm-votingensemble__2_.ipynb` | **Main pipeline.** Builds the proposed **GCMFF-Net** (SE-attention CNN + handcrafted branch + bidirectional gated fusion + label smoothing), reports accuracy/loss curves, confusion matrix, classification report and per-class ROC curves. Also includes XGBoost/LightGBM/MLP stacking and Optuna-tuned baselines, plus the CNN+XAI block. |

---

## Method overview

```
Raw image
   │
   ├─► HSV thresholding (green / yellow / brown) ─► morphological refine ─► largest-contour crop
   │                                                                          │
   │                                                          ┌───────────────┴───────────────┐
   │                                                     cropped_leaves                    overlays (leaf + disease mask)
   │                                                                                            │
   │                                          ┌─────────────────────────────────────────────────┤
   │                                          ▼                                                  ▼
   │                              CNN image branch (3× Conv + SE)                  Handcrafted branch (252-D)
   │                              GAP → Dense(128)                      [Stat-14 | GLCM-56 | GLDM-56 | FFT-14 | DWT-112]
   │                                          │                                                  │
   │                                          └──────────── Bidirectional Gated Fusion ──────────┘
   │                                                          (each modality gates the other)
   │                                                                     │
   │                                              Dense(128) → Dense(64) → Softmax(4)
   │                                                                     │
   └─────────────────────────────► Grad-CAM / Grad-CAM++ / Score-CAM ◄──┘
```

### Handcrafted feature vector (252-D)

| Group | Dims | Description |
|---|---|---|
| Statistical texture | 14 | First-order histogram statistics (mean, std, energy, entropy, skewness, kurtosis, MAD, RMS, uniformity, …) |
| GLCM | 56 | Gray-Level Co-occurrence Matrix, 4 orientations × 14 measures |
| GLDM | 56 | Gray-Level Difference Matrix, 4 offsets × 14 measures |
| FFT | 14 | Statistics of the centred 2D magnitude spectrum |
| DWT | 112 | 2-level Daubechies-2 wavelet, 8 sub-bands × 14 statistics |

Features are standardized with a `StandardScaler` fitted on the training split only.

### GCMFF-Net architecture (proposed model)

- **CNN branch:** `Conv2D(32) → MaxPool → BN → SE` → `Conv2D(64) → MaxPool → BN → SE` → `Conv2D(128) → GAP → Dense(128, ReLU) → Dropout(0.3)`
- **SE block:** GAP → `Dense(C/16, ReLU)` → `Dense(C, sigmoid)` → channel re-scale (reduction ratio = 16)
- **Bidirectional gated fusion:**
  - image → `Dense(252, sigmoid)` gates the handcrafted vector
  - handcrafted → `Dense(128, sigmoid)` gates the image vector
  - concatenate the two gated vectors (→ 380-D)
- **Classifier head:** `Dense(128) → BN → Dropout(0.4) → Dense(64) → BN → Dropout(0.4) → Dense(4, softmax)`
- **Training:** Adam (lr = 1e-3), `ReduceLROnPlateau`, label smoothing λ = 0.03, batch size 32, 256×256 input.

---

## Dataset

**MoringaLeafNet (Version 4)** — a primary, field-collected, expert-annotated dataset of 2,817 original images captured with smartphones across two nurseries in Bangladesh under natural lighting and background variation.

- Classes: `Healthy`, `Yellow`, `Cercospora Leaf Spot`, `Bacterial Leaf Spot`
- Includes original, normalized, and augmented variants, plus per-image metadata (location, date, device, conditions).
- Publicly available via Mendeley Data / *Data in Brief* (see Citation).

> The notebooks expect the data under Kaggle-style paths (e.g. `/kaggle/input/datasets/.../moringa-augmented` and `.../moringa-segmented`). Update the `INPUT_DIR` / `OVERLAY_DIR` / `FEATURES_CSV` variables at the top of each notebook to match your local or cloud layout.

---

## Getting started

### Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
opencv-python-headless
scikit-image
scikit-learn
scipy
PyWavelets
tensorflow>=2.10
matplotlib
seaborn
tqdm
xgboost
lightgbm
optuna
tf-keras-vis
shap
lime
```

A GPU is recommended (the experiments were run on an NVIDIA Tesla T4, 16 GB, with mixed-precision training).

### Run order

1. **Segment** — run `moringa-segmentation_ImageCentricCNN.ipynb` to produce `cropped_leaves/`, `masks/`, and `overlays/`.
2. **Extract features** — run `moringa-glcm.ipynb` (Cells 1–3) to build `features_252_from_overlays.csv`; the later cells run feature-engineering comparisons and the CNN+XAI block.
3. **Train GCMFF-Net** — run the main model cell in `moringa-glcm-votingensemble__2_.ipynb` (the "NOVEL LIGHTWEIGHT HYBRID MODEL" cell). It outputs the trained model, training curves, confusion matrix, classification report, and per-class ROC curves.

Update the path variables in each notebook's CONFIG block before running.

---

## Explainability

The XAI block loads the trained model and generates, per test image:

- **Grad-CAM** — gradient-based class activation maps
- **Grad-CAM++** — sharper, multi-instance lesion localization
- **Score-CAM** — gradient-free, spatially smooth maps
- **LIME / SHAP** — utility integrations for local attribution

The maps confirm the model attends to biologically relevant regions (necrotic lesion borders, chlorotic patches) rather than background.

---

## Citation

If you use this code or the dataset, please cite:

```bibtex
@article{preanto2025moringaleafnet,
  title   = {MoringaLeafNet: A Multi-Class Leaf Disease Dataset for Precision
             Agriculture and Deep Learning Research},
  author  = {Preanto, Sabit Ahmed and Paul, Tapon and Khan, A. and Bijoy, Md. Hasan Imam},
  journal = {Data in Brief},
  pages   = {112174},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.112174}
}
```

> A citation entry for the GCMFF-Net paper will be added here once published.

---

## License

Released under the MIT License. The MoringaLeafNet dataset is distributed under its own terms on Mendeley Data — please review them before redistribution.

## Acknowledgements

Thanks to the agricultural experts who annotated and validated the dataset, and to the Kaggle platform for compute resources.
