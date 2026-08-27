# Classical Computer Vision vs. Deep Learning for Object Detection & Counting on Raspberry Pi

A two-track comparative study evaluating whether a deep learning classifier outperforms a classical computer vision pipeline for real-world object detection and counting on low-cost hardware.

This project extends and critically re-evaluates a published research baseline (Abdulhamid et al., 2020) — rebuilding it from scratch, correcting a methodological flaw in an earlier implementation, and producing a validated, evidence-backed comparison.

**Master's Research Module in AI (14060, SoSe 26) — Department of Graphical Systems, BTU Cottbus-Senftenberg**
Emmad Ismail · Nico Pschipsch · Douglas W. Cunningham

---

## Key finding

The deep learning model's advantage is **not uniform** across conditions, as originally hypothesised. It is concentrated in a specific, well-defined failure mode of the classical approach:

| Condition                 | Classical (OpenCV) | Deep learning (ResNet18) | Advantage            |
| ------------------------- | ------------------ | ------------------------ | -------------------- |
| 3D objects, oblique angle | 34.5%              | 66.4%                    | **+31.9 pp**         |
| Flat 2D shapes            | 56.0%              | 54.9%                    | ≈ 0 (near-identical) |

When the classical method's core geometric assumptions hold — flat 2D shapes — the two approaches perform almost identically. The advantage appears specifically where those assumptions break down: a cube viewed obliquely presents 5–6+ outline vertices rather than 4, so vertex-counting cannot succeed regardless of image quality. This is a structural limitation, not a tuning problem.

That mechanistic explanation is the real result. "Deep learning wins" would have been the easy conclusion; the data supports something narrower and more useful.

## Overview

Two pipelines share the same detection front-end, differing only in how they classify detected shapes. This isolates a controlled, classification-only comparison.

- **Track A — Classical pipeline (OpenCV).** Grayscale conversion → adaptive (Otsu) thresholding for lighting-invariant table segmentation → convex-hull correction for occlusion artifacts → Gaussian blur → Canny edge detection → boundary-band removal → morphological closing → contour analysis → geometric shape classification by vertex count (3 = triangle, 4 = rectangle, otherwise "other").
- **Track B — Deep learning pipeline (PyTorch).** Transfer learning with a pretrained ResNet18 (ImageNet weights), fine-tuning the classification head on a geometric shapes dataset. **98.6% validation accuracy.** The trained classifier is dropped into Track A's detection pipeline as a replacement for the classical shape-classification stage.

The shared detection stage is deliberate. In an earlier version of this project the two tracks were not properly separated, so both always produced identical object counts and the comparison was structurally meaningless. Sharing detection explicitly and comparing only classification is the fix.

## Research questions

Following Abdulhamid et al., the project asked:

1. Will the classical pipeline replicate the reported 90.206% counting accuracy under clean conditions, and degrade under challenging ones?
2. Will a ResNet model outperform the classical pipeline specifically under those challenging conditions?

Both were confirmed — the second more precisely than predicted.

## Full results

| Metric                                       | Value         | Notes                                                              |
| -------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| ResNet — validation accuracy                 | 98.6%         | Transfer learning cleanly separates the 3 shape classes            |
| Track A — real-world counting efficiency     | 61.28%        | Original paper reported 90.206% on clean, cropped images           |
| Track A vs Track B — classification accuracy | 48.1% / 59.1% | ResNet ahead overall, but the gap is not uniform (see Key finding) |

The drop from 90.206% to 61.28% is expected and informative: the original paper evaluated on clean, cropped, well-separated images, while this test set uses real, cluttered room photographs.

## Methodology

- **Ground truth:** true object count and shape recorded manually per photo by the experimenter.
- **Counting metric:** `efficiency = 1 − |detected − true| / true`, per photo. Same formula as Abdulhamid et al., enabling direct comparison to their baseline.
- **Classification metric:** because both tracks share detection, accuracy is scored via label-overlap against true shapes rather than per-detection matching.
- **Fair-comparison framework:** evaluation accounts for each model's output vocabulary, so neither track is penalised for label sets it was never designed to produce.
- **Corrections over the earlier implementation:** identified and fixed a train/test data-leakage issue and a shared-detection confound present in a prior version.

## Data

### Test set — included

38 photos captured with a Raspberry Pi camera on a fixed tripod setup, showing physical 2D and 3D objects (cubes, pyramids, spheres, cylinders, flat cutouts).

- 14 clean (well-spaced) / 24 close-spacing (tightly spaced or touching)
- 14 containing 3D solid objects / 24 containing flat 2D cutouts
- Manually annotated with ground-truth counts and shape labels

### Training set — not included

Kaggle [geometric-shapes-dataset by dineshpiyasamara](https://www.kaggle.com/datasets/dineshpiyasamara/geometric-shapes-dataset) — 10,000 images per class (Circle, Square, Triangle). Not redistributed here; download it directly from Kaggle.

Used for training: 1,500 training / 300 validation images per class.

Extract into `dataset/` so the path matches `DATASET_DIR` in `src/train_resnet.py`:

```
dataset/geometric shapes dataset/
├── Circle/
├── Square/
└── Triangle/
```

**Note on terminology:** the Kaggle "Square" class contains only perfect squares. Random aspect-ratio augmentation was added during training to help the model generalise to real, non-square rectangles.

## Repository structure

```
.
├── src/
│   ├── train_resnet.py               # Train / fine-tune the ResNet18 classifier
│   ├── resnet_pipeline.py            # Track B: shared detection + ResNet classification
│   ├── opencv_pipeline.py            # Track A: classical CV detection + classification
│   ├── compare_pipelines.py          # Runs both tracks, produces comparison results
│   ├── run_baseline_batch.py         # Batch evaluation over the test set
│   ├── evaluate_saved_model.py       # Evaluate a saved .pth model
│   ├── label_ground_truth.py         # Tool: annotate object counts
│   ├── label_true_shapes.py          # Tool: annotate shape labels
│   ├── inspect_dataset.py            # Dataset inspection utility
│   ├── preview_crop.py               # Crop preview utility
│   └── test_single_*.py              # Single-image test scripts
├── models/
│   └── resnet_shape_classifier.pth   # Trained ResNet18 weights (~43 MB)
├── outputs/
│   ├── baseline_results.csv
│   └── comparison_results.csv
├── test_photos/
│   └── click_photos/                 # 38-image annotated test set
│       ├── click_photos.py           # Parallel capture from Pi cameras over SSH
│       ├── hosts.ini.example         # Template for Pi inventory
│       ├── ground_truth.csv
│       └── ground_truth_with_shapes.csv
├── dataset/                          # NOT included — see Data
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/shape-detection-cv-vs-dl.git
cd shape-detection-cv-vs-dl

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Usage

The trained model weights are included, so the evaluation reproduces without retraining:

```bash
# Reproduce the comparison over the test set
python src/compare_pipelines.py

# Evaluate the saved model on its own
python src/evaluate_saved_model.py

# Run Track A across the whole test set
python src/run_baseline_batch.py

# Inspect a single image
python src/test_single_image.py
python src/test_single_image_resnet.py
```

To retrain from scratch (requires the Kaggle dataset):

```bash
python src/train_resnet.py
```

### Capturing new photos from a Raspberry Pi

`click_photos.py` captures from one or more Pi cameras in parallel over SSH. Copy `hosts.ini.example` to `hosts.ini` and fill in your Pi's address and user. It tries key-based authentication first; to use password authentication as a fallback, set `PI_PASSWORD` in your environment rather than hardcoding it.

```bash
export PI_PASSWORD="your-pi-password"     # macOS / Linux
$env:PI_PASSWORD = "your-pi-password"     # Windows PowerShell
```

## Known limitations

1. **Vertex-counting vs. 3D objects at an angle.** Structural, not tunable — confirmed by the 34.5% vs 56.0% split for Track A on 3D vs flat objects.
2. **Horizon-line ambiguity.** Where an object's base touches the table/backdrop boundary, no pixel-level rule reliably separates touching objects from one merged shape.
3. **Fixed-camera assumptions.** The pipeline is calibrated for a single tripod position and does not generalise to arbitrary viewpoints.
4. **No genuine shadow condition.** The original paper tested shadow effects; this dataset does not contain a naturally-occurring shadow condition.

## Future work

- Give Track B independent detection rather than sharing Track A's, enabling a full end-to-end comparison
- Test genuine shadow conditions
- Generalise beyond a fixed camera position

## Tech stack

Python · OpenCV · PyTorch · ResNet18 (transfer learning) · NumPy · Raspberry Pi camera · Ansible-style SSH capture tooling

## Skills demonstrated

Computer vision pipeline design, transfer learning, systematic debugging and root-cause diagnosis, experimental design, statistical evaluation methodology, data annotation tooling, and technical/scientific communication.

## References

1. Abdulhamid, M., Odondi, O., & Al-Rawi, M. (2020). _Computer vision based on Raspberry Pi system._ Applied Computer Science, 16(4), 85–102.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). _Deep residual learning for image recognition._ Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778.
3. Piyasamara, D. _Geometric Shapes Dataset._ Kaggle. https://www.kaggle.com/datasets/dineshpiyasamara/geometric-shapes-dataset

## License

See [LICENSE](LICENSE).

## Contact

emmad@b-tu.de
