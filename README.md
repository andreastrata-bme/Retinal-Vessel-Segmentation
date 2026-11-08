# Retinal-Vessel-Segmentation

End-to-end pipeline for automated retinal vessel segmentation from fundus images,
developed as a group project for the Medical Image Processing course (MSc Biomedical
Engineering, Politecnico di Torino, A.Y. 2025-2026).

## Overview

The pipeline takes RGB fundus images as input and produces binary vessel segmentation
masks using a UNet++ architecture with a ResNet34 backbone. The approach emphasizes
topological correctness of the vascular structure alongside standard overlap metrics.

## Pipeline

1. **Preprocessing** — per-channel Box Filtering (3×3) → CLAHE (clip limit 2.0) →
   Gamma Correction (γ = 1.3) → normalization with training set statistics
2. **Inference** — UNet++ (ResNet34 backbone), input resized to 768×768,
   output upsampled to 1024×1024
3. **Post-processing** — probability map binarization (threshold 0.70) →
   skeletonization-based gap repair (Dmax = 15 px, K = 3) → small object removal
   (Amin = 150 px)

## Architecture Selection

Four state-of-the-art architectures were compared on the Validation Set:

| Architecture | clDice (%) | DSC (%) |
|---|---|---|
| **UNet++** | **86.88 ± 11.43** | **87.34 ± 12.25** |
| U-Net | 85.90 ± 17.84 | 86.17 ± 13.24 |
| MA-Net | 83.91 ± 9.64 | 82.54 ± 9.30 |
| DeepLabV3+ | 81.70 ± 12.20 | 79.48 ± 11.57 |

UNet++ was selected as the best-performing architecture.

## Results

Performance on the internal Test Set (40 images):

| Post-processing | clDice (%) | DSC (%) |
|---|---|---|
| No | 85.75 ± 9.34 | 80.24 ± 10.92 |
| **Yes** | **86.07 ± 8.60** | **81.63 ± 6.18** |

Evaluation metrics: Dice Similarity Coefficient (DSC) and topology-aware
Centerline Dice (clDice).

## Dataset

600 color fundus images (1024×1024 px, RGB) with expert-annotated binary vessel masks.
Final split: Training Set 470 images, Validation Set 90 images, internal Test Set 40 images.
Dataset not included in this repository.

## Requirements

```bash
pip install segmentation-models-pytorch torch torchvision opencv-python \
scikit-image scipy tqdm matplotlib Pillow numpy
```

## Usage

The notebook is designed to run on **Google Colab** with files on Google Drive.
Update the paths in the configuration section before running:

```python
CHECKPOINT_PATH = "path/to/BEST_MODEL_UnetPlusPlus.pt"
test_images_dir  = "path/to/test/images"
output_masks_dir = "path/to/output"
```

Then run all cells sequentially.

## Authors

Group BC9 — Politecnico di Torino
U. Perassi, C. Razeto, A. Riglietta, A. Strata, R. Tridente
