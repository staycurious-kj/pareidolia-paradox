# The Pareidolia Paradox — ResNet34 v3.3

## Overview

This repository contains the original v3.3 ResNet34 submission pipeline for
The Pareidolia Paradox image-classification challenge.

## Model

- Architecture: ResNet34
- ImageNet-pretrained backbone
- Binary classification
- Loss: BCEWithLogitsLoss
- Random seed: 42
- Cross-validation: 5-fold StratifiedKFold
- Epochs: 8
- Batch size: 64
- Learning rate: 3e-4
- Weight decay: 1e-4
- Training augmentation: horizontal flip
- Inference augmentation: horizontal-flip TTA
- Final decision threshold: 0.555

## Model weights

The trained v3.3 model checkpoints are available here:

[Download model weights from Google Drive](https://drive.google.com/drive/folders/1xRGTXIB_tavayeql_3jPIih3ZzAAsqiR?usp=sharing)

The folder contains the five seed-42 fold checkpoints:

- `seed_42_fold_1.pt`
- `seed_42_fold_2.pt`
- `seed_42_fold_3.pt`
- `seed_42_fold_4.pt`
- `seed_42_fold_5.pt`

## Azimuth normalization

The dataset provides `sun_azimuth_angle`, which describes the sun position
associated with each image.

The preprocessing pipeline normalizes the image using:

1. Grayscale conversion
2. Reflective padding
3. Rotation by `-sun_azimuth_angle`
4. Crop back to the original image framing
5. Resize to 224x224
6. Replication to three channels
7. ImageNet normalization

The same preprocessing is used during training and inference.

## Validation results

Original v3.3 fold balanced accuracy:

| Fold | Balanced Accuracy |
|------|------------------:|
| 1 | 0.588694 |
| 2 | 0.642244 |
| 3 | 0.653382 |
| 4 | 0.572414 |
| 5 | 0.603114 |

Original overall OOF balanced accuracy:

`0.627902`

Optimized classification threshold:

`0.555`

## Checkpoints

The final model ensemble consists of five seed-42 fold checkpoints:

```text
seed_42_fold_1.pt
seed_42_fold_2.pt
seed_42_fold_3.pt
seed_42_fold_4.pt
seed_42_fold_5.pt