# Member C Handoff: From-Scratch ResNet18

This package contains Member C's trained model and experimental outputs.
The Jupyter Notebook is intentionally not included and can be added separately.

## Method

- Architecture: ResNet18
- Initialization: random (`weights=None`)
- Classes: 500
- Training images: 20,000
- Validation images: 5,000
- Test images: 5,000
- Input size: 224 × 224
- Optimizer: AdamW
- Initial learning rate: 0.0003
- Weight decay: 0.0001
- Batch size: 16
- Scheduler: CosineAnnealingLR
- Selected checkpoint: `results/models/scratch_aug_best.pth`
- Selected validation epoch: 28

## Augmentation Ablation

| Setting | Val Top-1 | Val Top-5 | Val Macro F1 | Best epoch |
|---|---:|---:|---:|---:|
| No augmentation | 21.54% | 43.42% | 21.42% | 16 |
| With augmentation | 31.94% | 56.74% | 30.06% | 28 |

Absolute improvements from augmentation:

- Top-1: +10.40 percentage points
- Top-5: +13.32 percentage points
- Macro F1: +8.65 percentage points

The no-augmentation model reached approximately 99.93% training accuracy,
indicating severe overfitting.

## Clean Test Results

- Top-1 accuracy: 32.10%
- Top-5 accuracy: 57.66%
- Macro precision: 30.45%
- Macro recall: 32.10%
- Macro F1: 29.73%

## Robustness Summary

Strongest severity results:

| Degradation | Severity | Top-1 | Macro F1 |
|---|---:|---:|---:|
| Gaussian noise | 0.20 | 17.28% | 15.88% |
| Gaussian blur | 7 | 15.46% | 13.47% |
| Brightness | 0.40 | 20.94% | 19.63% |
| Contrast | 0.40 | 18.16% | 17.99% |
| JPEG quality | 20 | 30.36% | 28.15% |

The model was most sensitive to strong blur and comparatively robust to JPEG compression.

## Package Contents

- `results/models/`: selected checkpoint
- `results/logs/`: training logs, summaries, and ablation table
- `results/figures/`: training and robustness plots
- `results/clean/`: clean per-image predictions and complete metrics
- `results/robustness/predictions/`: 15 degraded-test prediction files
- `results/robustness/metrics/`: 15 degraded-test metric files
- `reports/`: optional experiment summary

## Integration Notes

- Preserve the existing directory structure when merging into the team project.
- Do not overwrite newer files produced by other members.
- The checkpoint is for internal reproduction, analysis, and demonstration.
- Remove `.pth` files and large prediction files from the final course code ZIP if required by the submission size limit.
