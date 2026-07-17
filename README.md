# COMP9517 Group Project

This repository contains the shared code, dataset split files, evaluation interface, and robustness testing utilities for the COMP9517 group project.

The project studies fine-grained species classification using a subset of the iNaturalist-2021 Mini dataset. All members should use the same dataset split and output format so that the traditional model, the from-scratch deep model, and the pretrained deep model can be compared fairly.

---

## 1. Project Structure

```text
COMP9517_Project/
├── data/
│   ├── raw/                         # Raw iNaturalist images and metadata, not included in final code submission
│   └── splits/                      # Fixed 500-class train/val/test split files
├── src/                             # Shared Python source code
│   ├── prepare_dataset.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── degradations.py
│   ├── evaluate.py
│   ├── plot_confusion_matrix.py
│   ├── summarize_robustness.py
│   └── plot_robustness_curves.py
├── results/
│   ├── clean/                       # Clean test predictions and metrics
│   ├── robustness/                  # Robustness predictions, metrics, templates, and summary CSV
│   └── figures/                     # Generated figures
├── reports/
│   └── A_dataset_interface_notes.md # Detailed interface notes for group members
├── notebooks/                       # Optional exploration notebooks
├── PROJECT_TEAM_GUIDE_CN.md          # Chinese internal team guide
├── README.md
└── requirements.txt
```

---

## 2. Dataset Placement

The raw iNaturalist images are not included in the code package because the dataset is large. Each member should place the raw data under:

```text
data/raw/
├── train_mini/
├── val/
├── train_mini.json
└── val.json
```

The expected full paths are:

```text
COMP9517_Project/data/raw/train_mini/
COMP9517_Project/data/raw/val/
COMP9517_Project/data/raw/train_mini.json
COMP9517_Project/data/raw/val.json
```

The fixed split files are already provided under:

```text
data/splits/
```

Important: `data/splits/*.csv` only stores image paths and labels. The actual images are still loaded from `data/raw/`.

---

## 3. Fixed Dataset Split

We use a fixed 500-class subset of the iNaturalist-2021 Mini dataset.

Current setting:

- Random seed: `9517`
- Number of selected species/classes: `500`
- Training images: `40` images per class, `20,000` images in total
- Validation images: `10` images per class, `5,000` images in total
- Test images: `10` images per class, `5,000` images in total
- Training and validation images are sampled from `train_mini`
- Test images are sampled from the official iNaturalist validation split

Generated split files:

```text
data/splits/selected_classes.txt
data/splits/class_to_idx.json
data/splits/idx_to_class.json
data/splits/split_summary.json
data/splits/train_paths.csv
data/splits/val_paths.csv
data/splits/test_paths.csv
```

Each CSV file contains:

```text
image_path,image_id,category_id,label,class_idx,split
```

All models must use `class_idx` as the label. Do not use the original iNaturalist `category_id`, because it is not a continuous 0-499 label space.

---

## 4. Environment Setup

Install dependencies using:

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch on Windows, the following command can be used if needed:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Members using GPU should install the PyTorch version that matches their CUDA environment.

---

## 5. Dataloader Interface for Deep Learning Models

Members working on the from-scratch and pretrained deep learning models can use the shared dataloader:

```python
from src.dataset import make_loader

train_loader = make_loader(
    csv_path="data/splits/train_paths.csv",
    split="train",
    batch_size=32,
    image_size=224,
    num_workers=0,
    augment=True,
)

val_loader = make_loader(
    csv_path="data/splits/val_paths.csv",
    split="val",
    batch_size=32,
    image_size=224,
    num_workers=0,
    augment=False,
)

test_loader = make_loader(
    csv_path="data/splits/test_paths.csv",
    split="test",
    batch_size=32,
    image_size=224,
    num_workers=0,
    augment=False,
)
```

The dataloader returns:

```python
images, labels = next(iter(train_loader))
```

Expected shapes:

```text
images: [batch_size, 3, 224, 224]
labels: [batch_size]
```

On Windows, start with `num_workers=0`. After confirming that the pipeline works, it can be increased to 2 or 4.

---

## 6. Interface for the Traditional Model

The traditional method can directly read the split CSV files:

```python
import pandas as pd
import cv2

train_df = pd.read_csv("data/splits/train_paths.csv")

for _, row in train_df.iterrows():
    img_path = row["image_path"]
    label = row["class_idx"]

    img = cv2.imread(img_path)
    # Extract SIFT / BoVW features here
```

Use `image_path` to read the image and `class_idx` as the label.

---

## 7. Required Clean Prediction Outputs

After training, each model owner must provide a clean test prediction CSV under `results/clean/`.

Required files:

```text
results/clean/traditional_predictions.csv
results/clean/scratch_predictions.csv
results/clean/pretrained_predictions.csv
```

The required CSV format is:

```text
image_path,true_idx,pred_idx,top5_idx
```

Example:

```text
data/raw/val/example.jpg,12,18,18 12 35 7 209
```

Column explanation:

- `image_path`: path to the test image
- `true_idx`: ground-truth class index
- `pred_idx`: top-1 predicted class index
- `top5_idx`: top-5 predicted class indices, separated by spaces

Important: `true_idx`, `pred_idx`, and `top5_idx` must all use the remapped 0-499 `class_idx` labels.

---

## 8. Unified Clean Evaluation

Member A will evaluate all clean prediction files using the same script.

Example commands:

```bash
python src/evaluate.py --pred-csv results/clean/traditional_predictions.csv --out-json results/clean/traditional_metrics.json --num-classes 500
python src/evaluate.py --pred-csv results/clean/scratch_predictions.csv --out-json results/clean/scratch_metrics.json --num-classes 500
python src/evaluate.py --pred-csv results/clean/pretrained_predictions.csv --out-json results/clean/pretrained_metrics.json --num-classes 500
```

Expected clean metrics files:

```text
results/clean/traditional_metrics.json
results/clean/scratch_metrics.json
results/clean/pretrained_metrics.json
```

The evaluation script reports:

- Top-1 accuracy
- Top-5 accuracy
- Macro precision
- Macro recall
- Macro F1
- Confusion matrix

---

## 9. Robustness Testing Workflow

Advanced Direction 2 studies robustness to test-time image degradation.

Supported degradation types in `src/degradations.py`:

- Gaussian noise
- Gaussian blur
- Brightness change
- Contrast change
- JPEG compression

Rules:

- Apply degradation only to test images
- Do not modify the training data
- Do not retrain the model
- Evaluate the same trained model on clean and degraded test sets
- Report top-1 accuracy and macro-F1 under each degradation and severity level

Recommended severity levels:

```text
noise:      0.05, 0.10, 0.20
blur:       3, 5, 7
brightness: 0.8, 0.6, 0.4
contrast:   0.8, 0.6, 0.4
jpeg:       70, 40, 20
```

The robustness template is provided at:

```text
results/robustness/robustness_result_template.csv
```

Members should use this template to check the expected degradation names, severity values, prediction file names, and metrics file names.

---

## 10. Required Robustness Outputs

For each degraded test condition, model owners should provide prediction CSV files under:

```text
results/robustness/predictions/
```

Suggested naming format:

```text
results/robustness/predictions/{model}_{degradation}_{severity}_predictions.csv
```

Examples:

```text
results/robustness/predictions/traditional_noise_0.05_predictions.csv
results/robustness/predictions/scratch_blur_5_predictions.csv
results/robustness/predictions/pretrained_jpeg_40_predictions.csv
```

Each degraded prediction CSV must use the same format as clean predictions:

```text
image_path,true_idx,pred_idx,top5_idx
```

Member A will run unified evaluation and save degraded metrics under:

```text
results/robustness/metrics/
```

Suggested naming format:

```text
results/robustness/metrics/{model}_{degradation}_{severity}_metrics.json
```

Examples:

```text
results/robustness/metrics/traditional_noise_0.05_metrics.json
results/robustness/metrics/scratch_blur_5_metrics.json
results/robustness/metrics/pretrained_jpeg_40_metrics.json
```

---

## 11. Robustness Summary and Plotting

After clean and degraded metrics are available, Member A can summarize robustness results using:

```bash
python src/summarize_robustness.py \
    --clean-dir results/clean \
    --metrics-dir results/robustness/metrics \
    --out-csv results/robustness/combined_robustness_results.csv
```

This generates:

```text
results/robustness/combined_robustness_results.csv
```

The combined CSV contains rows such as:

```text
model,degradation,severity,top1_accuracy,macro_f1
traditional,clean,0,0.120,0.095
traditional,noise,0.05,0.098,0.081
scratch,clean,0,0.180,0.150
pretrained,clean,0,0.420,0.390
```

Then generate robustness curves using:

```bash
python src/plot_robustness_curves.py \
    --input-csv results/robustness/combined_robustness_results.csv \
    --out-dir results/figures
```

This generates figures such as:

```text
results/figures/robustness_noise_top1_accuracy.png
results/figures/robustness_noise_macro_f1.png
results/figures/robustness_blur_top1_accuracy.png
results/figures/robustness_jpeg_macro_f1.png
```

These figures can be used in the final report and presentation video.

---

## 12. Member Responsibilities and Required Outputs

### Member A: Dataset, Evaluation, and Robustness Framework

Main responsibilities:

- Maintain dataset split files
- Provide dataloader and evaluation interface
- Collect clean prediction CSVs from B/C/D
- Run unified clean evaluation
- Collect or evaluate degraded prediction CSVs
- Generate `combined_robustness_results.csv`
- Generate robustness curves

Key files maintained by A:

```text
data/splits/
src/dataset.py
src/metrics.py
src/evaluate.py
src/degradations.py
src/summarize_robustness.py
src/plot_robustness_curves.py
results/robustness/robustness_result_template.csv
reports/A_dataset_interface_notes.md
```

### Member B: Traditional Model

Main output files:

```text
results/clean/traditional_predictions.csv
results/robustness/predictions/traditional_{degradation}_{severity}_predictions.csv
```

### Member C: From-scratch Deep Model

Main output files:

```text
results/clean/scratch_predictions.csv
results/robustness/predictions/scratch_{degradation}_{severity}_predictions.csv
```

### Member D: Pretrained / Transfer Learning Model

Main output files:

```text
results/clean/pretrained_predictions.csv
results/robustness/predictions/pretrained_{degradation}_{severity}_predictions.csv
```

Member D should also provide the best pretrained checkpoint and model architecture details for Grad-CAM.

### Member E: Error Analysis and Grad-CAM

Main responsibilities:

- Use prediction CSVs and metrics from B/C/D
- Analyse confusion matrix and hardest confused species pairs
- Prepare Grad-CAM visualisations using the pretrained model
- Discuss robustness results and model failure cases

---

## 13. Files Not Included in Final Code Submission

The final code submission should not include large files such as:

```text
data/raw/
*.jpg
*.jpeg
*.png
*.tar.gz
*.zip
*.pt
*.pth
*.ckpt
__pycache__/
.ipynb_checkpoints/
```

The raw dataset and large model checkpoints should be shared separately if needed. The final code ZIP should include code, split CSVs, small result CSV/JSON files, and documentation only.
