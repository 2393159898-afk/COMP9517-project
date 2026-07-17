# COMP9517 项目团队协作指南（当前版本）

这个文件是组内中文说明文档，用来解释当前项目文件夹里每个部分的作用，以及 A/B/C/D/E 每位同学后续应该在哪里写代码、读取什么数据、输出什么文件。

最终提交给老师看的说明以英文 `README.md` 为准；这个文件主要方便组内沟通。

---

## 1. 当前项目整体结构

```text
COMP9517_Project/
├── data/
│   ├── raw/                         # 原始 iNaturalist 图片和 json，不进入最终代码提交包
│   └── splits/                      # A 已经生成好的 500-class 固定划分
├── src/                             # 共享代码
│   ├── prepare_dataset.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── degradations.py
│   ├── evaluate.py
│   ├── plot_confusion_matrix.py
│   ├── summarize_robustness.py
│   └── plot_robustness_curves.py
├── results/
│   ├── clean/                       # clean test set 上的 prediction 和 metrics
│   ├── robustness/                  # robustness prediction、metrics、template 和汇总结果
│   └── figures/                     # 生成的结果图
├── reports/
│   └── A_dataset_interface_notes.md # A 给组员的英文接口说明
├── notebooks/                       # 可选探索 notebook
├── PROJECT_TEAM_GUIDE_CN.md          # 当前中文团队指南
├── README.md                         # 英文项目说明，给组员和最终 marker 看
└── requirements.txt
```

---

## 2. 我们目前已经完成的事情

A 目前已经完成了项目基础设施部分：

```text
1. 正确放置原始 iNaturalist 数据
2. 生成正式 500-class train/val/test split
3. 检查 split CSV 中的图片路径是否可用
4. 测试 PyTorch dataloader
5. 测试 metrics / evaluate 代码
6. 测试 image degradation functions
7. 准备英文接口说明文件 A_dataset_interface_notes.md
8. 准备 robustness result template
9. 准备 robustness metrics 汇总脚本
10. 准备 robustness 曲线绘图脚本
```

这些内容不是最终模型结果，但它们是所有模型公平比较的基础。

---

## 3. 数据文件夹说明

### `data/raw/`

这里放原始数据：

```text
data/raw/
├── train_mini/
├── val/
├── train_mini.json
└── val.json
```

这个文件夹用于本地训练和测试，但不要放进最终代码提交包，因为体积太大。

### `data/splits/`

这里放 A 已经生成好的 500-class 固定划分：

```text
data/splits/
├── selected_classes.txt
├── class_to_idx.json
├── idx_to_class.json
├── split_summary.json
├── train_paths.csv
├── val_paths.csv
└── test_paths.csv
```

所有模型都必须使用这里的 split。B/C/D 都要使用 `class_idx` 作为标签，不要使用原始 `category_id`。

---

## 4. `src/` 代码文件说明

### `src/prepare_dataset.py`

A 用来生成 500-class split。现在正式 split 已经生成，除非要重新划分，否则不用再运行。

### `src/dataset.py`

C/D 用来创建 PyTorch dataloader。

### `src/metrics.py`

A 用来统一计算 top-1 accuracy、top-5 accuracy、macro precision、macro recall、macro F1 和 confusion matrix。

### `src/evaluate.py`

读取 prediction CSV，输出 metrics JSON。后面 A 会用它统一评估 B/C/D 的模型。

### `src/degradations.py`

实现 robustness test 用到的图像退化，包括 noise、blur、brightness、contrast、JPEG compression。

### `src/plot_confusion_matrix.py`

后面用于画 confusion matrix，主要给 A/E 做 error analysis 和 report figures。

### `src/summarize_robustness.py`

新增脚本。作用是读取 clean metrics 和 robustness metrics，然后汇总成：

```text
results/robustness/combined_robustness_results.csv
```

运行方式：

```bash
python src/summarize_robustness.py --clean-dir results/clean --metrics-dir results/robustness/metrics --out-csv results/robustness/combined_robustness_results.csv
```

### `src/plot_robustness_curves.py`

新增脚本。作用是读取：

```text
results/robustness/combined_robustness_results.csv
```

然后生成 robustness 曲线图，放到：

```text
results/figures/
```

运行方式：

```bash
python src/plot_robustness_curves.py --input-csv results/robustness/combined_robustness_results.csv --out-dir results/figures
```

---

## 5. `results/` 结果文件夹说明

### `results/clean/`

这里放 clean test set 上的预测结果和统一指标。

B/C/D 需要分别提供：

```text
results/clean/traditional_predictions.csv
results/clean/scratch_predictions.csv
results/clean/pretrained_predictions.csv
```

A 会用 `src/evaluate.py` 生成：

```text
results/clean/traditional_metrics.json
results/clean/scratch_metrics.json
results/clean/pretrained_metrics.json
```

### `results/robustness/`

这里放 robustness 相关文件。

新增模板文件：

```text
results/robustness/robustness_result_template.csv
```

这个文件告诉大家：

```text
1. 要测试哪些 degradation
2. severity 应该怎么写
3. prediction CSV 应该叫什么名字
4. metrics JSON 应该叫什么名字
```

建议结构：

```text
results/robustness/
├── robustness_result_template.csv
├── predictions/
│   ├── traditional_noise_0.05_predictions.csv
│   ├── scratch_noise_0.05_predictions.csv
│   └── pretrained_noise_0.05_predictions.csv
├── metrics/
│   ├── traditional_noise_0.05_metrics.json
│   ├── scratch_noise_0.05_metrics.json
│   └── pretrained_noise_0.05_metrics.json
└── combined_robustness_results.csv
```

### `results/figures/`

这里放最后 report 和 video 会用的图，例如：

```text
robustness_noise_top1_accuracy.png
robustness_noise_macro_f1.png
robustness_blur_top1_accuracy.png
robustness_jpeg_macro_f1.png
confusion_matrix_pretrained.png
```

---

## 6. B 同学任务：Traditional Model

B 负责 SIFT + Bag-of-Visual-Words + SVM。

使用数据：

```text
data/splits/train_paths.csv
data/splits/val_paths.csv
data/splits/test_paths.csv
```

读取方式：

```text
用 image_path 读取图片
用 class_idx 作为标签
```

需要给 A 的 clean 输出：

```text
results/clean/traditional_predictions.csv
```

需要给 A 的 robustness 输出：

```text
results/robustness/predictions/traditional_{degradation}_{severity}_predictions.csv
```

例如：

```text
results/robustness/predictions/traditional_noise_0.05_predictions.csv
results/robustness/predictions/traditional_blur_5_predictions.csv
results/robustness/predictions/traditional_jpeg_40_predictions.csv
```

---

## 7. C 同学任务：From-scratch Deep Model

C 负责从零训练的 CNN / ResNet18。

使用接口：

```python
from src.dataset import make_loader
```

需要给 A 的 clean 输出：

```text
results/clean/scratch_predictions.csv
```

需要给 A 的 robustness 输出：

```text
results/robustness/predictions/scratch_{degradation}_{severity}_predictions.csv
```

C 还应该提供训练日志、training curves 和 best checkpoint。

---

## 8. D 同学任务：Pretrained / Transfer Learning Model

D 负责 ImageNet-pretrained ResNet18 / EfficientNet。

使用接口和 C 一样：

```python
from src.dataset import make_loader
```

需要给 A 的 clean 输出：

```text
results/clean/pretrained_predictions.csv
```

需要给 A 的 robustness 输出：

```text
results/robustness/predictions/pretrained_{degradation}_{severity}_predictions.csv
```

D 还需要给 E 提供：

```text
best pretrained checkpoint
model architecture
class mapping
correct examples
incorrect examples
```

这些用于 Grad-CAM。

---

## 9. E 同学任务：Error Analysis / Grad-CAM / Robustness Discussion

E 主要使用 B/C/D 和 A 生成的结果。

主要输入：

```text
results/clean/*_predictions.csv
results/clean/*_metrics.json
results/robustness/combined_robustness_results.csv
results/figures/robustness_*.png
```

主要任务：

```text
1. confusion matrix analysis
2. hardest confused species pairs
3. correct and incorrect examples
4. Grad-CAM visualisation
5. robustness discussion
```

---

## 10. prediction CSV 统一格式

所有模型输出的 prediction CSV 都必须是：

```text
image_path,true_idx,pred_idx,top5_idx
```

示例：

```text
data/raw/val/example.jpg,12,18,18 12 35 7 209
```

注意：

```text
true_idx、pred_idx、top5_idx 都必须使用 0-499 的 class_idx。
不要使用 iNaturalist 原始 category_id。
```

---

## 11. robustness 总表格式

最终汇总文件位置：

```text
results/robustness/combined_robustness_results.csv
```

关键格式：

```text
model,degradation,severity,top1_accuracy,macro_f1
traditional,clean,0,0.120,0.095
traditional,noise,0.05,0.098,0.081
scratch,clean,0,0.180,0.150
pretrained,clean,0,0.420,0.390
```

这个文件由 A 用 `src/summarize_robustness.py` 生成。

---

## 12. 当前阶段每个人最应该做什么

A：继续维护 README、接口、evaluation 和 robustness 汇总脚本。等 B/C/D 给 prediction CSV 后统一评估。

B：实现 traditional model，并输出 `traditional_predictions.csv`。

C：实现 from-scratch deep model，并输出 `scratch_predictions.csv`。

D：实现 pretrained model，并输出 `pretrained_predictions.csv`，同时支持 E 做 Grad-CAM。

E：等模型结果出来后，开始做 Grad-CAM、confusion matrix、失败案例和 robustness discussion。

---

## 13. 最终提交提醒

最终代码 ZIP 不要包含：

```text
data/raw/
*.jpg
*.jpeg
*.png
*.pt
*.pth
*.ckpt
大体积压缩包
```

最终代码包应该包含：

```text
src/
data/splits/
results/clean/ 中的小型 csv/json
results/robustness/ 中的小型 csv/json
README.md
requirements.txt
必要的说明文档
```



大家后续把自己新增的代码和结果按下面结构来。

B traditional model:
代码放 src/train_traditional.py, src/predict_traditional.py
clean 结果给 results/clean/traditional_predictions.csv
robustness 结果给 results/robustness/predictions/traditional_{degradation}_{severity}_predictions.csv

C from-scratch model:
代码放 src/train_scratch.py, src/predict_scratch.py
clean 结果给 results/clean/scratch_predictions.csv
robustness 结果给 results/robustness/predictions/scratch_{degradation}_{severity}_predictions.csv
另外给 training log / training curve / best checkpoint

D pretrained model:
代码放 src/train_pretrained.py, src/predict_pretrained.py
clean 结果给 results/clean/pretrained_predictions.csv
robustness 结果给 results/robustness/predictions/pretrained_{degradation}_{severity}_predictions.csv
另外给 best checkpoint 和 model notes，方便 E 做 Grad-CAM

E analysis:
代码放 src/error_analysis.py, src/gradcam_analysis.py
图放 results/figures/
分析文字放 reports/E_error_analysis_notes.md

所有 prediction CSV 格式都必须是：
image_path,true_idx,pred_idx,top5_idx

其中 true_idx / pred_idx / top5_idx 都用 0-499 的 class_idx，不要用原始 category_id。

具体细节请看 README.md, PROJECT_TEAM_GUIDE_CN.md 和 reports/A_dataset_interface_notes.md。



最后的代码格式应该是类似这样的
COMP9517_Project/
├── src/
│   ├── prepare_dataset.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── evaluate.py
│   ├── degradations.py
│   ├── summarize_robustness.py
│   ├── plot_robustness_curves.py
│   ├── train_traditional.py
│   ├── predict_traditional.py
│   ├── train_scratch.py
│   ├── predict_scratch.py
│   ├── train_pretrained.py
│   ├── predict_pretrained.py
│   ├── error_analysis.py
│   └── gradcam_analysis.py
│
├── results/
│   ├── clean/
│   │   ├── traditional_predictions.csv
│   │   ├── traditional_metrics.json
│   │   ├── scratch_predictions.csv
│   │   ├── scratch_metrics.json
│   │   ├── pretrained_predictions.csv
│   │   └── pretrained_metrics.json
│   │
│   ├── robustness/
│   │   ├── robustness_result_template.csv
│   │   ├── predictions/
│   │   │   ├── traditional_noise_0.05_predictions.csv
│   │   │   ├── scratch_noise_0.05_predictions.csv
│   │   │   └── pretrained_noise_0.05_predictions.csv
│   │   ├── metrics/
│   │   │   ├── traditional_noise_0.05_metrics.json
│   │   │   ├── scratch_noise_0.05_metrics.json
│   │   │   └── pretrained_noise_0.05_metrics.json
│   │   └── combined_robustness_results.csv
│   │
│   └── figures/
│       ├── robustness_noise_top1_accuracy.png
│       ├── robustness_noise_macro_f1.png
│       ├── confusion_matrix_pretrained.png
│       ├── gradcam_correct_examples.png
│       └── gradcam_wrong_examples.png
│
├── reports/
│   ├── A_dataset_interface_notes.md
│   ├── D_pretrained_model_notes.md
│   └── E_error_analysis_notes.md
│
├── data/
│   ├── raw/
│   └── splits/
│
├── README.md
├── PROJECT_TEAM_GUIDE_CN.md
└── requirements.txt