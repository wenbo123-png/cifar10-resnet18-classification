# CIFAR-10 ResNet18 Classification (PyTorch)

使用 **PyTorch** 在 **CIFAR-10** 数据集上训练/评估一个针对 `32×32` 小图做过结构适配的 **ResNet18** 分类模型，并提供：

- 最优模型权重：`model/image_model_best.pth`
- 训练可视化与混淆矩阵：`可视化/figures/`
- 测试图：`测试图片/`
- Streamlit 交互 Demo：`图像分类app.py`
- 日志解析与可视化、混淆矩阵/预测样例导出工具：`cnn_analysis_tools.py`

> 数据集不上传到仓库：运行时使用 `torchvision.datasets.CIFAR10(download=True)` 自动下载到本地 `./data`。

---

## Results
- **Test Accuracy：93.1%**（你可以按实际情况在这里更新/补充训练配置）

### Training Curves
| Accuracy | Loss |
| --- | --- |
| ![](可视化/figures/acc_curve.png) | ![](可视化/figures/loss_curve.png) |

| Learning Rate | Time / Epoch |
| --- | --- |
| ![](可视化/figures/lr_curve.png) | ![](可视化/figures/time_curve.png) |

### Base CNN vs ResNet18 (Comparison)
![](可视化/figures/compare_base_vs_best.png)

### Confusion Matrix (ResNet18)
![](可视化/figures/confusion_matrix_resnet18.png)

### Prediction Samples
![](可视化/figures/prediction_samples_resnet18.png)

---

## Environment & Installation
- Python 3.x
- PyTorch / torchvision
- Streamlit（用于 Web Demo）

安装依赖：

```bash
pip install -r requirements.txt
```

> 说明：`requirements.txt` 里包含了一些 Jupyter 相关依赖；如果你只想运行训练与 Demo，后续可以再精简依赖文件。

---

## Quick Start

### 1) Evaluate (default)
`CNN_CIFAR10图像分类.py` 当前默认执行 `evaluate(test_dataset)`，并从 `model/image_model_best.pth` 加载权重：

```bash
python CNN_CIFAR10图像分类.py
```

输出示例：`Acc:0.xxx`

### 2) Train (optional)
在 `CNN_CIFAR10图像分类.py` 的主入口中，将训练行取消注释：

```python
# train(train_dataset)
```

改为：

```python
train(train_dataset)
```

然后运行：

```bash
python CNN_CIFAR10图像分类.py
```

训练过程中会按验证集准确率保存最优权重到：

- `model/image_model_best.pth`

---

## Streamlit Demo
启动交互演示：

```bash
streamlit run 图像分类app.py
```

功能：
- 上传一张图片，输出预测类别与 **Top-3 概率**（柱状图）
- Demo 默认加载权重：`model/image_model_best.pth`

> 注意：该模型在 CIFAR-10 上训练，输入会被缩放到 `32x32`。对非 CIFAR-10 风格图片预测可能不稳定（正常现象）。

---

## Visualization & Analysis
运行日志解析与可视化脚本（会读取 `可视化/` 下的日志文件并生成图像到 `可视化/figures/`）：

```bash
python cnn_analysis_tools.py
```

另外，该脚本也提供：
- `export_confusion_matrix_resnet18(...)`
- `export_prediction_samples_resnet18(...)`

默认通过开关控制（`ENABLE_CONFUSION_MATRIX`、`ENABLE_PRED_SAMPLES`），需要时可改为 `True`。

---

## Project Structure
```text
.
├─ model/
│  └─ image_model_best.pth
├─ 可视化/
│  ├─ base_train_data.txt
│  ├─ best_train_data.txt
│  └─ figures/
│     ├─ acc_curve.png
│     ├─ loss_curve.png
│     ├─ lr_curve.png
│     ├─ time_curve.png
│     ├─ compare_base_vs_best.png
│     ├─ confusion_matrix_resnet18.png
│     └─ prediction_samples_resnet18.png
├─ 测试图片/
├─ CNN_CIFAR10图像分类.py
├─ cnn_analysis_tools.py
├─ 图像分类app.py
├─ requirements.txt
└─ README.md
```

---

## Notes
- CIFAR-10 会自动下载到本地 `./data`（该目录不建议提交到仓库）。
- 代码会自动选择设备：有 CUDA 则用 GPU，否则用 CPU。

---

## License
MIT License（见 `LICENSE`）。
