import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from resnet18_cifar10 import create_dataset, ImageModel, device, BATCH_SIZE
import random

# 模型对比图绘制
def read_text_auto(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"无法解码文件: {path}")

def parse_log(log_text: str) -> pd.DataFrame:
    # 兼容两种格式：
    # 1) base: epoch, loss, acc, time
    # 2) best: epoch, loss, acc, val_acc, lr, time
    pattern = re.compile(
        r"epoch:(\d+),\s*loss:([0-9.]+),\s*acc:([0-9.]+)"
        r"(?:,\s*val_acc:([0-9.]+),\s*lr:([0-9.]+))?"
        r",\s*time:([0-9.]+)s"
    )

    rows = []
    for line in log_text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        rows.append(
            {
                "epoch": int(m.group(1)),
                "loss": float(m.group(2)),
                "acc": float(m.group(3)),
                "val_acc": float(m.group(4)) if m.group(4) else None,
                "lr": float(m.group(5)) if m.group(5) else None,
                "time": float(m.group(6)),
            }
        )

    if not rows:
        raise ValueError("日志解析失败：未匹配到 epoch 记录")
    return pd.DataFrame(rows)

def main():
    base_dir = Path(__file__).resolve().parent
    vis_dir = base_dir / "可视化"
    fig_dir = vis_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_log = vis_dir / "base_train_data.txt"
    best_log = vis_dir / "best_train_data.txt"

    if not base_log.exists():
        raise FileNotFoundError(f"未找到基础日志: {base_log}")
    if not best_log.exists():
        raise FileNotFoundError(f"未找到最终日志: {best_log}")

    df_base = parse_log(read_text_auto(base_log))
    df_best = parse_log(read_text_auto(best_log))

    # 只对齐共同 epoch 区间，避免长度不一致影响观感
    max_epoch = min(df_base["epoch"].max(), df_best["epoch"].max())
    df_base = df_base[df_base["epoch"] <= max_epoch]
    df_best = df_best[df_best["epoch"] <= max_epoch]

    plt.figure(figsize=(12, 5))

    # 左图：训练准确率对比
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(df_base["epoch"], df_base["acc"], label="Base CNN train_acc", linewidth=2)
    ax1.plot(df_best["epoch"], df_best["acc"], label="ResNet18 train_acc", linewidth=2)
    ax1.set_title("Train Accuracy Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 右图：训练损失对比
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(df_base["epoch"], df_base["loss"], label="Base CNN train_loss", linewidth=2)
    ax2.plot(df_best["epoch"], df_best["loss"], label="ResNet18 train_loss", linewidth=2)
    ax2.set_title("Train Loss Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = fig_dir / "compare_base_vs_best.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"已生成: {out_path}")
    print(
        "最终训练准确率: "
        f"Base={df_base['acc'].iloc[-1]:.3f}, "
        f"Best={df_best['acc'].iloc[-1]:.3f}"
    )
    print(
        "最终训练损失: "
        f"Base={df_base['loss'].iloc[-1]:.3f}, "
        f"Best={df_best['loss'].iloc[-1]:.3f}"
    )

# 混淆矩阵绘制
def export_confusion_matrix_resnet18(
        test_dataset,
        weight_path='model/image_model_best.pth',
        figure_path='可视化/figures/confusion_matrix_resnet18.png',
        csv_path='可视化/confusion_matrix_resnet18.csv'):
    base_dir = Path(__file__).resolve().parent
    weight_file = Path(weight_path)
    if not weight_file.is_absolute():
        weight_file = base_dir / weight_file

    figure_file = Path(figure_path)
    if not figure_file.is_absolute():
        figure_file = base_dir / figure_file

    csv_file = Path(csv_path)
    if not csv_file.is_absolute():
        csv_file = base_dir / csv_file

    # 仅用测试集和已训练好的最优权重，生成最终模型的混淆矩阵
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel().to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()

    class_names = test_dataset.classes if hasattr(test_dataset, 'classes') else [str(i) for i in range(10)]
    num_classes = len(class_names)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            # cm[真实类别, 预测类别] 计数加一
            for real_label, pred_label in zip(y, pred):
                cm[real_label.long(), pred_label.long()] += 1

    # 保存原始计数矩阵（csv），便于后续文档分析
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('true\\pred,' + ','.join(class_names) + '\n')
        for i in range(num_classes):
            row_values = ','.join(str(int(v)) for v in cm[i].tolist())
            f.write(f'{class_names[i]},{row_values}\n')

    # 画按行归一化矩阵，表示“该真实类别被预测到各类别的比例”
    cm_float = cm.float()
    row_sum = cm_float.sum(dim=1, keepdim=True).clamp(min=1.0)
    cm_norm = cm_float / row_sum

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm.tolist(), cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_title('ResNet18 Confusion Matrix (Normalized)')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j].item()
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if val > 0.5 else 'black', fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    figure_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_file, dpi=300)
    plt.close(fig)

    print(f'Confusion matrix saved: {figure_file}')
    print(f'Confusion matrix csv saved: {csv_file}')

# 生成预测样例图
def export_prediction_samples_resnet18(
        test_dataset,
        weight_path='model/image_model_best.pth',
        figure_path='可视化/figures/prediction_samples_resnet18.png',
        num_correct=6,
        num_wrong=6,
        seed=42):
    """
    生成预测样例图：
    - 第一行：预测正确样本
    - 第二行：预测错误样本
    """
    base_dir = Path(__file__).resolve().parent

    weight_file = Path(weight_path)
    if not weight_file.is_absolute():
        weight_file = base_dir / weight_file

    figure_file = Path(figure_path)
    if not figure_file.is_absolute():
        figure_file = base_dir / figure_file

    # CIFAR10 标准化参数（与训练/测试一致），用于反标准化显示
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    def denormalize(img_tensor):
        # img_tensor: [3, H, W], 已标准化
        img = img_tensor.cpu() * std + mean
        return img.clamp(0, 1)

    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel().to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()

    class_names = test_dataset.classes if hasattr(test_dataset, "classes") else [str(i) for i in range(10)]

    correct_samples = []  # (img, true_label, pred_label)
    wrong_samples = []    # (img, true_label, pred_label)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            for i in range(len(y)):
                img_cpu = x[i].detach().cpu()
                true_label = int(y[i].item())
                pred_label = int(preds[i].item())

                if true_label == pred_label:
                    correct_samples.append((img_cpu, true_label, pred_label))
                else:
                    wrong_samples.append((img_cpu, true_label, pred_label))

            # 收集够了就提前结束，减少耗时
            if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
                break

    if len(correct_samples) == 0 and len(wrong_samples) == 0:
        raise RuntimeError("未收集到样本，请检查测试集或模型权重。")

    # 为了展示更自然，从已收集样本中随机抽取（可复现）
    rng = random.Random(seed)
    if len(correct_samples) > num_correct:
        correct_samples = rng.sample(correct_samples, num_correct)
    if len(wrong_samples) > num_wrong:
        wrong_samples = rng.sample(wrong_samples, num_wrong)

    n_cols = max(len(correct_samples), len(wrong_samples), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        # 统一成二维索引方式 axes[row, col]
        axes = [[axes[0]], [axes[1]]]

    # 第一行：正确样本
    for c in range(n_cols):
        ax = axes[0][c]
        if c < len(correct_samples):
            img, t, p = correct_samples[c]
            ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
            ax.set_title(f"Correct\nT:{class_names[t]} / P:{class_names[p]}", fontsize=9)
        ax.axis("off")

    # 第二行：错误样本
    for c in range(n_cols):
        ax = axes[1][c]
        if c < len(wrong_samples):
            img, t, p = wrong_samples[c]
            ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
            ax.set_title(f"Wrong\nT:{class_names[t]} / P:{class_names[p]}", fontsize=9, color="red")
        ax.axis("off")

    fig.suptitle("ResNet18 Prediction Samples (Top: Correct, Bottom: Wrong)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    figure_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_file, dpi=220)
    plt.close(fig)

    print(f"Prediction samples figure saved: {figure_file}")

if __name__ == "__main__":
    #main()

    # 默认不导出混淆矩阵，避免影响常规曲线分析流程
    ENABLE_CONFUSION_MATRIX = False
    if ENABLE_CONFUSION_MATRIX:
        _, test_dataset = create_dataset()
        export_confusion_matrix_resnet18(test_dataset)

    ENABLE_PRED_SAMPLES = False
    if ENABLE_PRED_SAMPLES:
        _, test_dataset = create_dataset()
        export_prediction_samples_resnet18(
            test_dataset,
            num_correct=6,
            num_wrong=6
        )

