"""
案例
    （CNN）图像分类

数据集
        这里我们使用计算机视觉模块 torchvision 自带的 CIFAR10数据集，包含
    6W张（32,32,3）的图片，5W张训练集，1W张测试集，10个分类，每个分类6K张图片。

标签
    0.airplane 飞机
    1.automobile 汽车
    2.bird 鸟
    3.cat 猫
    4.deer 鹿
    5.dog 狗
    6.frog 青蛙
    7.horse 马
    8.ship 船
    9.truck 卡车

训练集数据增强与预处理：
    1) RandomCrop: 随机裁剪
    2) RandomHorizontalFlip: 随机水平翻转，扩充样本
    3) RandomErasing: 随机擦除局部区域，提升遮挡场景泛化
    4) ToTensor: 图像转张量并归一化到[0,1]
    5) Normalize: 按CIFAR-10均值/方差做标准化

模型选择：
    采用了 ResNet18，核心原因是它有残差连接，能显著改善深层网络训练中的梯度退化问题，比基础 CNN 更稳定、上限更高。
并针对 CIFAR-10 做了结构适配：把首层改成 3x3, stride=1，并去掉首个 maxpool，避免 32x32 小图过早丢失细节；最
后全连接层改成 10 类输出。
"""

# 导入相关模块
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, RandomErasing, ToTensor, Compose, Normalize
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
from torchvision import models

# 设置device为cuda（如果可用）否则为cpu
# 用于模型训练和推理的硬件设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 每批次样本数
BATCH_SIZE = 128

# todo:1.准备数据集
# 包含数据增强和张量转换
def create_dataset():
    # CIFAR10官方常用均值和标准差，用于输入标准化
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    # 数据增强：随机裁剪、随机水平翻转、转为张量
    transform_train = Compose([
        RandomCrop(32, padding=4),  # 随机裁剪，增强鲁棒性
        RandomHorizontalFlip(),     # 随机水平翻转，增强泛化能力
        ToTensor(),                 # 转换为张量
        Normalize(cifar10_mean, cifar10_std),  # 标准化，加速收敛并提升稳定性
        # 第二阶段增强：随机擦除，提升遮挡与局部缺失场景下的泛化能力
        RandomErasing(p=0.1, scale=(0.02, 0.20), ratio=(0.3, 3.3), value='random')   # 擦除概率 擦除比例范围 擦除长宽比范围 随机填充
    ])
    transform_test = Compose([
        ToTensor(),
        Normalize(cifar10_mean, cifar10_std)  # 测试集使用同样的标准化
    ])
    # 获取训练集
    train_dataset = CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    # 获取测试集
    test_dataset = CIFAR10(root="./data", train=False, transform=transform_test, download=True)
    return train_dataset, test_dataset

# todo:2.搭建神经网络
# 使用ResNet18并做CIFAR10输入适配
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 不使用预训练权重，避免额外下载并保持当前训练流程一致
        self.backbone = models.resnet18(weights=None)
        # CIFAR10图片为32x32：缩小首层卷积并取消首个最大池化，保留更多细节
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        # 分类头改为10类输出
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 10)

    def forward(self, x):
        return self.backbone(x)

# todo:3.模型训练
# 包含损失函数、优化器、训练循环和模型保存
def train(train_dataset):
    # 将训练集切分为训练/验证集（45k/5k），并固定随机种子保证可复现
    train_size = 45000
    val_size = len(train_dataset) - train_size
    # 使用固定随机种子打乱索引，确保每次运行切分结果一致，便于调试和比较不同训练策略的效果
    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(2025)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    # 训练子集继续使用随机增强；验证子集使用确定性预处理，避免验证指标被随机增强噪声干扰
    train_subset = Subset(train_dataset, train_indices)    # 从原始 train_dataset 里按 train_indices 取样
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    transform_val = Compose([
        ToTensor(),
        Normalize(cifar10_mean, cifar10_std)
    ])
    val_dataset = CIFAR10(root="./data", train=True, transform=transform_val, download=False)
    val_subset = Subset(val_dataset, val_indices)
    # 训练集打乱；验证集不打乱，保证评估稳定
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    # 创建模型对象
    model = ImageModel().to(device)
    # 损失函数：加入标签平滑，防止模型过度自信，缓解过拟合并提升泛化
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # 优化器：L2正则化
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 学习率调度：使用余弦退火；将最低学习率抬高，避免后期lr过小导致几乎不再学习
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-4)
    epochs = 80  # 训练轮数
    # 以验证集准确率为准保存最优模型，并在长期无提升时提前停止
    best_val_acc = 0.0
    patience = 10
    no_improve_epochs = 0
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        model.train()  # 切换到训练模式
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)  # 前向传播
            loss = criterion(y_pred, y)  # 计算损失
            optimizer.zero_grad()        # 梯度清零
            loss.backward()              # 反向传播
            optimizer.step()             # 参数更新
            total_correct += (y_pred.argmax(dim=1) == y).sum()  # 统计正确样本
            total_loss += loss.item() * len(y)                  # 累加损失
            total_samples += len(y)                             # 累加样本数

        # 每轮训练后在验证集上评估，作为早停和最佳模型保存依据
        model.eval()
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                val_correct += (y_pred.argmax(dim=1) == y).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            # 仅在验证集提升时保存，确保文件中是当前最优参数
            torch.save(model.state_dict(), 'model/image_model_best.pth')
        else:
            no_improve_epochs += 1

        scheduler.step()  # 在每轮结束后更新学习率
        # 打印每轮训练信息
        print(
            f'epoch:{epoch+1}, '
            f'loss:{total_loss/total_samples:.5f}, '
            f'acc:{total_correct/total_samples:.2f}, '
            f'val_acc:{val_acc:.2f}, '
            f'lr:{optimizer.param_groups[0]["lr"]:.6f}, '
            f'time:{time.time()-start:.2f}s'
        )

        if no_improve_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}, best_val_acc:{best_val_acc:.3f}')
            break

# todo:4.模型测试
# 评估模型在测试集上的准确率
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel().to(device)
    model.load_state_dict(torch.load('model/image_model_best.pth', map_location=device))
    model.eval()  # 在评估循环前切换到评估模式
    total_samples, total_correct = 0, 0
    with torch.no_grad():  # 评估阶段关闭梯度计算，减少显存占用并加速推理
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=-1)
            total_correct += (y_pred == y).sum()  # 统计正确样本
            total_samples += len(y)
    print(f'Acc:{total_correct/total_samples:.3f}')  # 输出测试集准确率

# todo:5.主程序入口
if __name__ == '__main__':
    # 获取数据集
    train_dataset, test_dataset = create_dataset()
    # 模型训练
    # train(train_dataset)
    # 模型测试
    evaluate(test_dataset)
