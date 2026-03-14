from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from resnet18_cifar10.py import ImageModel, device

# CIFAR-10 标签
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 与训练脚本保持一致的标准化参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@st.cache_resource
def load_model(weight_path: Path) -> ImageModel:
    model = ImageModel().to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_transform():
    return Compose([
        Resize((32, 32)),  # 适配 CIFAR-10 输入大小
        ToTensor(),
        Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def predict(image: Image.Image, model: ImageModel):
    transform = build_transform()
    x = transform(image).unsqueeze(0).to(device)  # [1, 3, 32, 32]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    topk_prob, topk_idx = torch.topk(probs, k=3)
    return probs, topk_prob.tolist(), topk_idx.tolist()


def main():
    st.set_page_config(page_title="CIFAR-10 图像分类", layout="wide")
    st.title("CIFAR-10 图像分类")
    st.caption("上传一张图片，模型会输出预测类别与 Top-3 概率")

    base_dir = Path(__file__).resolve().parent
    weight_path = base_dir / "model" / "image_model_best.pth"

    if not weight_path.exists():
        st.error(f"未找到模型权重文件：{weight_path}")
        st.stop()

    model = load_model(weight_path)

    uploaded = st.file_uploader("请选择图片文件", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.info("请先上传图片。")
        st.stop()

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("输入图片")
        st.image(image, use_container_width=True)

    probs, topk_prob, topk_idx = predict(image, model)
    pred_idx = int(torch.argmax(probs).item())
    pred_name = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx].item())

    with col2:
        st.subheader("预测结果")
        st.success(f"预测类别：**{pred_name}**")
        st.write(f"置信度：`{pred_conf:.2%}`")

        top3_df = pd.DataFrame({
            "class": [CLASS_NAMES[i] for i in topk_idx],
            "probability": topk_prob
        })
        st.write("Top-3 概率：")
        chart = (
            alt.Chart(top3_df)
            .mark_bar()
            .encode(
                x=alt.X("class:N", sort=None, axis=alt.Axis(labelAngle=0, title="类别")),
                y=alt.Y("probability:Q", axis=alt.Axis(format=".2f", title="概率")),
                tooltip=[
                    alt.Tooltip("class:N", title="类别"),
                    alt.Tooltip("probability:Q", title="概率", format=".2%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            "#### 说明\n"
            "- 该模型在 CIFAR-10 上训练，输入会被缩放到 `32x32`。\n"
            "- 对非 CIFAR-10 风格图片，预测可能不稳定，这是正常现象。"
        )


if __name__ == "__main__":
    main()
