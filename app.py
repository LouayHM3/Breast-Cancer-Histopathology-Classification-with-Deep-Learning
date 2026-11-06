"""Small Streamlit inference demo for the research prototype."""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

from src.advanced_pipeline import ResNetGradCAM


st.set_page_config(page_title="BreaKHis Research Demo", layout="centered")
st.title("Breast Histopathology Research Demo")
st.warning("Research prototype only. This output is not a medical diagnosis.")

model_file = st.sidebar.text_input("Model file", "artifacts/resnet18.pt")
uploaded = st.file_uploader("Upload an RGB histopathology image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)
    checkpoint = Path(model_file)
    if not checkpoint.exists():
        st.error(f"Model file not found: {checkpoint}")
        st.stop()
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    label = ["benign", "malignant"][index]
    st.metric("Prediction", label, f"confidence {confidence:.1%}")
    if confidence < 0.70:
        st.info("Low-confidence case: human review is required.")
    cam = ResNetGradCAM(model, model.layer4[-1])
    heatmap = cam(tensor, class_index=index).squeeze().detach().numpy()
    cam.close()
    st.image(heatmap, caption="Grad-CAM activation map", clamp=True)
