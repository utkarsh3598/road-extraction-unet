import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from model import UNet  # Make sure model.py is in same folder
st.set_page_config(page_title="Road Extraction App", layout="wide")
st.title("🛣️ Road Extraction from Satellite Images")
# =============================
# CONFIG
# =============================
MODEL_PATH = "checkpoints/road_unet_epoch70.pth"
IMG_SIZE = (128, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

model = load_model()

# =============================
# UI
# =============================

st.caption("Upload a satellite image to predict the road layout using a trained U-Net model.")

uploaded_file = st.file_uploader("Upload a .jpg or .png file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 📷 Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🧼 Preprocess
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 🔮 Predict
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.3).float()

    # 🎨 Post-process
    pred_np = pred.squeeze().cpu().numpy() * 255
    pred_img = Image.fromarray(pred_np.astype(np.uint8))

    # 🖼️ Display prediction
    st.image(pred_img, caption="🧠 Predicted Road Mask", use_column_width=True)
    st.success("Prediction complete!")

    # 💾 Save option
    with open("predicted_mask.jpg", "wb") as f:
        pred_img.save(f)
    st.download_button("Download Predicted Mask", data=open("predicted_mask.jpg", "rb"), file_name="road_mask.jpg", mime="image/jpeg")

