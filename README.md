---
title: Road Extraction UNet
emoji: 🛣
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# 🛣 Road Extraction Using U-Net

This web app extracts road networks from satellite imagery using a *U-Net deep learning model* trained on high-resolution aerial images.

## 🔗 *Live Demo*
👉 [Click here to try the app](https://huggingface.co/spaces/<YourUsername>/<SpaceName>)

---

## 📌 *Features*
✅ Real-time road extraction from uploaded satellite images  
✅ Side-by-side input and predicted mask comparison  
✅ Downloadable road mask (PNG)  
✅ Deployed publicly via Hugging Face Spaces  

---

## 🚀 *Model Info*
- *Architecture*: U-Net  
- *Trained on*: 1108 satellite images & corresponding road masks  
- *Framework*: PyTorch  
- *Input Size*: 256×256  

---

## ⚡ *How to Use*
1. Upload a satellite image (.jpg or .png).  
2. Click on *Predict Roads*.  
3. View the predicted road mask and download it.  

---

## 🖼 *Sample Output*
| Input Image | Predicted Road Mask |
|-------------|----------------------|
| ![Input](https://via.placeholder.com/150) | ![Mask](https://via.placeholder.com/150) |

---

## 🏗 *Tech Stack*
- *Python, PyTorch, OpenCV* – Model Training  
- *Streamlit* – Interactive Web UI  
- *Hugging Face Spaces* – Deployment

---

## 👩‍💻 *Author*
Developed by *Riya Bhardwaj* | [LinkedIn](https://www.linkedin.com)