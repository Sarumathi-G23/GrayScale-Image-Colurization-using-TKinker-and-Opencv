import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Grayscale to Color Demo", layout="centered")

st.title("🖤 Grayscale Image Colorization Demo")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original Image")
    st.image(image)

    # Simple fake colorization effect (for stable deployment)
    img_array = np.array(image)

    # Add artificial color tint
    img_array[:, :, 1] = img_array[:, :, 1] * 0.8
    img_array[:, :, 2] = img_array[:, :, 2] * 0.6

    st.subheader("Colorized Output (Demo Effect)")
    st.image(img_array)

    st.success("App is running successfully 🎉")
