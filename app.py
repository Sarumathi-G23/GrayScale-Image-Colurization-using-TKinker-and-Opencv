import streamlit as st
import numpy as np
import cv2
import os
import requests

st.set_page_config(page_title="Image Colorization", layout="centered")

st.title("🖤 Grayscale Image Colorization using Deep Learning")

# -----------------------------
# Create model folder
# -----------------------------
if not os.path.exists("model"):
    os.makedirs("model")

# -----------------------------
# File paths
# -----------------------------
PROTOTXT_PATH = os.path.join("model", "colorization_deploy_v2.prototxt")
POINTS_PATH = os.path.join("model", "pts_in_hull.npy")
MODEL_PATH = os.path.join("model", "colorization_release_v2.caffemodel")

# -----------------------------
# Safe download function
# -----------------------------
def download_file(url, save_path):
    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)

# -----------------------------
# Download required files
# -----------------------------
if not os.path.exists(PROTOTXT_PATH):
    st.info("Downloading prototxt file...")
    download_file(
        "https://github.com/richzhang/colorization/raw/master/models/colorization_deploy_v2.prototxt",
        PROTOTXT_PATH
    )

if not os.path.exists(POINTS_PATH):
    st.info("Downloading pts_in_hull.npy...")
    download_file(
        "https://github.com/richzhang/colorization/raw/master/resources/pts_in_hull.npy",
        POINTS_PATH
    )

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model file (this may take 1-2 minutes)...")
    download_file(
        "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
        MODEL_PATH
    )
    st.success("Model downloaded successfully!")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    st.info("Colorizing image...")

    # -----------------------------
    # Load Model
    # -----------------------------
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    pts = np.load(POINTS_PATH)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # -----------------------------
    # Preprocess
    # -----------------------------
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # -----------------------------
    # Colorization
    # -----------------------------
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L_original = cv2.split(lab)[0]

    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    st.subheader("Colorized Image")
    st.image(colorized, channels="BGR")

    st.success("Done! 🎉")
