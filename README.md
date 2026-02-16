# Deeplearning
# Grayscale Image Colorization using Deep Learning

This is a simple Python project I worked on to convert black and white (grayscale) images into color using deep learning. The model is based on OpenCV's DNN module and uses a pre-trained Caffe model to predict color information
# What It Does
- Takes a grayscale image as input.
- Uses a deep neural network to predict the missing color channels.
- Reconstructs and saves a full-color version of the image.
- Also shows the original and colorized images side by side for comparison.

# Tools and Libraries Used
- Python  
- OpenCV  
- NumPy  
- Matplotlib  
- Tkinter (for selecting images through a file dialog)
# Model Files Needed
To run this project, download the following files and place them inside a folder named `model/`:
- `colorization_deploy_v2.prototxt`  
- `colorization_release_v2.caffemodel`  
- `pts_in_hull.npy`  

These files are available from OpenCV and Richard Zhang’s official repositories. You can find the links easily online.
# How to Run
1. Make sure you have Python installed.
2. Install the required libraries:
   ```bash
   pip install opencv-python numpy matplotlib
