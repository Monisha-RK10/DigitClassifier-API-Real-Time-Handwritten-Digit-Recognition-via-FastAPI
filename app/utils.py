# app/utils.py

# Step 2
# This piece of code does the following:
# Preprocesses the image as per MNIST dataset before performing inference
  # Grayscale: Converts them to grayscale
  # Resize: Resizes them to (28x28)
  # ToTensor: Converts PIL image (28x28 grayscale) to PyTorch tensor [1, 28, 28] and scales pixel values from [0, 255] to [0.0, 1.0]
  # Normalize: Substracts the mean (0.1307) and divides by std (0.3081)
  # Adds batch dimension

from PIL import Image
import torchvision.transforms as transforms
import torch

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),                       # MNIST dataset: Grayscale
        transforms.Resize((28, 28)),                  # MNIST dataset: 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))    # MNIST dataset: Mean and std values
    ])
    image = transform(image).unsqueeze(0)             # Adds batch dim -> 1x1x28x28
    return image
