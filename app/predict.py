# app/predict.py

# Step 3:
# This piece of code does the following:
# Loads trained model's parameters from model.py
# Loads preprocessing function from utils.py
# Perform inference on the preprocessed image using trained weights

from app.model import load_model
from app.utils import preprocess_image
import torch

model = load_model()

def predict_digit(image)
    tensor = preprocess_image(image)                 # 1x1x28x28
    with torch.no_grad():                            # No gradient computation
        output = model(tensor)
        prediction = output.argmax(dim=1).item()     # Prediction with highest probability score
    return prediction
