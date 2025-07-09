# app/camera.py
# Step 4
# This code does the following:
# Look inside app/mock_camera_feed
# Pick a random image like digit_5.png
# Open it as a PIL image
# Send it to the model

from pathlib import Path
from PIL import Image
import random

def capture_image_from_virtual_camera():
    camera_folder = Path("app/mock_camera_feed")
    sample_images = list(camera_folder.glob("*.png"))
    if not sample_images:
        raise FileNotFoundError("No images in virtual camera feed.")
    return Image.open(random.choice(sample_images))
