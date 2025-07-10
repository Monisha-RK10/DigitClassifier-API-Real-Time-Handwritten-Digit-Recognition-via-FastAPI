# app/camera.py

# Step 4
# This code does the following:
# Looks inside app/mock_camera_feed
# Picks a random image like digit 2 (camera_digit_2_1.png) to mimic a real-world camera
# Opens it as a PIL image
# Sends it to the model

from pathlib import Path
from PIL import Image
import random

def capture_image_from_virtual_camera():
    camera_folder = Path("app/mock_camera_feed")
    sample_images = list(camera_folder.glob("*.png"))
    if not sample_images:
        raise FileNotFoundError("No images in virtual camera feed.")
    return Image.open(random.choice(sample_images))                          # Mimics a real-world camera (picks random images like a feed)
