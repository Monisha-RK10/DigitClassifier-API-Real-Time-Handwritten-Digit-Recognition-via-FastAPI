# DigitClassifier API: Real-Time Handwritten Digit Recognition via FastAPI

## Description of Use Case

### Real-Time Handwritten Digit Recognition

- This project implements a lightweight, modular FastAPI application that allows users to classify handwritten digits (0–9) using a trained CNN model on the MNIST dataset.
- Users can either upload an image or simulate a camera capture.
- The goal is to demonstrate how a simple computer vision model can be wrapped inside a clean API, ready for testing, scaling, and hardware integration.

### Why MNIST?

- Simple and well-established benchmark dataset.
- Fast to train (a few minutes on CPU).
- Suitable for testing end-to-end API integration.
- Ideal to simulate camera-based inference pipelines.

> This setup mirrors real-world applications where a physical camera might capture digits (e.g., meter readings, postal codes, or handwritten forms), and a backend system processes the image for classification.
>
> The focus is on clean API design, modular codebase (training, prediction, utils, camera), and extensibility, making it a strong foundation for scaling up to real-time industrial vision tasks.

---

## Dataset Source

The model is trained on the **MNIST** dataset, a classic benchmark dataset for handwritten digit classification.

- **Name:** Modified National Institute of Standards and Technology (MNIST)
- **Classes:** 10 (digits 0–9)
- **Size:** 70,000 grayscale images (60,000 for training, 10,000 for testing)
- **Image Size:** 28×28 pixels
- **Format:** PNG (converted to tensor format during preprocessing)

The dataset is automatically downloaded using `torchvision.datasets.MNIST` and stored locally under the `./data/` directory if not already present.

**Source:**  [https://pytorch.org/vision/stable/datasets.html#mnist](https://pytorch.org/vision/stable/datasets.html#mnist)

> No manual download or preprocessing is required, it is fully handled by the training pipeline.

---

## Steps to Run API

---

## Example API Call

---

## Handling Real-Time Camera Input in Production

---

## Project Structure

```bash

Image_Classification_using_CNN_on_MNIST (Python & FastAPI)/
│
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── camera.py        # Simulated camera module
│   ├── model.py         # Model training and loading
│   ├── predict.py       # Prediction logic
│   ├── utils.py         # Preprocessing, etc.
│   ├── mnist_cnn.pth    # Model weights
│   ├── mock_camera_feed # Camera Flag = True, i.e., no image, only flag
│   ├──README.md        
│
├── test_images          # Image Upload (from disk), Client sends an attached image file i.e., Camera Flag = False
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md

```
---
### Sample Output from Render

---
