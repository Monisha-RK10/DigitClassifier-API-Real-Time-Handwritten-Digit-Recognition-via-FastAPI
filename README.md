# DigitClassifier API: Real-Time Handwritten Digit Recognition via FastAPI

## Use Case: Real-Time Handwritten Digit Recognition

> This project implements a lightweight, modular FastAPI application that allows users to classify handwritten digits (0–9) using a trained CNN model on the MNIST dataset.
> Users can either upload an image or simulate a camera capture.
> The goal is to demonstrate how a simple computer vision model can be wrapped inside a clean API, ready for testing, scaling, and hardware integration.

---

## Dataset

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
