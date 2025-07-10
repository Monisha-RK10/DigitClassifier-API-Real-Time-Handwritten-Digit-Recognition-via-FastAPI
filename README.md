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

## Optional: Deploying on Render.com (Public URL)

To test the API online, you can deploy it using [Render.com](https://render.com/):

### Steps:

  1. Go to [https://render.com](https://render.com)
  2. Click on `New` → `Web Service`
  3. Select `Public Git Repository`
  4. Paste this GitHub repo URL: https://github.com/Monisha-RK10/DigitClassifier-API-Real-Time-Handwritten-Digit-Recognition-via-FastAPI 
  6. Click `Connect Repository`
  7. Under **Region**, select `Frankfurt (EU Central)` or your preferred one.
  8. Under **Instance Type**, select `Free (for hobby projects)`
  9. Leave the build command empty (FastAPI auto detects)
  10. Click `Deploy Web Service` (Wait for ~15–20 min until the build is complete).
  11. Once done, visit: https://digitclassifier-api-real-time.onrender.com/docs (This will launch Swagger UI to interact with the `/predict` and `/health` endpoints).
  13. **Using the endpoints:**
      - **Health Check:**
        - Click `/health` → `Try it out` → `Execute`
        - Should return: `{ "status": "ok" }`
      
      - **Predict with Camera Simulation:**
        - Use `POST /predict`
        - Set:
          - `use_camera` = `true`
          - Leave `file` empty (uncheck “Send empty value”)
          - Returns a simulated digit prediction from test images
      
      - **Predict with Uploaded Image:**
        - Use `POST /predict`
        - Set:
          - `use_camera` = `false`
          - Upload an image (e.g., `app/test_images/3.png`)
          - Returns the predicted digit


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
