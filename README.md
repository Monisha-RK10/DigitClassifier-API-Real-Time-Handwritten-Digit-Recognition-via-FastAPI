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

> This setup mirrors real-world applications where a physical camera might capture digits (e.g., food processing, electronics where each product maybe stamped with a **quality grade digit** between 0 and 9), and a backend system processes the image for classification.
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

## Steps to Run API (Running Locally using uvicorn)

1. Clone this repo
   1. git clone https://github.com/Monisha-RK10/DigitClassifier-API-Real-Time-Handwritten-Digit-Recognition-via-FastAPI.git
   2. cd DigitClassifier-API-Real-Time-Handwritten-Digit-Recognition-via-FastAPI
2. (Optional) Create and activate a virtual environment
   1. python -m venv venv
   2. source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install the dependencies
   1. pip install -r requirements.txt
4. Run the FastAPI app
   1. uvicorn app.main:app --reload
5. Open your browser at:
   1. http://127.0.0.1:8000/docs

---

## Example API Calls (without Swagger UI)

### Health Check

```bash
curl http://127.0.0.1:8000/health
```
> **Expected Response:** `{ "status": "ok" }`

### Predict with Uploaded Image
  
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "use_camera=false" \
     -F "file=@app/upload_digit_1_2"
```
> **Expected Response:** `{ "predicted_class": 2 }`

### Predict with Camera Simulation

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "use_camera=true"
```
> **Expected Response (Random):** `{ "predicted_class": 2 }`

---

## Optional: Deploying on Render.com (Public URL)

To test the API online, you can deploy it using [Render.com](https://render.com/):

### Steps:

  1. Go to [https://render.com](https://render.com)
  2. Click on `New` → `Web Service`
  3. Select `Public Git Repository`
  4. Paste this GitHub repo URL: https://github.com/Monisha-RK10/DigitClassifier-API-Real-Time-Handwritten-Digit-Recognition-via-FastAPI 
  5. Click `Connect Repository`
  6. Under **Region**, select `Frankfurt (EU Central)` or your preferred one.
  7. Under **Instance Type**, select `Free (for hobby projects)`
  8. Leave the build command empty (FastAPI auto detects)
  9. Click `Deploy Web Service` (Wait for ~15–20 min until the build is complete).
  10. Once done, visit: https://digitclassifier-api-real-time.onrender.com/docs (This will launch Swagger UI to interact with the `/predict` and `/health` endpoints).
      > Note: Check the url on the render build screen, then visit the corresponding url's docs to use the endpoints
  12. **Use the endpoints:**
      1. **Health Check**:
           1. Click `/health` → `Try it out` → `Execute`
           2. Should return `{ "status": "ok" }`
      2. **Predict with Camera Simulation**:
           1. Click `/predict` → `Try it out`
           2. Set:
              1. `use_camera` = `true`
              2. Leave `file` empty (uncheck “Send empty value”)
          3. Returns a simulated prediction from test images
      3. **Predict with Uploaded Image**:
         1. Click `/predict` → `Try it out`
         2. Set:
            1. `use_camera` = `false`
            2. Upload a digit image (e.g., `app/test_images/upload_digit_2_1.png`)
         3. Returns the predicted digit


---

## Handling Real-Time Camera Input in Production

In a production setup, a real camera (e.g., USB webcam, IDS/Ximea industrial camera) would replace the simulated image source.

For example:
- A USB camera can be accessed using OpenCV's `cv2.VideoCapture(0)`
- An industrial camera might use a specific SDK (e.g., IDS uEye or Ximea APIs) to stream frames

The `/predict` endpoint would then connect to this live camera feed and periodically grab frames (e.g., 1 every 10 seconds), convert the frame to a compatible image format (e.g., PIL or NumPy), and pass it to the prediction pipeline.

In real-time setups:
- The FastAPI backend could use a background thread or a scheduler to run periodic captures
- OpenCV, GStreamer, or vendor SDKs would handle low-latency streaming
- The prediction output could be displayed on a dashboard, stored, or sent to downstream systems


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
│   ├── utils.py         # Preprocessing
│   ├── mnist_cnn.pth    # Model weights
│   ├── mock_camera_feed # Camera Flag = True, i.e., no image, only flag
│   ├──README.md        
│
├── test_images          # Image Upload (from disk), Client sends an attached image file i.e., Camera Flag = False
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
│
├── output              # Swagger UI and sample predicion for camera simulation

```
---
### Sample Output using Render

Visit: https://digitclassifier-api-real-time.onrender.com/docs

#### Swagger UI

![Swagger UI Screenshot](output/swagger_ui.png)

#### Sample Prediction (Camera Simulation)

![Prediction result](output/predict_output.png)

---
