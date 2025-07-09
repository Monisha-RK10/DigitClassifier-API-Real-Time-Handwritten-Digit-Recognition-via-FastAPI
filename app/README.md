# app/

## Files

### main.py
 **Image acquisition can happen from**:
| Mode              | Client Sends        | Backend Handles With              | Notes                                |
| ----------------- | ------------------- | --------------------------------- | ------------------------------------ |
| Image Upload      | Attached image file | `UploadFile = File(...)`          | Common for Postman, frontends        |
| Use Camera (flag) | No image, only flag | `use_camera: bool = Query(False)` | Load simulated image from local path |

- **Image Upload (from disk)**
  - `Static input (uploaded file)`: e.g., from a UI or API call
  - Client sends a POST request to /predict with an attached image file
  - In FastAPI route, we will receive it as  `async def predict(file: UploadFile = File(...))`
  -  Typical behavior for frontend upload buttons or Postman tests
  -  No need for camera logic here as the image is already attached by the client

- **Camera Flag (simulate a camera snapshot)**
  - `Dynamic capture (camera feed)`: e.g., from an actual industrial camera in production (like IDS, Ximea, etc.)
  - Client sends a POST request to /predict?use_camera=true
  - In this case, there is no image file attached.
  - The backend should:
    - Detect that `use_camera == true`
    - Internally load an image from disk that simulates a "camera" (e.g., camera_image.png)

**In a real-world setup:**
- A USB webcam or industrial camera (e.g., IDS, Ximea) would be interfaced using SDKs like OpenCV (cv2.VideoCapture), camera-specific Python APIs (e.g., pyueye, ximea-python), or GStreamer pipelines.
- The application would continuously acquire frames via a loop (cap.read() or async frame grabbing), and those frames would be passed into the inference pipeline.
- In production, proper error handling, timeouts, and frame buffering would be used to ensure robustness.

---

### camera.py 

---

### model.py 

- Downloads and preprocesses the MNIST dataset:
  - Converts images to tensors
  - Normalizes pixel values
  - Splits data into training, validation, and test sets
- Defines a simple CNN architecture with:
  - Two convolutional layers + ReLU activations
  - Max pooling layer for downsampling
  - Dropout layers to prevent overfitting (25% after conv, 50% after fully connected)
  - Fully connected layers
  - LogSoftmax output for classification probabilities
- Trains the CNN model with validation monitoring and early stopping to avoid overfitting
- Saves the trained model weights (.pth) for later use in inference

**Model Architecture Details**
| Layer                  | Description                          | Output Shape |
| ---------------------- | ------------------------------------ | ------------ |
| Input                  | Grayscale image (1 channel)          | (1, 28, 28)  |
| Conv1 + ReLU           | 32 filters, kernel 3x3               | (32, 26, 26) |
| Conv2 + ReLU           | 64 filters, kernel 3x3               | (64, 24, 24) |
| MaxPooling             | 2x2 pooling                          | (64, 12, 12) |
| Dropout (Conv2D)       | 25% dropout rate                     | (64, 12, 12) |
| Flatten                | Flattens to 9216 features            | (9216,)      |
| Fully Connected + ReLU | 128 neurons                          | (128,)       |
| Dropout (FC)           | 50% dropout rate                     | (128,)       |
| Output Layer           | 10 neurons (digits 0-9) + LogSoftmax | (10,)        |


Creates data loaders with shuffling for training
---

### predict.py

---

### utils.py 

---
### mnist_cnn.pth

---

### mock_camera_feed
```bash
55,000 samples
↓  split into
859 batches (each batch = 64 samples)
↓
for each batch:
    → forward pass
    → compute loss
    → backprop
    → optimizer step
```
