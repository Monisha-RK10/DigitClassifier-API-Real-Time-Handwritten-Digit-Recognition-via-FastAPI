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
