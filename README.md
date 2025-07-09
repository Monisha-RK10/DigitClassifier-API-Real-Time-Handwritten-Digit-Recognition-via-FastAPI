# Image_Classification_using_CNN_on_MNIST (Python & FastAPI)

### Description of the use case
### Dataset source
### Steps to run the API (uvicorn, etc.)
### Example API call
### How to handle real-time camera input in the production

### Code structure for this project is as follows
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
├── README.md
├── Dockerfile
├── .dockerignore

```
