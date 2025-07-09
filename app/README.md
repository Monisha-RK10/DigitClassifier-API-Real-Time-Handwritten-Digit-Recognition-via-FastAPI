# app/

## Files

### `main.py` (**FastAPI Application and Routing**)

This is the main application file that:
- Instantiates a FastAPI app instance to serve the API.
- Defines two HTTP endpoints:
  - `GET /health`
    - Returns a simple JSON status `{ "status": "ok" }` to indicate the server is running.
  - `POST /predict`
    - Accepts image input for digit prediction. It supports:
      - Uploading an image file via multipart/form-data.
      - Using a simulated camera image by passing the query parameter `?use_camera=true`.
- Handles image processing by converting input images (uploaded or from camera) to RGB format, normalizing channels for consistent inference.
- Implements error handling:
  - Returns HTTP 400 if no image is provided either via file upload or camera flag.
  - Returns HTTP 500 for internal server errors.
  - Responds with JSON containing the predicted digit on successful inference.

---

### `camera.py` (**Virtual Camera Image Capture**)

This module simulates a camera feed for testing purposes by:
- Accessing the `app/mock_camera_feed` directory, which contains sample digit images like digit 2 (`camera_digit_2_1.png`).
- Randomly selecting one image from this folder.
- Opening the selected image as a PIL image.
- Returning this image for inference, allowing the API to simulate capturing an image from a camera without requiring actual hardware.
---

### `model.py` (**Model Training** & **Model Architecture Details**)

**Model Training**

The code in `app/model.py` performs the following:
- Downloads and preprocesses the MNIST dataset:
  - Converts images to tensors
  - Normalizes pixel values
  - Splits data into training, validation, and test sets
  - Creates data loaders with shuffling for training
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

> Note:
>
> - **Transforms**: Normalization and conversion to tensors ensures consistent input scaling.
> -  **Dropout**: Reduces overfitting by randomly disabling neurons during training.
> -  **LogSoftmax**: Outputs log-probabilities to improve numerical stability.
> -  **Early Stopping**: Stops training if validation accuracy plateaus to save compute and prevent overfitting.
> -  **Model Performance**: Val set: `Avg loss: 0.0337`, `Accuracy: 4952/5000 (99.04%)`
---

### `predict.py` (**Model Loading and Inference**)

This module handles:
- Loading the trained CNN model weights from `app/model.py` saved during training.
- Using the preprocessing utilities from `app/utils.py` to prepare incoming images.
- Running a forward pass on the preprocessed image tensor through the model in evaluation mode `(torch.no_grad())`, to efficiently compute predictions.
- Extracting the predicted digit by selecting the class with the highest output probability.

> This module provides the main function `predict_digit(image)` that takes a PIL image as input and returns the predicted digit as an integer, which is then used by the API for digit classification.

---

### `utils.py` (**Image Preprocessing for MNIST**)

This module prepares input images to be compatible with the MNIST-trained CNN model by performing the following steps:
- **Grayscale Conversion**: Converts input images to grayscale, as MNIST digits are single-channel.
- **Resize**: Resizes images to the standard MNIST dimension of 28×28 pixels.
- **ToTensor**: Converts the PIL grayscale image to a PyTorch tensor with shape `[1, 28, 28]`. This also scales pixel values from `[0, 255]` to `[0.0, 1.0]`.
- **Normalize**: Applies normalization using MNIST mean `(0.1307)` and standard deviation `(0.3081)`, matching the model’s training conditions.
- **Batch Dimension**: Adds a batch dimension to the tensor to form `[1, 1, 28, 28]` which is required by the PyTorch model.

> This ensures that any input image matches the format and distribution expected by the CNN model during inference.

---

### `mnist_cnn.pth` (**Model Weights**)

- The file `app/mnist_cnn.pth` contains the trained weights of the CNN model for MNIST digit classification.
- This file is generated after training the model on the MNIST dataset.
- It is loaded by the application during inference to make predictions.
- If you want to retrain the model, you can run the training script (if provided) to generate a new weights file.

> Note: Make sure this file is included in your repository or accessible at the correct path when deploying or running the app.
