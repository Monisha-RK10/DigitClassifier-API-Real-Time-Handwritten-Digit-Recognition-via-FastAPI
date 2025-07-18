# app/main.py

# Step 5: FastAPI Application and Routing
# This code does the following:
# Instantiates the FastAPI app
# Creates two endpoints via decorators:
  # 1. @app.get("/health") -> Returns {"status": "ok"} to indicate server is running.
  # 2. @app.post("/predict") -> Handles prediction requests.
  # /predict accepts:
    # Either: a) an uploaded image via POST form-data (file=...)
    # Or: b) a simulated camera flag via query: /predict?use_camera=true
  # Image processing:
    # Uploaded or camera images are converted to RGB to normalize color channels.
    # This is because some uploads may be grayscale, RGBA, or CMYK, therefore, RGB standardizes this.
  # Error handling:
    # Returns HTTP 400 if neither file nor camera flag is provided (client error).
    # Returns HTTP 500 if something fails internally.
    # Returns JSON {"prediction": ...} if successful.

from fastapi import FastAPI, UploadFile, File, HTTPException, Query                               # 'FastAPI': Main app engine, 'UploadFile': Allows image file uploads, 'File': Marks the image as file input, 'HTTPException': Raises API errors, 'Query': To define URL query params like ?use_camera=true
from fastapi.responses import JSONResponse                                                        # Returns a custom JSON response with prediction to add clarity and control over format
from PIL import Image                                                                             # Processes image files
import io                                                                                         # Converts raw bytes from uploaded file into an image using BytesIO

from app.camera import capture_image_from_virtual_camera                                          # Handles mock camera input
from app.predict import predict_digit                                                             # Takes a PIL image and returns a digit prediction

app = FastAPI()                                                                                   # 'app' gets picked up while running uvicorn app.main:app

@app.get("/health")                                                                               # To verify if API is running
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(None), use_camera: bool = Query(False)):            # async def: Enables non-blocking, concurrent handling of I/O tasks. Provides either: a) file upload: POST form-data with file=... or b) camera flag: POST to /predict?use_camera=true
    try:
        if use_camera:
            image = capture_image_from_virtual_camera()
        elif file:
            image_data = await file.read()                                                        # await: Asynchronously reads uploaded image from client. file.read(): Gives binary image data
            image = Image.open(io.BytesIO(image_data)).convert("RGB")                             # Reads it as bytes and converts it to an image. Creates an in-memory file from bytes, RGB: To standardize
        else:
            raise HTTPException(status_code=400, detail="No image provided.")                     # 400 Bad Request (Client's side issue)

        prediction = predict_digit(image)
        return JSONResponse({"prediction": prediction})                                           # JSONResponse: Cleaner, controlled output -> good habit in APIs

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))                                       # 500 Internal Server Error

