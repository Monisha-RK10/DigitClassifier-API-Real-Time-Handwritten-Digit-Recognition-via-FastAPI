# Manual test script for FastAPI endpoints using TestClient.
# This script simulates requests to the running app (without needing a real server) and checks the /health and /predict endpoints.

from fastapi.testclient import TestClient                                              # TestClient: Simulates requests to FastAPI app without running a server
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    print(" /health:", response.status_code, response.json())                          #  /health: 200 {'status': 'ok'}
    assert response.status_code == 200                                                 # Checks if the response status code is 200
    assert response.json() == {"status": "ok"}                                         # Checks if the JSON response exactly matches {"status": "ok"}

def test_predict_dummy_image():
    import io
    from PIL import Image

    # Create dummy image
    image = Image.new("L", (28, 28), color=0)                                           # Grayscale blank image

    # For testing, upload is simulated without writing any file to disk by creating the file in memory
    # Perform the below steps to pass it to the test client exactly like a real uploaded file
    buf = io.BytesIO()                                                                  # Creates an in-memory binary stream 'buf' (a file-like object). /predict expects an uploaded file
    image.save(buf, format="PNG")                                                       # Saves the PIL image to this buffer in PNG format
    buf.seek(0)                                                                         # Moves the read/write pointer back to the beginning of the in-memory file (after saving, the pointer is at the end of the stream)

    response = client.post("/predict", files={"file": ("test.png", buf, "image/png")})  # test.png: Filename to simulate uploaded file, buf: actual image data, "image/png": MIME (Multipurpose Internet Mail Extensions)
    print(" /predict:", response.status_code, response.json())                          # /predict: 200 {'prediction': 1}
    assert response.status_code == 200
    assert "prediction" in response.json()

test_health()
test_predict_dummy_image()
