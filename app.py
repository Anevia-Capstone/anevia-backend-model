from fastapi import FastAPI, UploadFile, File, Response
import uvicorn
import shutil
import os
from Run_detection_model.run_detection_model import predict_image
from Run_cropping_model.run_cropping_model import crop_conjunctiva
from typing import Optional
import tempfile
import cv2
import numpy as np
import io
import requests
from pathlib import Path

app = FastAPI(
    title="Anemia Detection API",
    description="API for detecting anemia from conjunctiva images",
    version="1.0.0"
)

def download_model_from_gdrive(file_id: str, destination: str):
    """Download model from Google Drive"""
    if os.path.exists(destination):
        print(f"Model already exists at {destination}")
        return

    print(f"Downloading model to {destination}...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"Model downloaded successfully to {destination}")

# Model file IDs from Google Drive (you'll need to replace these)
CROP_MODEL_GDRIVE_ID = "YOUR_CROP_MODEL_FILE_ID"  # Replace with actual file ID
DETECTION_MODEL_GDRIVE_ID = "YOUR_DETECTION_MODEL_FILE_ID"  # Replace with actual file ID

# Download models if they don't exist
download_model_from_gdrive(CROP_MODEL_GDRIVE_ID, "Run_cropping_model/best.pt")
download_model_from_gdrive(DETECTION_MODEL_GDRIVE_ID, "Run_detection_model/model.h5")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """
    Predict if an image shows anemic or non-anemic conjunctiva
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    # Get prediction
    result = predict_image(temp_path)

    # Clean up
    os.unlink(temp_path)

    return result

@app.post("/crop/")
async def crop_conjunctiva_endpoint(file: UploadFile = File(...)):
    """
    Crop and segment conjunctiva from an image
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # Use the proper cropping function
        success, cropped_img, message = crop_conjunctiva(temp_path, "Run_cropping_model/best.pt")

        # Clean up
        os.unlink(temp_path)

        if not success:
            return Response(content=message, status_code=404)

        # Convert image to bytes
        is_success, buffer = cv2.imencode(".png", cropped_img)
        if not is_success:
            return Response(content="Failed to encode image", status_code=500)

        # Return the image directly
        return Response(content=buffer.tobytes(), media_type="image/png")

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return Response(content=f"Error processing image: {str(e)}", status_code=500)

@app.get("/")
async def root():
    return {"message": "Anemia Detection API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)


