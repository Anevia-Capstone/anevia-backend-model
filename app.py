from fastapi import FastAPI, UploadFile, File, Response
import uvicorn
import shutil
import os
from Run_detection_model.run_detection_model import predict_image, load_detection_model
from Run_cropping_model.run_cropping_model import crop_conjunctiva, load_cropping_model
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

# Load models
CROP_MODEL_PATH = "Run_cropping_model/best.pt"
DETECTION_MODEL_PATH = "Run_detection_model/model.h5"

# Check if models exist, otherwise remind user to download
if not os.path.exists(CROP_MODEL_PATH):
    print(f"Error: Cropping model not found at {CROP_MODEL_PATH}")
    print("Please check download_model.txt for instructions on how to download the models.")
    exit()

if not os.path.exists(DETECTION_MODEL_PATH):
    print(f"Error: Detection model not found at {DETECTION_MODEL_PATH}")
    print("Please check download_model.txt for instructions on how to download the models.")
    exit()

load_cropping_model(CROP_MODEL_PATH)
load_detection_model(DETECTION_MODEL_PATH)

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
