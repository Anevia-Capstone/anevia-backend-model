from fastapi import FastAPI, UploadFile, File, Response
import uvicorn
import shutil
import os
from Run_detection_model.run_detection_model import predict_single_image
from Run_cropping_model.run_cropping_model import infer_single_image
from typing import Optional
import tempfile
import cv2
import numpy as np
import io
import requests
from pathlib import Path

def limit_filename_length(filename: str, max_length: int = 8) -> str:
    """Limit the base filename to specified length while preserving extension"""
    base_name = Path(filename).stem
    file_ext = Path(filename).suffix
    # If filename is longer than max_length, truncate it
    if len(base_name) > max_length:
        base_name = base_name[-max_length:]
    return f"{base_name}{file_ext}"

app = FastAPI(
    title="Anemia Detection API",
    description="API for detecting anemia from conjunctiva images",
    version="1.0.0"
)

# Model paths
DETECTION_MODEL_PATH = "Run_detection_model/model.h5"
SEGMENTATION_MODEL_PATH = "Run_cropping_model/ModelSegmentasi.pt"

# Check if models exist
if not os.path.exists(DETECTION_MODEL_PATH):
    print(f"Error: Detection model not found at {DETECTION_MODEL_PATH}")
    print("Please check download_model.txt for instructions on how to download the models.")
    exit()

if not os.path.exists(SEGMENTATION_MODEL_PATH):
    print(f"Error: Segmentation model not found at {SEGMENTATION_MODEL_PATH}")
    print("Please check download_model.txt for instructions on how to download the models.")
    exit()

INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Pastikan direktori input dan output ada
for dir_path in [INPUT_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def format_detection_response(prediction_result):
    """
    Convert the output from predict_single_image to the required API format
    
    Input format from predict_single_image:
    {
        'filename': str,
        'anemic_probability': float,
        'non_anemic_probability': float,
        'predicted_class': str,
        'confidence': float
    }
    
    Output format:
    {
        "detection": "Anemic" | "Non-Anemic",
        "confidence": {
            "Anemic": number,
            "Non-Anemic": number
        }
    }
    """
    if 'error' in prediction_result:
        return {
            "error": prediction_result['error'],
            "detection": None,
            "confidence": {
                "Anemic": 0.0,
                "Non-Anemic": 0.0
            }
        }
    
    return {
        "detection": prediction_result['predicted_class'],
        "confidence": {
            "Anemic": prediction_result['anemic_probability'],
            "Non-Anemic": prediction_result['non_anemic_probability']
        }
    }

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """
    Predict if an image shows anemic or non-anemic conjunctiva
    Returns: {
        "detection": "Anemic" | "Non-Anemic",
        "confidence": {
            "Anemic": number,
            "Non-Anemic": number
        }
    }
    """
    input_filename = None
    try:
        # Limit filename length and create filename with _detect suffix
        limited_filename = limit_filename_length(file.filename)
        base_name = Path(limited_filename).stem
        file_ext = Path(limited_filename).suffix
        detect_filename = f"{base_name}_detect{file_ext}"
        
        # Save uploaded file to input directory
        input_filename = os.path.join(INPUT_DIR, detect_filename)
        with open(input_filename, "wb") as input_file:
            shutil.copyfileobj(file.file, input_file)

        # Get prediction using the existing predict_single_image function
        print(f"\nProcessing detection request for file: {file.filename}")
        raw_result = predict_single_image(input_filename)
        
        # Format the response to match the required API format
        formatted_result = format_detection_response(raw_result)
        
        if 'error' not in formatted_result:
            print(f"Detection completed successfully: {formatted_result['detection']} (Anemic: {formatted_result['confidence']['Anemic']:.4f}, Non-Anemic: {formatted_result['confidence']['Non-Anemic']:.4f})")
        else:
            print(f"Detection failed: {formatted_result['error']}")

        return formatted_result
        
    except Exception as e:
        print(f"Error in /detect endpoint: {str(e)}")
        return {
            "error": str(e),
            "detection": None,
            "confidence": {
                "Anemic": 0.0,
                "Non-Anemic": 0.0
            }
        }
    finally:
        # Cleanup: Remove input file after processing
        if input_filename and os.path.exists(input_filename):
            try:
                os.remove(input_filename)
                print(f"Cleaned up input file: {input_filename}")
            except Exception as e:
                print(f"Error cleaning up input file: {str(e)}")

@app.post("/crop/")
async def crop(file: UploadFile = File(...)):
    """
    Process an image and return the segmented (cropped) version with transparency
    """
    input_filename = None
    output_files = []
    
    try:
        # Limit filename length and create filename with _crop suffix
        limited_filename = limit_filename_length(file.filename)
        base_name = Path(limited_filename).stem
        file_ext = Path(limited_filename).suffix
        crop_filename = f"{base_name}_crop{file_ext}"
        
        # Save uploaded file to input directory
        input_filename = os.path.join(INPUT_DIR, crop_filename)
        with open(input_filename, "wb") as input_file:
            shutil.copyfileobj(file.file, input_file)

        # Run the segmentation model
        infer_single_image(input_filename, SEGMENTATION_MODEL_PATH)
        
        # Move generated files to output directory
        base_name = os.path.splitext(limited_filename)[0]
        suffixes = ["_segmented512.png", "_mask512.jpg", "_overlay512.jpg"]
        
        for suffix in suffixes:
            src_path = input_filename.replace(".jpg", suffix)
            dst_path = os.path.join(OUTPUT_DIR, f"{base_name}{suffix}")
            if os.path.exists(src_path):
                # Use copy instead of move to keep original files
                shutil.copy2(src_path, dst_path)
                os.remove(src_path)  # Remove the file from input directory
                output_files.append(dst_path)
        
        # Read the segmented image from output directory
        segmented_path = os.path.join(OUTPUT_DIR, f"{base_name}_segmented512.png")
        with open(segmented_path, "rb") as f:
            image_bytes = f.read()
        
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e
    finally:
        # Cleanup: Remove input and output files after processing
        if input_filename and os.path.exists(input_filename):
            try:
                os.remove(input_filename)
                print(f"Cleaned up input file: {input_filename}")
            except Exception as e:
                print(f"Error cleaning up input file: {str(e)}")
        
        # Clean up output files
        for output_file in output_files:
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    print(f"Cleaned up output file: {output_file}")
                except Exception as e:
                    print(f"Error cleaning up output file {output_file}: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anemia Detection API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Anemia Detection API is running",
        "endpoints": ["/detect/", "/crop/", "/health"]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)