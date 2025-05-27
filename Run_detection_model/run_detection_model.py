import json
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# establish available classes for prediction and image size for input shape
class_names = ["Anemic", "Non-Anemic"]
img_size = (224, 224)
# Use absolute path for model loading to ensure it works from any directory
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")
detection_model = load_model(model_path)

# Process the input image prior to model inference
def load_and_preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image at path: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0), img

def predict_image(image_path):
    input_img, _ = load_and_preprocess_image(image_path)
    prediction = detection_model.predict(input_img)
    pred_class = class_names[np.argmax(prediction)]
    
    # Create JSON output
    result = {
        "detection": pred_class,
        "confidence": {
            "Anemic": float(prediction[0][0]),
            "Non-Anemic": float(prediction[0][1])
        }
    }
    
    return result

if __name__ == "__main__":
    # Check if image path is provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image path if none provided
        image_path = "WhatsApp Image 2025-05-26 at 15.17.28_471c9e71.jpg"
    
    result = predict_image(image_path)
    print(json.dumps(result, indent=2))
