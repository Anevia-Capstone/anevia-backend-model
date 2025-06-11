import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from PIL import Image

# Establish available classes for prediction and image size for input shape
class_names = ["Anemic", "Non-Anemic"]
img_size = (224, 224)
model_path = os.path.join(os.path.dirname(__file__), "model.h5")
model = load_model(model_path)  # model in the same directory as this script

# Process single image
def load_and_preprocess_image(path):
    """Load and preprocess image for model inference"""
    with Image.open(path).convert("RGBA") as img:
        # Auto-crop using alpha channel (removes transparent padding)
        bbox = img.getbbox()
        img_cropped = img.crop(bbox) if bbox else img
        
        # Flatten onto white background & convert to RGB
        background = Image.new("RGB", img_cropped.size, (255, 255, 255))
        img_rgb = Image.alpha_composite(background.convert("RGBA"), img_cropped).convert("RGB")
        
        # Resize and normalize for model
        img_resized = img_rgb.resize(img_size)
        img_array = np.array(img_resized) / 255.0
        
        return np.expand_dims(img_array, axis=0)

def predict_single_image(image_path):
    """Predict anemia for a single image and return numerical results"""
    try:
        input_img = load_and_preprocess_image(image_path)
        prediction = model.predict(input_img, verbose=0)
        
        anemic_prob = prediction[0][0]
        non_anemic_prob = prediction[0][1]
        pred_class_idx = np.argmax(prediction)
        pred_class = class_names[pred_class_idx]
        confidence = prediction[0][pred_class_idx]
        
        return {
            'filename': os.path.basename(image_path),
            'anemic_probability': float(anemic_prob),
            'non_anemic_probability': float(non_anemic_prob),
            'predicted_class': pred_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        return {
            'filename': os.path.basename(image_path),
            'error': str(e)
        }

# Example usage for single image prediction
if __name__ == "__main__":
    image_path = "eye_image.png"  # example image path
    result = predict_single_image(image_path)
    
    print("Image Prediction Results:")
    print(f"File: {result['filename']}")
    if 'error' not in result:
        print(f"Anemic Probability: {result['anemic_probability']:.4f}")
        print(f"Non-Anemic Probability: {result['non_anemic_probability']:.4f}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {result['error']}")