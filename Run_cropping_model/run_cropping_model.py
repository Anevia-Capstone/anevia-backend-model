import cv2
import numpy as np
from ultralytics import YOLO
import os

# Global variable for the cropping model
cropping_model = None

def load_cropping_model(model_path_arg: str):
    """Load the cropping model from the specified path."""
    global cropping_model
    if cropping_model is None:
        print(f"Loading cropping model from {model_path_arg}...")
        cropping_model = YOLO(model_path_arg)
        print("Cropping model loaded.")

def crop_conjunctiva(image_path: str, model_path: str = "best.pt"):
    """
    Crop and segment conjunctiva from an image using YOLO model

    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the YOLO model file

    Returns:
        tuple: (success: bool, cropped_image: np.ndarray or None, message: str)
    """
    try:
        if cropping_model is None:
            raise RuntimeError("Cropping model not loaded. Call load_cropping_model first.")

        # Inference
        results = cropping_model(image_path, save=False, verbose=False)[0]

        # Read original image
        img = cv2.imread(image_path)
        if img is None:
            return False, None, "Failed to read image"

        # Process results
        found = False
        cropped_img = None

        for j, mask in enumerate(results.masks.data):
            class_id = int(results.boxes.cls[j].item())
            if model.names[class_id] != 'conjunctiva':
                continue

            found = True
            binary_mask = mask.cpu().numpy().astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(binary_mask)
            img_crop = img[y:y+h, x:x+w]
            mask_crop = binary_mask[y:y+h, x:x+w]
            segmented = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            cropped_img = segmented
            break  # Use the first conjunctiva found

        if not found:
            return False, None, "No conjunctiva found in the image"

        return True, cropped_img, "Conjunctiva successfully cropped"

    except Exception as e:
        return False, None, f"Error processing image: {str(e)}"

def crop_conjunctiva_from_array(image_array: np.ndarray, model_path: str = "best.pt"):
    """
    Crop and segment conjunctiva from an image array using YOLO model

    Args:
        image_array (np.ndarray): Input image as numpy array
        model_path (str): Path to the YOLO model file

    Returns:
        tuple: (success: bool, cropped_image: np.ndarray or None, message: str)
    """
    try:
        if cropping_model is None:
            raise RuntimeError("Cropping model not loaded. Call load_cropping_model first.")

        # Inference directly on array
        results = cropping_model(image_array, save=False, verbose=False)[0]

        # Process results
        found = False
        cropped_img = None

        for j, mask in enumerate(results.masks.data):
            class_id = int(results.boxes.cls[j].item())
            if model.names[class_id] != 'conjunctiva':
                continue

            found = True
            binary_mask = mask.cpu().numpy().astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(binary_mask)
            img_crop = image_array[y:y+h, x:x+w]
            mask_crop = binary_mask[y:y+h, x:x+w]
            segmented = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
            cropped_img = segmented
            break  # Use the first conjunctiva found

        if not found:
            return False, None, "No conjunctiva found in the image"

        return True, cropped_img, "Conjunctiva successfully cropped"

    except Exception as e:
        return False, None, f"Error processing image: {str(e)}"

# For backward compatibility and standalone usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tkinter import Tk, filedialog

    # For standalone usage, load the model
    model_path_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    load_cropping_model(model_path_local)

    # Open file dialog to select image
    Tk().withdraw()  # Hide main Tkinter window
    img_path = filedialog.askopenfilename(title="Pilih Gambar")

    if img_path:
        success, cropped_img, message = crop_conjunctiva(img_path, model_path_local)

        if success:
            # Save result
            filename = "hasil_conjunctiva_cropped.png"
            cv2.imwrite(filename, cropped_img)
            print(f"Result saved as {filename}")

            # Display result
            plt.figure()
            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.title("Hasil Segmentasi Konjungtiva")
            plt.axis('off')
            plt.show()
        else:
            print(f"Error: {message}")
    else:
        print("No image selected")
