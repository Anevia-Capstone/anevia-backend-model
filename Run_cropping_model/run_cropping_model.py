import torch
import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn

IMAGE_PATH = "contoh_gambar2.jpg" 
MODEL_PATH = "ModelSegmentasi.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
def infer_single_image(image_path, model_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    original = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = transform(image=image_rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    # Load TorchScript model
    model = torch.jit.load(model_path, map_location=DEVICE).to(DEVICE)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        mask_pred = torch.sigmoid(output).squeeze().cpu().numpy()

    mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_binary, (original.shape[1], original.shape[0]))

    overlay = original.copy()
    overlay[mask_resized > 0] = [0, 255, 0]
    blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

    # Modifikasi bagian pembuatan segmented image
    segmented_rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
    # Buat alpha channel berdasarkan mask
    segmented_rgba[:, :, 3] = mask_resized  # Gunakan mask sebagai alpha channel
    
    # Save results
    mask_out_path = image_path.replace(".jpg", "_mask512.jpg")
    overlay_out_path = image_path.replace(".jpg", "_overlay512.jpg")
    segmented_out_path = image_path.replace(".jpg", "_segmented512.png")

    cv2.imwrite(mask_out_path, mask_resized)
    cv2.imwrite(overlay_out_path, blended)
    cv2.imwrite(segmented_out_path, segmented_rgba)
    
    print(f"Hasil segmented transparan: {segmented_out_path}")
    print(f"Hasil mask: {mask_out_path}")
    print(f"Hasil overlay: {overlay_out_path}")

if __name__ == "__main__":
    infer_single_image(IMAGE_PATH, MODEL_PATH)

