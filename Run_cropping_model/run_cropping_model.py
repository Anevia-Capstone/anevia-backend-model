import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Buka file dialog untuk pilih gambar
Tk().withdraw()  # Sembunyikan jendela utama Tkinter
img_path = filedialog.askopenfilename(title="Pilih Gambar")

# Load model
model = YOLO("best.pt")  # ganti path model kamu

# Inference
results = model(img_path, save=False, verbose=False)[0]

# Baca gambar asli
img = cv2.imread(img_path)

found = False
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
    
    filename = f"hasil_conj_{j}.png"
    cv2.imwrite(filename, segmented)

    # Tampilkan hasil
    plt.figure()
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title(f"Hasil Segmentasi Konjungtiva {j+1}")
    plt.axis('off')

if not found:
    print("Tidak ditemukan kelas 'conjunctiva' pada gambar.")
else:
    plt.show()
