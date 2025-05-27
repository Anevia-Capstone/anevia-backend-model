# Anemia Detection API

API untuk deteksi dini anemia berdasarkan citra digital conjunctiva mata menggunakan machine learning dan computer vision.

## Deskripsi

Aplikasi ini adalah REST API yang dapat:
1. **Mendeteksi anemia** dari gambar conjunctiva mata menggunakan model deep learning
2. **Memotong dan mensegmentasi** area conjunctiva dari gambar mata menggunakan YOLO model

Sistem ini terdiri dari dua model utama:
- **Detection Model**: Model TensorFlow untuk klasifikasi anemia/non-anemia
- **Cropping Model**: Model YOLO untuk segmentasi conjunctiva

## Fitur

- ✅ **Deteksi Anemia**: Klasifikasi gambar conjunctiva sebagai anemia atau non-anemia
- ✅ **Segmentasi Conjunctiva**: Otomatis memotong area conjunctiva dari gambar mata
- ✅ **REST API**: Interface yang mudah digunakan untuk integrasi
- ✅ **Confidence Score**: Memberikan tingkat kepercayaan prediksi
- ✅ **Format Response JSON**: Output terstruktur dan mudah diparse

## Teknologi

- **FastAPI**: Web framework untuk API
- **TensorFlow**: Deep learning framework untuk model deteksi
- **Ultralytics YOLO**: Computer vision untuk segmentasi
- **OpenCV**: Image processing
- **Uvicorn**: ASGI server

## Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd Capstone-Project-Early-Detection-of-Anemia-Based-on-Digital-Imagery
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:8000`

## API Endpoints

### 1. Health Check
```
GET /
```
**Response:**
```json
{
  "message": "Anemia Detection API is running"
}
```

### 2. Deteksi Anemia
```
POST /detect/
```
**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (gambar conjunctiva)

**Response:**
```json
{
  "detection": "Anemic",
  "confidence": {
    "Anemic": 0.85,
    "Non-Anemic": 0.15
  }
}
```

### 3. Crop Conjunctiva
```
POST /crop/
```
**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (gambar mata)

**Response:**
- Content-Type: image/png
- Body: Gambar conjunctiva yang sudah di-crop dan di-segment

## Cara Penggunaan

### Menggunakan cURL

**1. Test Health Check:**
```bash
curl http://localhost:8000/
```

**2. Deteksi Anemia:**
```bash
curl -X POST "http://localhost:8000/detect/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

**3. Crop Conjunctiva:**
```bash
curl -X POST "http://localhost:8000/crop/" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg" \
     --output cropped_conjunctiva.png
```

### Menggunakan Python

```python
import requests

# Deteksi anemia
with open('conjunctiva_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect/', files=files)
    result = response.json()
    print(f"Detection: {result['detection']}")
    print(f"Confidence: {result['confidence']}")

# Crop conjunctiva
with open('eye_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/crop/', files=files)
    with open('cropped_result.png', 'wb') as output:
        output.write(response.content)
```

## Struktur Project

```
├── app.py                          # Main FastAPI application
├── requirements.txt                # Python dependencies
├── README.md                      # Documentation
├── Run_detection_model/           # Detection model files
│   ├── model.h5                   # TensorFlow model
│   ├── run_detection_model.py     # Detection logic
│   └── README.md                  # Model documentation
└── Run_cropping_model/            # Cropping model files
    ├── best.pt                    # YOLO model weights
    └── run_cropping_model.py      # Cropping logic
```

## Model Information

### Detection Model
- **Type**: TensorFlow/Keras CNN
- **Input**: 224x224 RGB images
- **Output**: Binary classification (Anemic/Non-Anemic)
- **Classes**: ["Anemic", "Non-Anemic"]

### Cropping Model
- **Type**: YOLOv8 Segmentation
- **Input**: Variable size images
- **Output**: Segmentation masks for conjunctiva
- **Classes**: ["conjunctiva"]

## Error Handling

- **404**: Conjunctiva tidak ditemukan dalam gambar
- **500**: Gagal memproses gambar
- **422**: Format file tidak valid

## Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

## Tim Pengembang

Capstone Project - Early Detection of Anemia Based on Digital Imagery
DBS Foundation - Semester 6

## Dokumentasi API

Setelah menjalankan aplikasi, dokumentasi interaktif tersedia di:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
