# Model Deployment Guide

Since we removed the large model files from Git, here are several ways to get your models onto the server:

## Option 1: Google Drive (Recommended - Free & Easy)

### Step 1: Upload your models to Google Drive
1. Upload `best.pt` and `model.h5` to your Google Drive
2. Right-click each file → "Get link" → "Anyone with the link can view"
3. Copy the file ID from the URL (the long string between `/d/` and `/view`)

Example URL: `https://drive.google.com/file/d/1ABC123xyz789/view?usp=sharing`
File ID: `1ABC123xyz789`

### Step 2: Update the app.py file
Replace these lines in `app.py`:
```python
CROP_MODEL_GDRIVE_ID = "YOUR_CROP_MODEL_FILE_ID"  # Replace with actual file ID
DETECTION_MODEL_GDRIVE_ID = "YOUR_DETECTION_MODEL_FILE_ID"  # Replace with actual file ID
```

With your actual file IDs:
```python
CROP_MODEL_GDRIVE_ID = "1ABC123xyz789"  # Your best.pt file ID
DETECTION_MODEL_GDRIVE_ID = "1XYZ789abc123"  # Your model.h5 file ID
```

### Step 3: Deploy
The models will be automatically downloaded when your app starts for the first time.

## Option 2: Direct Server Upload

### Using SCP (if you have SSH access):
```bash
scp best.pt user@your-server:/path/to/app/Run_cropping_model/
scp model.h5 user@your-server:/path/to/app/Run_detection_model/
```

### Using FTP/SFTP:
Use FileZilla or similar FTP client to upload the files directly to the server.

## Option 3: Cloud Storage Services

### AWS S3:
```python
import boto3

def download_from_s3(bucket, key, destination):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, destination)
```

### Azure Blob Storage:
```python
from azure.storage.blob import BlobServiceClient

def download_from_azure(connection_string, container, blob_name, destination):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    with open(destination, "wb") as download_file:
        download_file.write(blob_service_client.get_blob_client(
            container=container, blob=blob_name).download_blob().readall())
```

## Option 4: Hugging Face Model Hub

Upload your models to Hugging Face and use:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="your-username/your-model", filename="best.pt")
```

## Current Setup

The app is already configured to download from Google Drive. You just need to:
1. Upload your models to Google Drive
2. Get the file IDs
3. Update the IDs in `app.py`
4. Deploy your app

The models will be downloaded automatically on first run and cached for subsequent runs.
