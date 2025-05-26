import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# establish available classes for prediction and image size for input shape
class_names = ["Anemic", "Non-Anemic"]
img_size = (224, 224)
model = load_model("/path/to/model.h5") # adjust path based on model location

# Process the input image prior to model inference
def load_and_preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0), img

input_img, display_img = load_and_preprocess_image("/path/to/image.jpeg") # adjust path based on image location
prediction = model.predict(input_img) # output prediction in percentage; ex. [[0.37858227 0.57706666]] : first index is Anemic, second index is Non-Anemic
pred_class = class_names[np.argmax(prediction)] # classify based on previous prediction values; ex. [[0.37858227 0.57706666]] will classify as Non-Anemic

# OPTIONAL : Display output image and it's predicted value
plt.subplot(1, 1, 1)
plt.imshow(display_img)
plt.title(f"Predicted: {pred_class}")
plt.axis("off")

plt.tight_layout()
plt.show()