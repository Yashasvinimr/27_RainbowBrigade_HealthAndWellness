import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
model_path = 'D:/jyothika/Gradient/models/tooth_classifier_with_closed_mouth.h5'
class_indices_path = 'D:/jyothika/Gradient/models/class_indices_with_closed_mouth.npy'
img_height, img_width = 150, 150

# Load Model and Class Indices
model = load_model(model_path)
class_indices = np.load(class_indices_path, allow_pickle=True).item()

# Reverse Class Indices Mapping
class_labels = {v: k for k, v in class_indices.items()}

# Function to Predict Class
def classify_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Handle Closed Mouth Case
    if predicted_class == 'Closed_Mouth':
        return "Uploaded image doesn't show teeth. Cannot detect.", confidence
    return predicted_class, confidence

# Test Classification
image_path = input("Enter the path of the image to classify: ")
predicted_class, confidence = classify_image(image_path)
print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
