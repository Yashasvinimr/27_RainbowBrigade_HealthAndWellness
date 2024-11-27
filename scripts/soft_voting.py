from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the models
model1 = load_model('D:/jyothika/Gradient/models/tooth_classifier_model.h5')
model2 = load_model('D:/jyothika/Gradient/models/tooth_classifier_with_closed_mouth.h5')
# Define image parameters
img_height, img_width = 150, 150

def classify_image(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions from both models
    pred1 = model1.predict(img_array)  # (1, 4)
    pred2 = model2.predict(img_array)  # (1, 5)

    # Ensure both predictions have the same shape
    if pred1.shape[1] != pred2.shape[1]:
        # If the models have different numbers of output classes, adjust
        min_classes = min(pred1.shape[1], pred2.shape[1])
        pred1 = pred1[:, :min_classes]  # Truncate to the same number of classes
        pred2 = pred2[:, :min_classes]

    # Average the predictions
    avg_pred = (pred1 + pred2) / 2

    # Map the predicted class (this may depend on your model's output format)
    class_indices = {0: 'Gingivitis', 1: 'Mouth Ulcer', 2: 'Tooth Discoloration', 3: 'Healthy Gum'}
    
    predicted_class = class_indices[np.argmax(avg_pred)]
    confidence = np.max(avg_pred)

    return predicted_class, confidence

# Test with an example image
image_path = input("enter path: ")
predicted_class, confidence = classify_image(image_path)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
