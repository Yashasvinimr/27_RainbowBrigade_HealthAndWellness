from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the models
model1 = load_model('D:/jyothika/Gradient/models/tooth_classifier_model.h5')
model2 = load_model('D:/jyothika/Gradient/models/tooth_classifier_with_closed_mouth.h5')

# Image parameters
img_height, img_width = 150, 150

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to classify the image
def classify_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions from both models
    pred1 = model1.predict(img_array)  # (1, 4)
    pred2 = model2.predict(img_array)  # (1, 5)

    # Ensure both predictions have the same shape
    if pred1.shape[1] != pred2.shape[1]:
        min_classes = min(pred1.shape[1], pred2.shape[1])
        pred1 = pred1[:, :min_classes]  # Truncate to the same number of classes
        pred2 = pred2[:, :min_classes]

    # Average the predictions
    avg_pred = (pred1 + pred2) / 2

    # Map the predicted class
    class_indices = {0: 'Gingivitis', 1: 'Mouth Ulcer', 2: 'Tooth Discoloration', 3: 'Healthy Gum'}
    predicted_class = class_indices[np.argmax(avg_pred)]
    confidence = np.max(avg_pred)

    return predicted_class, confidence

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict the image class
        predicted_class, confidence = classify_image(filepath)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f'{confidence*100:.2f}%'
        })
    else:
        return jsonify({'error': 'Invalid file format'}), 400

# Helper function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# import os
# from flask import Flask, render_template, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# from werkzeug.utils import secure_filename
# import ray
# from ray import tune

# # Initialize Flask app
# app = Flask(__name__)

# # Model paths
# MODEL_PATH = 'models/tooth_classifier_model.h5'
# MODEL_PATH_CLOSED_MOUTH = 'models/tooth_classifier_with_closed_mouth.h5'

# # Hyperparameters for Ray Tune optimization
# HYPERPARAMS = {
#     'learning_rate': tune.uniform(1e-5, 1e-3),
#     'num_layers': tune.choice([2, 3, 4])
# }

# # Initialize models
# model1 = load_model(MODEL_PATH)
# model2 = load_model(MODEL_PATH_CLOSED_MOUTH)

# # Parameters
# img_height, img_width = 150, 150
# UPLOAD_FOLDER = 'uploads'  # Folder to save uploaded images
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # Configure upload folder
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Create uploads folder if it doesn't exist
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Function to classify image using the two models
# def classify_image(img_path):
#     # Load and preprocess the image
#     img = load_img(img_path, target_size=(img_height, img_width))
#     img_array = img_to_array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)

#     # Get predictions from both models
#     pred1 = model1.predict(img_array)  # (1, 4)
#     pred2 = model2.predict(img_array)  # (1, 5)

#     # Ensure both predictions have the same shape
#     if pred1.shape[1] != pred2.shape[1]:
#         min_classes = min(pred1.shape[1], pred2.shape[1])
#         pred1 = pred1[:, :min_classes]  # Truncate to the same number of classes
#         pred2 = pred2[:, :min_classes]

#     # Average the predictions
#     avg_pred = (pred1 + pred2) / 2

#     # Class indices for 4 possible outcomes (customize as per your needs)
#     class_indices = {0: 'Gingivitis', 1: 'Mouth Ulcer', 2: 'Tooth Discoloration', 3: 'Healthy Gum'}

#     predicted_class = class_indices[np.argmax(avg_pred)]
#     confidence = np.max(avg_pred)

#     return predicted_class, confidence

# # Helper function to check if file is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Route for home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle image upload and prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Predict the image class
#         predicted_class, confidence = classify_image(filepath)
        
#         return jsonify({
#             'predicted_class': predicted_class,
#             'confidence': f'{confidence*100:.2f}%'
#         })
#     else:
#         return jsonify({'error': 'Invalid file format'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
