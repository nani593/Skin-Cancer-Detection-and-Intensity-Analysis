from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
app = Flask(__name__)

# Load the pre-trained DenseNetFull model
model = load_model('DenseNetFull.h5')

# Define class names based on the dataset
class_names = [
    'Actinic_keratoses', 'Basal_cell_carcinoma', 'Benign_keratosis-like_lesions',
    'Dermatofibroma', 'Melanocytic_nevi', 'Vascular_lesions', 'Melanoma']

# Define a function to preprocess an input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(192, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Save the file to the 'static' folder
        file_path = os.path.join('static', 'uploaded_image.jpg')
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        # Predict intensity based on the tumor class
        if predicted_class_name == 'Melanoma':
            tumor = 'malignant (Cancerous)'
            intensity = 'critical'
        elif predicted_class_name in ['Basal_cell_carcinoma', 'Actinic_keratoses']:
            tumor = 'malignant (Cancerous)'
            intensity = 'Severe'
        elif predicted_class_name in ['Melanocytic_nevi', 'Benign_keratosis-like_lesions', 'Vascular_lesions', 'Dermatofibroma']:
            tumor = 'benign (Non-Cancerous)'
            intensity = 'Mild'
        
        # Create a dictionary with prediction details and image path
        prediction_result = {'class': predicted_class_name, 'tumor': tumor, 'intensity': intensity, 'image_path': file_path}

        # Render the template with the prediction result
        return render_template('index.html', prediction=prediction_result)
        

if __name__ == '__main__':
    app.run(debug=True)
