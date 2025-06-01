import os
import mlflow.keras
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model from DagsHub (MLflow tracking URI)
mlflow.set_tracking_uri("https://dagshub.com/jadhavgaurav/Kidney_disease_classification_cnn.mlflow")
model_name = "KidneyDiseaseVGG16"
model_version = "1"
model = mlflow.keras.load_model(f"models:/{model_name}/{model_version}")

# Labels for prediction
class_labels = ["Cyst", "Normal", "Stone", "Tumor"]

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # VGG16 input size
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img_tensor = preprocess_image(file_path)
    predictions = model.predict(img_tensor)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return render_template('result.html', 
                           image_path=file_path, 
                           predicted_class=predicted_class, 
                           confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
