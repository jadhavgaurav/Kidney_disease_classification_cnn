import os
import mlflow.pyfunc
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.keras
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Flask Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# DagsHub credentials
DAGSHUB_USERNAME = "jadhavgaurav"
DAGSHUB_REPO = "Kidney_disease_classification_cnn"
DAGSHUB_TOKEN = "b488473ae5ff04ec493007592680c395a4ff9160"

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

# MLflow Tracking Configuration (DagsHub)
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
# MODEL_URI = "models:/KidneyDiseaseVGG16/1"
# model = mlflow.pyfunc.load_model(MODEL_URI)

client = MlflowClient()
model_uri = client.get_model_version_download_uri("KidneyDiseaseVGG16", "5")

model = mlflow.keras.load_model(model_uri)

# Label Mapping
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Image Preprocessing Function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Home Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

           # Preprocess and Predict
            img = preprocess_image(filepath)
            preds = model.predict(img)[0]
            predicted_index = np.argmax(preds)
            predicted_class = CLASS_NAMES[predicted_index]

            # ✅ Keep confidence scores as floats (not strings with '%')
            confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

            return render_template('result.html',
                                filename=filename,
                                prediction=predicted_class,
                                confidences=confidences)

    return render_template('index.html')

# Image Serving Route
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename='uploads/' + filename)

# Run Server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
