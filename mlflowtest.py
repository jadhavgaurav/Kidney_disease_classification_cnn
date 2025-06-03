import mlflow
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# === Set MLflow Tracking URI to DagsHub ===
mlflow.set_tracking_uri("https://dagshub.com/jadhavgaurav/Kidney_disease_classification_cnn.mlflow")

# === Load model from DagsHub-registered MLflow model ===
logged_model_uri = "models:/KidneyDiseaseVGG16/2"
model = mlflow.keras.load_model(logged_model_uri)

# === Define class labels (must match training labels order) ===
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']

# === Image Preprocessing ===
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# === Test image path (replace with your own test image) ===
test_img_path = 'data/kidney_split/val/Tumor/Tumor- (16).jpg'  

# === Predict ===
img = preprocess_image(test_img_path)
preds = model.predict(img)[0]  # Assuming model.predict returns shape (1, 4)
class_index = np.argmax(preds)
class_label = CLASS_NAMES[class_index]
confidence = preds[class_index]

# === Output Result ===
print(f"Predicted class: {class_label}")
print("All class probabilities:")
for i, prob in enumerate(preds):
    print(f"{CLASS_NAMES[i]}: {prob:.4f}")
