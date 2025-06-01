import mlflow

logged_model_uri = "models:/KidneyDiseaseVGG16/2"  # Example URI

# Set MLflow tracking URI to your DagsHub repo
mlflow.set_tracking_uri("https://dagshub.com/jadhavgaurav/Kidney_disease_classification_cnn.mlflow")

model = mlflow.keras.load_model(logged_model_uri)

# Test prediction on dummy image
# prediction = model.predict(preprocessed_image)
