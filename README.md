
# Kidney Disease Classifier using VGG16-CNN

A full-stack end-to-end machine learning project for **Kidney Disease Classification** using deep learning and deployed on AWS EC2 using **Docker**, **Gunicorn**, and **Nginx**. The pipeline uses **MLflow**, **DVC**, **Google Cloud**, and **DagsHub** for experiment tracking and data versioning.

---

## 📂 Project Structure

```
Kidney_disease_classification_cnn/
├── data/                         # Raw and processed data
├── model/                        # Saved ML model artifacts
├── app/                          # Flask application
│   ├── static/                   # Static files (CSS, JS)
│   └── templates/                # HTML templates
├── Dockerfile
├── requirements.txt
├── dvc.yaml
├── app.py                        # Main Flask app
├── README.md
└── .dvc/                         # DVC config
```

---

## 📊 Dataset

- CT scan image dataset (classified as healthy or kidney disease)
- Images preprocessed and resized to match VGG16 input format (224x224)

---

## 🔍 Data Preprocessing

- Resize images
- Normalize pixel values
- Convert grayscale to RGB (if needed)
- Create labels from folder structure
- Train-test split (80-20)

---

## 🧠 Model Building (Transfer Learning - VGG16)

- Base Model: `VGG16(weights='imagenet', include_top=False)`
- Added custom dense layers for binary classification
- Frozen base layers
- Used `Adam` optimizer and `binary_crossentropy` loss

---

## ⚙️ Training and Logging (MLflow + DagsHub)

- Tracked hyperparameters, metrics, artifacts using MLflow
- MLflow tracking URI configured for **DagsHub**
- Model registered under `KidneyDiseaseVGG16`
- Sample logging snippet:

```python
import mlflow
import mlflow.keras

mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")
mlflow.keras.log_model(model, "vgg16_model", registered_model_name="KidneyDiseaseVGG16")
```

---

## 📦 Data Versioning (DVC + Google Cloud)

- `data/` folder versioned using DVC
- Remote storage configured to **Google Cloud Storage (GCS)**
- Commands used:

```bash
dvc init
dvc remote add -d gcsremote gcs://<your-gcs-bucket>
dvc add data/
dvc push
```

---

## 🌐 Flask Web App

- Built simple UI for image upload & prediction
- Uses the saved model for inference
- `app.py` handles routing and predictions

---

## 🐳 Dockerization

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Build & Push

```bash
docker build -t jadhavgaurav007/kidney_classifier .
docker push jadhavgaurav007/kidney_classifier
```

---

## ☁️ AWS EC2 Deployment

### 1. Launch Ubuntu EC2 Instance
- Open ports `80`, `443`, `5000`, `8888`

### 2. SSH into Instance

```bash
ssh -i <your-key.pem> ubuntu@<ec2-ip>
```

### 3. Install Docker

```bash
sudo apt update && sudo apt install docker.io -y
sudo usermod -aG docker ubuntu
newgrp docker
```

### 4. Pull and Run Container

```bash
docker pull jadhavgaurav007/kidney_classifier
docker run -d -p 5000:5000 jadhavgaurav007/kidney_classifier
```

---

## 🚀 Production Deployment (Gunicorn + Nginx)

### Install Nginx

```bash
sudo apt install nginx -y
```

### Configure Nginx

```nginx
server {
    listen 80;
    server_name ec2-<your-public-ip>.compute.amazonaws.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Enable Nginx

```bash
sudo systemctl restart nginx
sudo systemctl enable nginx
```

---

## ✅ Access App

http://ec2-51-21-160-142.eu-north-1.compute.amazonaws.com/

---


## 🙏 Acknowledgements

- DagsHub for experiment tracking
- GCP for cloud storage
- TensorFlow team for VGG16
- Docker & AWS for deployment
