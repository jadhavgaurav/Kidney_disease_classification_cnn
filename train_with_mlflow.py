import os
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# DagsHub credentials
DAGSHUB_USERNAME = "jadhavgaurav"
DAGSHUB_REPO = "Kidney_disease_classification_cnn"
DAGSHUB_TOKEN = "b488473ae5ff04ec493007592680c395a4ff9160"

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
TRAIN_DIR = "data/kidney_split/train"
VAL_DIR = "data/kidney_split/val"
mlf = "mlflowruns/V5/"

# MLflow experiment
mlflow.set_experiment("Kidney_Disease_Classification")

with mlflow.start_run(run_name=f"vgg16_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

    mlflow.set_tag("mlflow.note.content", "Training VGG16 model for Kidney Disease Classification with fine-tuning only last CONV layer i.e. block5_conv3.")

    # Log Params
    mlflow.log_params({
        "model": "VGG16",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_data_dir": TRAIN_DIR,
        "val_data_dir": VAL_DIR
    })

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Log data generator configs
    # mlflow.log_dict(train_datagen.get_config(), "train_datagen_config.json")
    # mlflow.log_dict(val_datagen.get_config(), "val_datagen_config.json")

    mlflow.log_dict(
        {
        "rescale": "1./255",
        # "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "fill_mode": "nearest"
        },
        "train_datagen_config.json")
    
    mlflow.log_dict(
        {
        "rescale": "1./255"
        }, 
        "val_datagen_config.json")
    

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    num_classes = train_generator.num_classes
    class_labels = list(train_generator.class_indices.keys())

    # Load base model
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze only the last conv block (block5_conv1, block5_conv2, block5_conv3)
    for layer in base_model.layers:
        # if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        if layer.name in [ 'block5_conv3']:
            layer.trainable = True

    # Custom classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    # x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Use a smaller learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # Log model summary
    with open(mlf + "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    mlflow.log_artifact(mlf + "model_summary.txt")


    # --------- Log Model Architecture as Image (Safe Execution) ---------
    try:
        model_plot_file = mlf + "model_architecture.png"
        plot_model(model, to_file=model_plot_file, show_shapes=True, show_layer_names=True)
        mlflow.log_artifact(model_plot_file)
    except Exception as e:
        print(f"[WARNING] Failed to log model architecture image: {e}")

    # Train
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

    # Log epoch-wise metrics
    for epoch in range(EPOCHS):
        mlflow.log_metric("train_accuracy_epoch", history.history["accuracy"][epoch], step=epoch)
        mlflow.log_metric("train_loss_epoch", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_accuracy_epoch", history.history["val_accuracy"][epoch], step=epoch)
        mlflow.log_metric("val_loss_epoch", history.history["val_loss"][epoch], step=epoch)

    # Final metrics
    mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])

    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(mlf + "accuracy.png")
    mlflow.log_artifact(mlf + "accuracy.png")

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(mlf + "loss.png")
    mlflow.log_artifact(mlf + "loss.png")

    # Predictions
    val_preds = model.predict(val_generator)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = val_generator.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(mlf + "confusion_matrix.png")
    mlflow.log_artifact(mlf + "confusion_matrix.png")

    # Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    with open(mlf + "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)
    with open(mlf + "classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_labels))
    mlflow.log_artifact(mlf + "classification_report.json")
    mlflow.log_artifact(mlf + "classification_report.txt")

    plt.figure(figsize=(8, 4))
    sns.heatmap(pd.DataFrame(report_dict).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
    plt.title("Classification Report")
    plt.savefig(mlf + "classification_report.png")
    mlflow.log_artifact(mlf + "classification_report.png")

    # ROC Curve (macro-averaged for multiclass)
    if num_classes > 2:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        auc_score = roc_auc_score(y_true_bin, val_preds, average='macro')
        mlflow.log_metric("macro_auc", auc_score)
    else:
        fpr, tpr, _ = roc_curve(y_true, val_preds[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(mlf + "roc_curve.png")
        mlflow.log_artifact(mlf + "roc_curve.png")
        mlflow.log_metric("auc", roc_auc)

    # Save model (HDF5 format) locally
    model.save(mlf + "vgg16_model.h5")
    mlflow.log_artifact(mlf + "vgg16_model.h5")

    # Log model to MLflow
    # mlflow.keras.log_model(model, "vgg16_model")

    try:
        mlflow.keras.log_model(
            model,
            artifact_path="vgg16_model",
            registered_model_name="KidneyDiseaseVGG16"
        )
    except Exception as e:
        print(f"[WARNING] Failed to register model: {e}")
