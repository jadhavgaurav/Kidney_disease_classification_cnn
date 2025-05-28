import os
import mlflow
import mlflow.tensorflow
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import itertools

# Paths
train_dir = "data/kidney_split/train"
val_dir = "data/kidney_split/val"

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50  # Keep low for testing multiple runs
BEST_ACCURACY_FILE = "best_accuracy.txt"


def read_best_accuracy():
    if os.path.exists(BEST_ACCURACY_FILE):
        with open(BEST_ACCURACY_FILE, 'r') as f:
            return float(f.read().strip())
    return 0.0


def update_best_accuracy(acc):
    with open(BEST_ACCURACY_FILE, 'w') as f:
        f.write(str(acc))


# Confusion matrix plot function
def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# MLflow experiment setup
mlflow.set_experiment("Kidney_Classification_Experiment")

dropout_values = [0.3, 0.5]
learning_rates = [1e-3, 1e-4]

for dropout in dropout_values:
    for lr in learning_rates:
        with mlflow.start_run():
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("learning_rate", lr)

            # Build model
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(dropout)(x)
            outputs = Dense(train_generator.num_classes, activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

            # Train
            history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, verbose=0)
            val_acc = history.history['val_accuracy'][-1]
            mlflow.log_metric("val_accuracy", val_acc)

            # Evaluation
            y_pred_probs = model.predict(val_generator)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = val_generator.classes

            class_report = classification_report(y_true, y_pred, output_dict=True)
            mlflow.log_metric("precision", class_report['weighted avg']['precision'])
            mlflow.log_metric("recall", class_report['weighted avg']['recall'])
            mlflow.log_metric("f1-score", class_report['weighted avg']['f1-score'])

            try:
                y_score = y_pred_probs[:, 1] if y_pred_probs.shape[1] == 2 else np.max(y_pred_probs, axis=1)
                auc = roc_auc_score(y_true, y_score, multi_class='ovr')
                mlflow.log_metric("roc_auc", auc)
            except:
                pass

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            class_labels = list(val_generator.class_indices.keys())
            cm_filename = f"confusion_matrix_d{int(dropout*100)}_lr{int(lr*1e5)}.png"
            plot_confusion_matrix(cm, class_labels, filename=cm_filename)
            mlflow.log_artifact(cm_filename)

            # Model registration logic
            best_val_acc = read_best_accuracy()
            if val_acc > best_val_acc:
                print(f"[INFO] New best model found! val_accuracy: {val_acc:.4f} (prev: {best_val_acc:.4f})")
                update_best_accuracy(val_acc)
                mlflow.tensorflow.log_model(model, "model", registered_model_name="KidneyDiseaseClassifier")
            else:
                print(f"[INFO] Model not registered. val_accuracy ({val_acc:.4f}) <= best ({best_val_acc:.4f})")
                mlflow.tensorflow.log_model(model, "model")
