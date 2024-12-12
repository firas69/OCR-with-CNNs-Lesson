# OCR Number Detection with CNNs

This README provides a comprehensive guide to building an Optical Character Recognition (OCR) system for number detection using Convolutional Neural Networks (CNNs). The lesson includes everything from setup to deployment, with practical steps and code snippets.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Setup and Environment](#setup-and-environment)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Building](#model-building)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Data Augmentation](#data-augmentation)
8. [Fine-Tuning and Transfer Learning](#fine-tuning-and-transfer-learning)
9. [Model Deployment](#model-deployment)
10. [Troubleshooting](#troubleshooting)

---

## **1. Project Overview**

This project demonstrates how to use CNNs for OCR, focusing on detecting handwritten digits from images. We'll use the popular MNIST dataset for training and testing the model, though it can be extended to other datasets.

---

## **2. Setup and Environment**

### **Prerequisites**
- Python 3.8 or higher
- Basic knowledge of Python, machine learning, and deep learning

### **Install Required Libraries**
```bash
pip install numpy pandas matplotlib tensorflow keras optuna
```

### **Directory Structure**
```
OCR-CNN/
├── data/                # Dataset files
├── models/              # Saved models
├── results/             # Model performance results
├── utils/               # Helper scripts (e.g., metrics, logging)
└── main.py              # Main script
```

---

## **3. Dataset Preparation**

### **Download the Dataset**
For this project, we'll use the MNIST dataset, which is readily available in Keras.

### **Loading and Splitting Data**
```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for CNN input
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

---

## **4. Model Building**

### **Define the CNN Architecture**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## **5. Model Training and Evaluation**

### **Training the Model**
```python
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

### **Evaluate the Model**
```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

---

## **6. Hyperparameter Tuning**

### **Using Keras Tuner**
```python
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('filters', 32, 128, step=32), (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(hp.Int('units', 64, 256, step=64), activation='relu'))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Tuner
tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, factor=3)
tuner.search(x_train, y_train, validation_split=0.2, epochs=10)
```

---

## **7. Data Augmentation**

### **Applying Data Augmentation**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# Training with augmented data
model.fit(datagen.flow(x_train, y_train, batch_size=64), validation_data=(x_test, y_test), epochs=10)
```

---

## **8. Fine-Tuning and Transfer Learning**

### **Using a Pre-Trained Model**
```python
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## **9. Model Deployment**

### **Saving and Loading the Model**
```python
# Save model
model.save("models/cnn_model.h5")

# Load model
from tensorflow.keras.models import load_model
model = load_model("models/cnn_model.h5")
```
---

## **10. Troubleshooting**

- **Issue:** Model training is slow.
  - **Solution:** Use a GPU (e.g., Google Colab or a local GPU-enabled setup).
- **Issue:** Dimension mismatch errors.
  - **Solution:** Ensure input data shape matches the model's expected input shape.
- **Issue:** Overfitting on training data.
  - **Solution:** Apply data augmentation and regularization (e.g., Dropout).

---

### **Conclusion**
By following this guide, you can build, train, and deploy an OCR model for number detection. Experiment with hyperparameters, data augmentation, and fine-tuning to improve performance.
