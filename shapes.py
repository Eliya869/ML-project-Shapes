import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. Load the images from the dataset i downloaded
def load_images(folder, label):

    images, labels = [], []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
            img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
            images.append(img)
            labels.append(label)
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels)


# 2. Create Dataset Paths and Loading
BASE_PATH = r"C:\Users\eliya\Desktop\shapes"  # Adjust to your dataset location
categories = ["circles", "squares", "triangles"]  # Circle, square, triangle categories

data, labels = [], []
for idx, category in enumerate(categories):
    folder = os.path.join(BASE_PATH, category)
    images, category_labels = load_images(folder, idx)
    data.append(images)
    labels.append(category_labels)

X = np.vstack(data)  # Combined all images into a single array
y = np.concatenate(labels)  # Combined all labels into a single array
y = to_categorical(y, len(categories))  # Converted labels to one-hot encoding

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
)


# 3. Building the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Reduce overfitting
    layers.Dense(len(categories), activation='softmax')  # Output layer for shape categories
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# 4. Train the Model
augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

history = model.fit(
    augmentation.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=30,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]
)

# 5. Evaluate the Model
final_acc = model.evaluate(X_test, y_test, verbose=0)[1] * 100
print(f"Final accuracy: {final_acc:.2f}%")  # Print final accuracy

# Plot learning curve (accuracy and loss)
def plot_learning_curve(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Learning Curve (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Confusion matrix
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

plot_learning_curve(history)  # Visualize learning curve
plot_confusion_matrix(model, X_test, y_test)  # Visualize confusion matrix