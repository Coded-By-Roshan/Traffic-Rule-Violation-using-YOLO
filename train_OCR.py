import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split  # Importing train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set dataset path
dataset_path = 'final_dataset'  # Update this path to your dataset

# Load data function
def load_data(dataset_path):
    images = []
    labels = []
    label_dict = {}
    label_index = 0

    # Loop through each folder (digit)
    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            label_dict[label_index] = folder_name  # Map index to digit
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (32, 32))  # Resize to (32, 32)
                    images.append(img)
                    labels.append(label_index)
            label_index += 1

    return np.array(images), np.array(labels), label_dict

# Load dataset
X, y, label_dict = load_data(dataset_path)

# Normalize pixel values to [0, 1]
X = X.astype('float32') / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=len(label_dict))

# Split dataset into training (80%) and validation/testing (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_dict), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                    height_shift_range=0.1, zoom_range=0.1)
val_datagen = ImageDataGenerator()  # No augmentation for validation

# Train the model
model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
          validation_data=val_datagen.flow(X_val, y_val),
          steps_per_epoch=len(X_train) // 32,
          validation_steps=len(X_val) // 32,
          epochs=10)

# Save the model using the native Keras format
model.save('OCR_MODEL.keras')

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_datagen.flow(X_val, y_val), verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Confusion Matrix Code
# Make predictions on the validation set
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)  # Get the class with the highest probability
y_true = np.argmax(y_val, axis=1)  # True labels

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
