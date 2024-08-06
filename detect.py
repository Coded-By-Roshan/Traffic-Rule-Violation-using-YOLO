import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, Add, BatchNormalization, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# Define image dimensions and batch size
img_width, img_height = 32, 32
batch_size = 32

# Define custom preprocessing function
def preprocess_image(image):
    image = img_to_array(image).astype(np.uint8)
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    grayscale_image = grayscale_image.astype(np.float32) / 255.0
    # Convert back to 3 channels
    grayscale_image = np.stack((grayscale_image,) * 3, axis=-1)
    return grayscale_image

# Split dataset into training and testing sets
dataset_dir = 'datasets'
train_dir = 'new_dataset/train'
test_dir = 'new_dataset/test'

# Ensure the directories exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# # Split the dataset
# for class_name in os.listdir(dataset_dir):
#     class_dir = os.path.join(dataset_dir, class_name)
#     if os.path.isdir(class_dir):
#         images = os.listdir(class_dir)
#         train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)
        
#         # Create class directories in train and test directories
#         train_class_dir = os.path.join(train_dir, class_name)
#         test_class_dir = os.path.join(test_dir, class_name)
        
#         if not os.path.exists(train_class_dir):
#             os.makedirs(train_class_dir)
#         if not os.path.exists(test_class_dir):
#             os.makedirs(test_class_dir)
        
#         # Move images to the respective directories
#         for image in train_images:
#             shutil.move(os.path.join(class_dir, image), os.path.join(train_class_dir, image))
#         for image in test_images:
#             shutil.move(os.path.join(class_dir, image), os.path.join(test_class_dir, image))

# Define data generators with data augmentation and custom preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Verify number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Define skip connection function
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Load the MobileNetV2 model with pre-trained weights (excluding the top classification layers)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Unfreeze some of the layers in the base model for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Add custom layers on top of the base model
x = base_model.output
x = residual_block(x, 64)
x = residual_block(x, 128)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Calculate class weights
counter = Counter(train_generator.classes)
max_count = max(counter.values())
class_weights = {class_id: max_count / count for class_id, count in counter.items()}

# Train the model with callbacks
epochs = 100
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    class_weight=class_weights
)

# Save the final model
model.save('newdata_final_model2.h5')

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')

# Make predictions on test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Create confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=test_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.savefig('newdata_final_confusion_matrix2.png')
plt.show()


# Test loss: 0.10644467175006866
# Test accuracy: 0.9723326563835144