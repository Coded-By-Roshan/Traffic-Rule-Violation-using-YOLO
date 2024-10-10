import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
ocr_model_path = 'OCR_MODEL.keras'
ocr_model = load_model(ocr_model_path)

# Character list (Nepali + English)
nepali_chars = "०१२३४५६७८९अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह"
english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_list = nepali_chars + english_chars

# Function to preprocess the number plate image and segment characters
def segment_characters(number_plate_image):
    # Convert to grayscale
    gray = cv2.cvtColor(number_plate_image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to remove noise and separate characters
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours of the characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    character_images = []
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out contours that are too small or too large to be characters
        if h > 10 and w > 10:  # Adjust based on image size
            char_img = number_plate_image[y:y+h, x:x+w]
            char_img = cv2.resize(char_img, (32, 32))  # Resize to model input size
            character_images.append(char_img)
    
    return character_images

# Function to predict character from segmented images
def predict_character(character_image):
    # Preprocess character image for model
    character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    character_image = np.expand_dims(character_image, axis=0)  # Add batch dimension
    character_image = character_image.astype('float32') / 255.0  # Normalize pixel values
    
    # Predict character using the model
    prediction = ocr_model.predict(character_image)
    predicted_index = np.argmax(prediction)
    predicted_character = char_list[predicted_index]
    
    return predicted_character

# Function to recognize full number plate
def recognize_number_plate(number_plate_image):
    # Segment characters from number plate
    segmented_characters = segment_characters(number_plate_image)
    
    # Predict each character and combine them
    predicted_text = ""
    for char_img in segmented_characters:
        predicted_char = predict_character(char_img)
        predicted_text += predicted_char
    
    return predicted_text

# Main function to test on a number plate image
def main(number_plate_image_path):
    # Load number plate image
    number_plate_image = cv2.imread(number_plate_image_path)
    
    # Recognize the full number plate text
    predicted_text = recognize_number_plate(number_plate_image)
    
    print(f"Predicted Number Plate: {predicted_text}")

# Example usage
if __name__ == "__main__":
    number_plate_image_path = 'D:\Major Project\license-plates-datasets\plate100.png'  # Update with the image path
    main(number_plate_image_path)
