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



def segment_characters(number_plate_image):
    # Resize the image for consistent processing (standard size)
    fixed_size = (1000, 480)  # Adjust this as needed
    number_plate_image = cv2.resize(number_plate_image, fixed_size)

    # Convert the image to grayscale
    gray = cv2.cvtColor(number_plate_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 13, 15, 15)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)
    # Use adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(canny, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    # # Use morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Open operation removes small objects
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close operation fills small holes



    # Find contours of the characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    character_images = []
    for contour in contours:
    # Calculate the area of the contour
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / float(w)
        if area > 200 and 0.5 < aspect_ratio < 4:
            
            cv2.rectangle(binary, (x, y), (x + w, y + h), (0, 255, 0), 2)

            char_img = gray[y:y + h, x:x + w]

            char_img = cv2.resize(char_img, (32, 32))
            character_images.append(char_img)

            # Debug: Show each character
            cv2.imshow(f"Character {len(character_images)}", char_img)
            cv2.waitKey(0)

    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return segmented characters
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
# Main function to test on a number plate image
def main(number_plate_image_path):
    # Load number plate image
    number_plate_image = cv2.imread(number_plate_image_path)
    
    # Recognize the full number plate text
    predicted_text = recognize_number_plate(number_plate_image)
    
    print(f"Predicted Number Plate: {predicted_text}")

# Example usage
if __name__ == "__main__":
    # Use os.path.join for platform-independent file path handling
    number_plate_image_path = "license-plates-datasets\plate1.png"
    # number_plate_image_path = "grey_numberplate.png"
    main(number_plate_image_path)


