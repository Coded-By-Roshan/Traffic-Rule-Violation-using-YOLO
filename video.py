from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
# Load the YOLOv8 model for license plate detection
model = YOLO('license_plate_detector.pt')
font_path = "nepalifont.otf"
# Load the OCR model
ocr_model_path = 'OCR_MODEL.keras'
ocr_model = load_model(ocr_model_path)

# Character list (Nepali + English)
nepali_chars = "०१२३४५६७८९अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह"
english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_list = nepali_chars + english_chars

# Open video for processing
cap = cv2.VideoCapture('video.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


# resize_width = 640
resize_width = 1000
resize_height = 680
# resize_height = 480
# Function to overlay Unicode text on a video frame
def put_unicode_text(frame, text, position, font_path="nepalifont.otf", font_size=32, color=(255, 255, 255)):
    # Convert the frame (numpy array) to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Initialize ImageDraw
    draw = ImageDraw.Draw(pil_image)
    
    # Load the Nepali Unicode font
    font = ImageFont.truetype(font_path, font_size)
    
    # Draw the text
    draw.text(position, text, font=font, fill=color)
    
    # Convert the PIL Image back to a numpy array (OpenCV format)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def segment_characters(number_plate_image):

    fixed_size = (1000, 480)
    number_plate_image = cv2.resize(number_plate_image, fixed_size)

    # Convert to grayscale
    gray = cv2.cvtColor(number_plate_image, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast by normalizing
    gray = cv2.equalizeHist(gray)

    # Apply bilateral filter for noise reduction
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 30, 200)

    # Use adaptive thresholding to create binary image
    binary = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    
    # Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    # cv2.imshow('Detection', binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    character_images = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / float(w)

        # Filter based on area and aspect ratio
        if area > 200 and 0.5 < aspect_ratio < 4:
            char_img = gray[y:y + h, x:x + w]
            char_img = cv2.resize(char_img, (32, 32))
            character_images.append(char_img)
    return character_images


def predict_character(character_image):
    character_image = cv2.cvtColor(character_image, cv2.COLOR_GRAY2RGB)
    character_image = np.expand_dims(character_image, axis=0)
    character_image = character_image.astype('float32') / 255.0
    prediction = ocr_model.predict(character_image)
    predicted_index = np.argmax(prediction)
    
    # Print the predicted probabilities for debugging
    print(f"Prediction Probabilities: {prediction}")
    
    return char_list[predicted_index]


# Function to recognize text from number plate
def recognize_number_plate(number_plate_image):
    segmented_characters = segment_characters(number_plate_image)
    predicted_text = ""
    for char_img in segmented_characters:
        predicted_char = predict_character(char_img)
        predicted_text += predicted_char
    return predicted_text

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection using YOLO
    results = model(frame)
    for result in results:
        for box in result.boxes.xyxy:  # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure bounding box coordinates are within frame dimensions
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
            
            # Crop the detected license plate area
            cropped_plate = frame[y1:y2, x1:x2]

            # Recognize text from the cropped plate
            plate_text = recognize_number_plate(cropped_plate)
            print("platetext = ",plate_text)
            if not plate_text.strip():  # Handle cases where OCR fails
                plate_text = "Detection Failed"

            print(f"Recognized Plate Text: {plate_text}")

            # Annotate the frame with detected plate and recognized text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = put_unicode_text(frame, plate_text, position=(x1, y1-80), font_path=font_path, font_size=50, color=(0, 255, 0))
            # cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resize frame for display and write to output video
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    # out.write(frame)
    cv2.imshow('Detection', resized_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()

cv2.destroyAllWindows()
