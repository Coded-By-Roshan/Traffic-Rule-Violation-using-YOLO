import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture('v2.mp4')
font = cv2.FONT_HERSHEY_PLAIN

# Ground truth and predicted lists for confusion matrix
ground_truth = []
predicted = []

# Initialize frame_id
frame_id = 0

# Function to simulate ground truth data
def simulate_ground_truth(frame_id):
    # Placeholder function for getting ground truth
    # Replace with actual ground truth data for real-world usage
    if frame_id % 2 == 0:
        return ["car", "bus"]
    else:
        return ["truck", "motorbike"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["car", "bus", "truck", "motorbike"]:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)
            predicted.append(label)

    # Simulate ground truth data
    gt_labels = simulate_ground_truth(frame_id)
    ground_truth.extend(gt_labels)
    frame_id += 1

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


