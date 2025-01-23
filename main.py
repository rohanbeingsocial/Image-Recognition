import cv2
import numpy as np
import threading

# Load YOLO model (YOLOv4-tiny in this case)
net = cv2.dnn.readNet("yolo_model/yolov4-tiny.weights", "yolo_model/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Disable CUDA
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU

# Load COCO class labels
with open("yolo_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variable for user input
object_to_detect = ""

# Function to get user input in real-time
def get_user_input():
    global object_to_detect
    while True:
        object_to_detect = input("Enter the object name to detect (press Enter to keep it unchanged): ").lower()

# Start the input thread
input_thread = threading.Thread(target=get_user_input, daemon=True)
input_thread.start()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Target FPS limit
fps_limit = 30

# Start time to calculate FPS
prev_time = 0

frame_skip = 2  # Skip frames for better performance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for better performance
    if frame_skip > 0:
        frame_skip -= 1
        continue
    else:
        frame_skip = 2  # Reset frame skipping counter

    # Resize the frame to improve speed (optional)
    frame_resized = cv2.resize(frame, (416, 416))

    # Prepare frame for YOLO
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process outputs and draw bounding boxes
    class_ids, confidences, boxes = [], [], []
    height, width, _ = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]).lower()
            confidence = confidences[i]

            # Draw box only if the detected object matches the input
            if object_to_detect in label:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for 'q' key to stop the script
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
