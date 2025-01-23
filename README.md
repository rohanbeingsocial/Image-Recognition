# Image-Recognition
Basic image recognition software using Python and chatgpt in 40 mins without having any prior knowledge to image recognition
# Real-Time Object Detection and Image Recognition with Python

This documentation provides a comprehensive guide to setting up and running a real-time object detection system in Python. It includes using a webcam feed, leveraging the COCO dataset, and experimenting with YOLO and pre-trained models.

---

## **Requirements**

### **Hardware Requirements**
- A computer with Python installed (3.8 or above recommended).
- A webcam (built-in or external).
- Sufficient memory and GPU (optional but recommended for YOLO).

### **Software Requirements**
- Python 3.x
- IDE: VSCode
- Libraries:
  - `opencv-python`
  - `numpy`
  - `torch`
  - `torchvision`
  - `pycocotools`
  - `tensorflow` (optional)

---

## **Setup**

### **Step 1: Environment Setup**
1. Install Python packages:
   ```bash
   pip install opencv-python numpy torch torchvision pycocotools tensorflow
   ```

2. Ensure your project folder contains:
   - `main.py` for the image recognition script.
   - `webcam.py` for webcam-based real-time detection.
   - `yolo_model/` folder for YOLO files (`cfg`, `weights`, `names`).

3. Example folder structure:
   ```plaintext
   project/
   ├── main.py
   ├── webcam.py
   ├── yolo_model/
   │   ├── yolov4.cfg
   │   ├── yolov4.weights
   │   ├── coco.names
   └── images/
   ```

---

## **YOLO Setup**

### **Step 2: Download YOLO Files**
1. Download YOLO files from:
   - [YOLOv4 weights](https://github.com/AlexeyAB/darknet/releases/)
   - Configuration file (e.g., `yolov4.cfg`).
   - COCO names file (`coco.names`).

2. Place them in the `yolo_model` directory.

### **Step 3: Write YOLO Detection Script**
#### **File: `main.py`**
```python
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolo_model/yolov4.weights", "yolo_model/yolov4.cfg")
with open("yolo_model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get YOLO outputs
    output_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(output_layers)

    # Analyze the outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Draw bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## **Webcam Real-Time Object Selection**

### **Step 4: Adding Object Selection**
Modify the script to accept user input for objects to detect.

#### **File: `main.py`**
```python
import threading

# Shared variable for object selection
selected_object = None

def input_thread():
    global selected_object
    while True:
        selected_object = input("Enter object name: ").strip()

# Start input thread
thread = threading.Thread(target=input_thread, daemon=True)
thread.start()

# Detection logic (add a check for `selected_object`)
for detection in out:
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    if confidence > 0.5 and classes[class_id] == selected_object:
        # Draw bounding box (same as above)
```

---

## **Using COCO Dataset (`train2014` Folder)**

### **Step 5: Convert COCO to YOLO Format**
To use `train2014` images for custom training:
1. Convert COCO annotations to YOLO format using `pycocotools`.
2. Use the script below:

```python
import os
import json
from pycocotools.coco import COCO

# Paths
coco_images_path = "path/to/train2014"
coco_annotations_path = "path/to/instances_train2014.json"
output_path = "path/to/yolo_annotations"

# Load COCO annotations
coco = COCO(coco_annotations_path)
for img_id in coco.getImgIds():
    # Fetch image and annotations (see full script above)
    pass
```

---

## **Troubleshooting**

### **Common Issues**
1. **File Not Found Errors**:
   - Ensure paths to YOLO weights, config, and `coco.names` are correct.
2. **Low FPS in Webcam Feed**:
   - Reduce image size by scaling down the input to YOLO.
   - Use a lightweight model like YOLOv4-tiny.
3. **AttributeError for `getLayers`**:
   - Use `getUnconnectedOutLayersNames()` for newer OpenCV versions.

---

## **References**
- [COCO Dataset](https://cocodataset.org/)
- [YOLO GitHub Repository](https://github.com/AlexeyAB/darknet)
- [OpenCV Documentation](https://docs.opencv.org/)

