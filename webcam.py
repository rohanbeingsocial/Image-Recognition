import tensorflow as tf
import numpy as np
import cv2
from bounding_box import draw_bounding_box

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Preprocessing function
def preprocess_image(image, img_size=(224, 224)):
    image = cv2.resize(image, img_size)  # Resize the image
    image = image.astype("float32")  # Convert to float32
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess for MobileNetV2
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to decode the model's prediction
def decode_prediction(preds):
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
    return decoded_preds

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the image for MobileNetV2
    processed_image = preprocess_image(frame)

    # Make prediction
    preds = model.predict(processed_image)
    decoded_preds = decode_prediction(preds)

    # Display the prediction on the image
    label = f"{decoded_preds[1]}: {decoded_preds[2]*100:.2f}%"

    # Simulate bounding box (drawing a square around the object)
    height, width, _ = frame.shape
    top_left = (width//4, height//4)  # Top-left corner of the square
    bottom_right = (3*width//4, 3*height//4)  # Bottom-right corner of the square

    # Call the function from bounding_box.py to draw the bounding box
    frame_with_box = draw_bounding_box(frame, top_left, bottom_right, label)

    # Show the frame with bounding box and label
    cv2.imshow("Image Recognition with Bounding Box", frame_with_box)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
