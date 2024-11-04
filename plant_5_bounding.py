import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import tkinter as tk

# Load the TensorFlow model (assuming a standard TensorFlow model, not TFLite)
model_path = r"C:\Users\weake\OneDrive\Desktop\plant\plant_disease_model.tflite"  # Use a .h5 model instead
model = tf.keras.models.load_model(model_path)

# Define the class labels
class_labels = [
    'Apple Scab', 'Apple Cedar Rust', 'Apple Leaf Spot', 'Apple Powdery Mildew', 'Apple Fruit Rot',
    'Blueberry Leaf Spot', 'Blueberry Mummy Berry', 'Cherry Leaf Spot', 'Cucumber Downy Mildew',
    'Cucumber Powdery Mildew', 'Tomato Early Blight', 'Tomato Late Blight',
    'Potato Late Blight', 'Potato Early Blight', 'Pepper Bacterial Spot',
    'Aloe Vera Leaf Spot', 'Neem Leaf Blight', 'Tulsi Powdery Mildew', 
    'Ashwagandha Root Rot', 'Basil Downy Mildew', 'Oregano Powdery Mildew',
    'Rice Blast', 'Wheat Rust', 'Bean Rust', 'Rose Black Spot',
    'Banana Leaf Spot', 'Coconut Lethal Yellowing', 'Cotton Wilt', 'unknown'
]

# Initialize webcam (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Function to preprocess the frame for prediction
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (128, 128))  # Adjust to your model's input size
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0).astype(np.float32)
    return input_frame

# Function to detect possible affected areas and draw bounding boxes
def detect_affected_areas(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([15, 40, 40])
    upper_bound = np.array([35, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
    
    return frame

# Function to update the image displayed in the Tkinter window
def update_image():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return

    # Preprocess the frame for classification
    input_frame = preprocess_frame(frame)
    
    # Get prediction using the model
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)

    # Ensure predicted_class is within the bounds of class_labels
    predicted_label = class_labels[predicted_class] if predicted_class < len(class_labels) else 'unknown'

    # Detect affected areas and draw bounding boxes
    frame_with_boxes = detect_affected_areas(frame.copy())

    # Display the predicted class label on the video frame
    frame_with_text = cv2.putText(frame_with_boxes, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Update the Tkinter image
    tk_img = ImageTk.PhotoImage(image=frame_pil)
    label.config(image=tk_img)
    label.image = tk_img

    root.after(10, update_image)  # Schedule the function to be called again after 10ms

# Set up Tkinter window
root = tk.Tk()
root.title("Plant Disease Detection")
label = tk.Label(root)
label.pack()

# Start the update loop
update_image()
root.mainloop()

# Release the webcam
cap.release()
