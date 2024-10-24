import cv2
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import pickle

# Load the model paths from pickle file
with open('model_paths.pkl', 'rb') as f:
    model_paths = pickle.load(f)

# Load your custom YOLO models for garbage detection and dry/wet classification
garbage_model = YOLO(model_paths['garbage_model_path'])
dry_wet_model = YOLO(model_paths['dry_wet_model_path'])

# Open video capture
cap = cv2.VideoCapture(0)

# Create a dataset for KNN (bounding box size vs distance)
bbox_size = np.array([[50, 50], [100, 100], [150, 150], [200, 200], [250, 250]])  # Example values
distance = np.array([3, 2, 1.5, 1, 0.5])  # Corresponding distances in meters

# Initialize KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(bbox_size, distance)  # Train the KNN model

# Define boundary parameters
boundary_distance = 1.0  # 1 meter

# Function to draw a curved boundary line centered in the frame
def draw_curved_boundary(frame):
    frame_height, frame_width = frame.shape[:2]
    curve_depth = 30  # Control the maximum depth of the curve above and below the midpoint
    curve_center_y = frame_height // 2  # Midpoint height

    num_points = frame_width
    curve_points = []

    for x in range(num_points):
        t = x / (frame_width - 1)  # Parameter from 0 to 1
        y = curve_center_y + int(curve_depth * np.sin(np.pi * t))
        curve_points.append((x, y))

    for i in range(len(curve_points) - 1):
        cv2.line(frame, curve_points[i], curve_points[i + 1], (0, 255, 0), 2)

    return frame

# Noise handling using Gaussian Blur and morphological operations
def reduce_noise(frame):
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Apply morphological operations to clean the image
    kernel = np.ones((3, 3), np.uint8)
    clean_frame = cv2.morphologyEx(blurred_frame, cv2.MORPH_CLOSE, kernel)
    
    return clean_frame

# Function to detect motion using frame differencing
def detect_motion(prev_frame, current_frame):
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate frame difference
    frame_diff = cv2.absdiff(gray_prev, gray_current)

    # Apply threshold to highlight regions with significant motion
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Dilation to enhance the motion areas
    dilated = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
    
    return dilated

# Function to update the frame in Tkinter
def update_frame(prev_frame=None):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    
    # Reduce noise in the frame
    frame = reduce_noise(frame)

    # Handle motion detection if a previous frame is provided
    if prev_frame is not None:
        motion_mask = detect_motion(prev_frame, frame)
        # Only process if there is motion detected
        if np.sum(motion_mask) < 500:  # Adjust threshold as needed
            prev_frame = frame.copy()
            panel.after(10, lambda: update_frame(prev_frame))
            return

    # Perform garbage detection
    garbage_results = garbage_model(frame)

    garbage_count = 0
    closest_outside_distance = float('inf')
    closest_label = ""

    for result in garbage_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = box.cls[0]

            if conf < 0.5:  # Skip low-confidence detections
                continue

            garbage_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f'{garbage_model.names[int(class_id)]} {conf:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cropped_garbage = frame[y1:y2, x1:x2]

            # Perform dry/wet classification
            dry_wet_results = dry_wet_model(cropped_garbage)
            dry_wet_label = 'Common'
            dry_wet_confidence = 0.0

            for dw_result in dry_wet_results:
                if len(dw_result.boxes) > 0:
                    dw_class_id = dw_result.boxes[0].cls[0]
                    dry_wet_label = dry_wet_model.names[int(dw_class_id)]
                    dry_wet_confidence = dw_result.boxes[0].conf[0]

            cv2.putText(frame, f"Dry/Wet: {dry_wet_label} ({dry_wet_confidence:.2f})", 
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Calculate bounding box size
            box_width = x2 - x1
            box_height = y2 - y1
            predicted_distance = knn.predict([[box_width, box_height]])[0]

            if predicted_distance <= boundary_distance:
                cv2.putText(frame, f"Inside 1m ({predicted_distance:.2f}m)", 
                            (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Outside 1m ({predicted_distance:.2f}m)", 
                            (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if predicted_distance < closest_outside_distance:
                    closest_outside_distance = predicted_distance
                    closest_label = dry_wet_label

    # Draw the boundary curve on the frame
    frame = draw_curved_boundary(frame)

    cv2.putText(frame, f"Garbage Count: {garbage_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the image from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)  # Convert to PIL Image
    imgtk = ImageTk.PhotoImage(image=img)  # Convert to ImageTk
    panel.imgtk = imgtk  # Keep a reference
    panel.config(image=imgtk)  # Update the panel with the new image

    # Check for closest object outside 1 meter
    if closest_outside_distance < float('inf'):
        print(f"Closest object outside 1 meter: {closest_label} at {closest_outside_distance:.2f}m")

    # Schedule the next frame update
    panel.after(10, lambda: update_frame(frame))

# Set up the Tkinter window
root = tk.Tk()
root.title("Garbage Detection")

# Create a label to display the video frames
panel = tk.Label(root)
panel.pack()

# Start updating the frames
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
