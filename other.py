import os
import numpy as np
os.add_dll_directory(r'C:\Users\Ruben\Documents\opencv\build\x64\vc16\bin')

from ultralytics import YOLO
import cv2
import math
import colorsys

# Generate unique colors for each class
def generate_unique_colors(num_classes):
    colors = []
    for i in range(num_classes):
        # Use HSV color space to generate evenly distributed colors
        hue = i / num_classes
        # Full saturation and value for vibrant colors
        saturation = 0.9
        value = 0.9
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to 0-255 range and to BGR for OpenCV
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

def select_camera():
    # List available cameras
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available")
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("No cameras detected!")
        return 0
    
    # Let user select a camera
    selected = 0
    if len(available_cameras) > 1:
        print("\nSelect a camera by entering its index:")
        for idx in available_cameras:
            print(f"  {idx}: Camera {idx}")
        
        try:
            selected = int(input("\nEnter camera index: "))
            if selected not in available_cameras:
                print(f"Invalid selection. Using default camera 0")
                selected = 0
        except ValueError:
            print(f"Invalid input. Using default camera 0")
    
    return selected

# Select camera
camera_idx = select_camera()

# Start webcam
cap = cv2.VideoCapture(camera_idx)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Generate unique colors for each class
class_colors = generate_unique_colors(len(classNames))

# Window name
window_name = f'YOLO Detection (Camera {camera_idx})'

while True:
    success, img = cap.read()
    if not success:
        print(f"Failed to read from camera {camera_idx}")
        break
        
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            
            # Confidence
            confidence = math.ceil((box.conf[0]*100))/100
            
            # Get color for this class
            color = class_colors[cls]
            
            # Put box in cam with class-specific color
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size for better positioning
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                img, 
                (x1, y1 - text_height - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw text with white color for better visibility
            cv2.putText(
                img, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # Print detection info to console
            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

    # Show frame rate
    cv2.imshow(window_name, img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()