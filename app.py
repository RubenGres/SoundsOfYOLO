import os
import numpy as np
import argparse
import time
import colorsys
from ultralytics import YOLO
import cv2
import math
import mido
from mido import Message


allowed_classes = ["backpack", "umbrella", "suitcase", "sports ball",
                "skateboard", "bottle", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "orange", "carrot", "bottle",
                "pottedplant", "remote", "cell phone", "book", "teddy bear", "toothbrush"]

def calculate_note(class_name):
    name_to_note = {
        "backpack": 34,
        "umbrella": 34,
        "suitcase": 34,
        "sports ball": 34,
        "skateboard": 34,
        "bottle": 34,
        "cup": 34,
        "fork": 34,
        "knife": 34,
        "spoon": 34,
        "bowl": 34,
        "banana": 34,
        "apple": 34,
        "orange": 34,
        "carrot": 34,
        "bottle": 34,
        "pottedplant": 34,
        "remote": 34,
        "cell phone": 34,
        "book": 34,
        "teddy bear": 34,
        "toothbrush": 34
    }

    return name_to_note[class_name]

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


def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO Object Detection with MIDI Output')
    parser.add_argument('--camera', type=int, help='Camera device index')
    parser.add_argument('--midi', type=int, help='MIDI output port index')
    return parser.parse_args()

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

def select_camera(camera_idx=None):
    # If camera index provided, use it
    if camera_idx is not None:
        return camera_idx
    
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

def select_midi_port(port_idx=None):
    # Get available output ports
    output_ports = mido.get_output_names()
    
    if not output_ports:
        print("No MIDI output ports available. Please connect a MIDI device or install a virtual MIDI port.")
        return None
    
    # If port index provided, validate and use it
    if port_idx is not None:
        if 0 <= port_idx < len(output_ports):
            return output_ports[port_idx]
        else:
            print(f"Invalid MIDI port index {port_idx}. Will prompt for selection.")
    
    # Display available ports with numbers
    print("Available MIDI output ports:")
    for i, port_name in enumerate(output_ports):
        print(f"[{i}] {port_name}")
    
    # Get user input for port selection
    while True:
        selection = input("\nSelect a port number: ")
        try:
            port_index = int(selection)
            if 0 <= port_index < len(output_ports):
                return output_ports[port_index]
            else:
                print(f"Please enter a number between 0 and {len(output_ports)-1}")
        except ValueError:
            print("Please enter a valid number")

def calculate_velocity(box_area, image_area):
    # Calculate velocity based on the box size relative to the image
    # Minimum velocity is 40, maximum is 127 (MIDI velocity range is 0-127)
    percentage = (box_area / image_area) * 100
    
    # Scale percentage to MIDI velocity (40-127)
    velocity = int(40 + (percentage * 0.87))
    # Ensure velocity is within MIDI range
    return max(40, min(127, velocity))

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Select camera
    camera_idx = select_camera(args.camera)
    
    # Select MIDI port
    selected_port = select_midi_port(args.midi)
    if selected_port is None:
        print("No MIDI port selected. Exiting.")
        return
    
    # Open the selected MIDI port
    print(f"Opening MIDI port: {selected_port}")
    port = mido.open_output(selected_port)
    print(f"Successfully opened MIDI port: {selected_port}")
    
    # Start webcam
    cap = cv2.VideoCapture(camera_idx)
    cap.set(3, 640)
    cap.set(4, 480)
    
    # Get frame dimensions for area calculations
    image_width = int(cap.get(3))
    image_height = int(cap.get(4))
    image_area = image_width * image_height
    
    # Model
    model = YOLO("yolo-Weights/yolov8n.pt")

    
    # Generate unique colors for each class
    class_colors = generate_unique_colors(len(classNames))
    
    # Store previously detected classes to track when objects appear/disappear
    prev_detected_classes = set()
    
    # Window name
    window_name = f'YOLO Detection with MIDI (Camera {camera_idx})'
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                print(f"Failed to read from camera {camera_idx}")
                break
                
            results = model(img, stream=True)
            
            # Track current detections
            current_detected_classes = set()
            
            # Coordinates
            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
                    
                    # Calculate box area
                    box_area = (x2 - x1) * (y2 - y1)
                    
                    # Class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    #Only accept the allowed classes
                    if class_name not in allowed_classes:
                        continue

                    current_detected_classes.add(cls)
                    
                    # Calculate MIDI parameters
                    note = calculate_note(class_name)
                    velocity = calculate_velocity(box_area, image_area)
                    
                    # Send MIDI note_on message for newly detected objects
                    if cls not in prev_detected_classes:
                        port.send(Message('note_on', note=note, velocity=velocity))
                        print(f"MIDI Note ON: Class={class_name}, Note={note}, Velocity={velocity}")
                    
                    # Confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    
                    # Get color for this class
                    color = class_colors[cls]
                    
                    # Put box in cam with class-specific color
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label with class name, confidence and MIDI info
                    label = f"{class_name}: {confidence:.2f} | Note: {note}, Vel: {velocity}"
                    
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
            
            # Send MIDI note_off for objects that disappeared
            for cls in prev_detected_classes - current_detected_classes:
                note = calculate_note(classNames[cls])
                port.send(Message('note_off', note=note, velocity=0))
                print(f"MIDI Note OFF: Class={classNames[cls]}, Note={note}")
            
            # Update previous detections
            prev_detected_classes = current_detected_classes
            
            # Show frame
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping detection and MIDI output")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Turn off any remaining notes
        for cls in prev_detected_classes:
            note = calculate_note(classNames[cls])
            port.send(Message('note_off', note=note, velocity=0))
        
        # Close resources
        port.close()
        print("MIDI port closed")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()