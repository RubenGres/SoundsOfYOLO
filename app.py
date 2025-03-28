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

# Function to calculate MIDI note values for each class
def calculate_note(class_name):
    name_to_note = {
        "backpack": 60,
        "umbrella": 60,
        "suitcase": 60,
        "sports ball": 60,
        "skateboard": 60,
        "bottle": 60,
        "cup": 60,
        "fork": 60,
        "knife": 60,
        "spoon": 60,
        "bowl": 60,
        "banana": 60,
        "apple": 60,
        "orange": 60,
        "carrot": 60,
        "bottle": 60,
        "pottedplant": 60,
        "remote": 60,
        "cell phone": 60,
        "book": 60,
        "teddy bear": 60,
        "toothbrush": 60
    }

    return name_to_note[class_name]

# Map class names to CC control channels for X and Y positions
def get_control_channels(class_name):
    # Calculate control channels dynamically based on index position in allowed_classes
    # Each class gets two control channels: one for X position, one for Y
    # Start control channels at 1 and 2 for the first class, 3 and 4 for the second, etc.
    
    if class_name not in allowed_classes:
        # Fallback for safety, though this shouldn't happen due to filtering
        print(f"Warning: {class_name} not in allowed classes list")
        return (1, 2)
        
    # Find the index of the class in the allowed_classes list
    class_index = allowed_classes.index(class_name)
    
    # Calculate control channels: 
    # First class gets CC 1,2; second gets 3,4; etc.
    x_cc = class_index * 2 + 1
    y_cc = class_index * 2 + 2
    
    return (x_cc, y_cc)

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
    parser.add_argument('--fullscreen', action='store_true', help='Start in fullscreen mode')
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

def calculate_position_cc_values(x_center, y_center, img_width, img_height):
    # Convert x, y coordinates to MIDI CC values (0-127)
    x_cc_value = int((x_center / img_width) * 127)
    y_cc_value = int((y_center / img_height) * 127)
    
    # Ensure values are within MIDI range
    x_cc_value = max(0, min(127, x_cc_value))
    y_cc_value = max(0, min(127, y_cc_value))
    
    return x_cc_value, y_cc_value

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
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # If fullscreen flag is set, enable fullscreen
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Flag to track fullscreen state
    is_fullscreen = args.fullscreen
    
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
                    
                    # Calculate box area and center
                    box_area = (x2 - x1) * (y2 - y1)
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    
                    # Class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    # Only accept the allowed classes
                    if class_name not in allowed_classes:
                        continue

                    current_detected_classes.add(class_name)
                    
                    # Calculate MIDI parameters
                    note = calculate_note(class_name)
                    velocity = calculate_velocity(box_area, image_area)
                    
                    # Get control channels for this class
                    x_cc, y_cc = get_control_channels(class_name)
                    
                    # Calculate CC values for x and y positions
                    x_pos_value, y_pos_value = calculate_position_cc_values(
                        x_center, y_center, image_width, image_height
                    )
                    
                    port.send(Message('note_on', note=note, velocity=velocity))
                    # if class_name not in prev_detected_classes:
                    #     print(f"MIDI Note ON: Class={class_name}, Note={note}, Velocity={velocity}")
                    
                    # Send CC messages for X and Y positions
                    port.send(Message('control_change', control=x_cc, value=x_pos_value))
                    port.send(Message('control_change', control=y_cc, value=y_pos_value))
                    print(f"MIDI CC: Class={class_name}, X(CC{x_cc})={x_pos_value}, Y(CC{y_cc})={y_pos_value}")
                    
                    # Confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    
                    # Get color for this class
                    color = class_colors[cls]
                    
                    # Put box in cam with class-specific color
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center point
                    cv2.circle(img, (x_center, y_center), 5, (0, 255, 0), -1)
                    
                    # Create label with class name, confidence and MIDI info
                    label = f"{class_name}: {confidence:.2f} | Note: {note}, Vel: {velocity}"
                    pos_label = f"X: {x_pos_value}(CC{x_cc}), Y: {y_pos_value}(CC{y_cc})"
                    
                    # Calculate text size for better positioning
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    (pos_width, pos_height), baseline = cv2.getTextSize(
                        pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw background rectangle for text (main label)
                    cv2.rectangle(
                        img, 
                        (x1, y1 - text_height - 5), 
                        (x1 + text_width, y1), 
                        color, 
                        -1
                    )
                    
                    # Draw text with white color for better visibility (main label)
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
                    
                    # Draw background rectangle for position text
                    cv2.rectangle(
                        img, 
                        (x1, y2), 
                        (x1 + pos_width, y2 + pos_height + 5), 
                        color, 
                        -1
                    )
                    
                    # Draw position text
                    cv2.putText(
                        img, 
                        pos_label, 
                        (x1, y2 + pos_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1, 
                        cv2.LINE_AA
                    )
            
            # Send MIDI note_off for objects that disappeared
            for class_name in prev_detected_classes - current_detected_classes:
                note = calculate_note(class_name)
                port.send(Message('note_off', note=note, velocity=0))
                print(f"MIDI Note OFF: Class={class_name}, Note={note}")
            
            # Update previous detections
            prev_detected_classes = current_detected_classes
            
            # Show frame
            cv2.imshow(window_name, img)
            
            # Check for keypresses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('f'):  # Toggle fullscreen on 'f' key
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    except KeyboardInterrupt:
        print("\nStopping detection and MIDI output")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Turn off any remaining notes
        for class_name in prev_detected_classes:
            note = calculate_note(class_name)
            port.send(Message('note_off', note=note, velocity=0))
        
        # Close resources
        port.close()
        print("MIDI port closed")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()