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


allowed_classes = ["teddy bear", "bottle", "bowl", "cup"]

# "backpack", "umbrella", "suitcase", "sports ball",
# "skateboard", "bottle", "cup", "fork", "knife", "spoon",
# "bowl", "banana", "apple", "orange", "carrot", "bottle",
# "pottedplant"

# Function to calculate MIDI note values for each class
def calculate_note(class_name):
    name_to_note = {
        "cup": 60,
        "bowl": 62,
        "bottle": 64,
        "teddy bear": 65,
    }

    return name_to_note[class_name]

# Map class names to CC control channels for X and Y positions and size
def get_control_channels(class_name):
    # Calculate control channels dynamically based on index position in allowed_classes
    # Each class gets three control channels: X position, Y position, and Size
    # Start control channels at 1, 2, 3 for the first class, 4, 5, 6 for the second, etc.
    
    if class_name not in allowed_classes:
        # Fallback for safety, though this shouldn't happen due to filtering
        print(f"Warning: {class_name} not in allowed classes list")
        return (1, 2, 3)
        
    # Find the index of the class in the allowed_classes list
    class_index = allowed_classes.index(class_name)
    
    # Calculate control channels: 
    # First class gets CC 1,2,3; second gets 4,5,6; etc.
    x_cc = class_index * 3 + 1
    y_cc = class_index * 3 + 2
    size_cc = class_index * 3 + 3
    
    return (x_cc, y_cc, size_cc)

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
    parser.add_argument('--width', type=int, default=1280, help='Initial window width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Initial camera height (default: 720)')
    parser.add_argument('--persistence', type=int, default=3, help='Number of frames to keep a detection alive (default: 3)')
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

def calculate_size_cc_value(box_area, image_area):
    # Convert box area to a MIDI CC value (0-127)
    # Small objects will have low values, large objects will have high values
    percentage = (box_area / image_area) * 100
    
    # Scale percentage to MIDI CC value (0-127)
    # Apply a curve to make small changes more noticeable
    # Use a logarithmic scale to give more resolution to smaller objects
    if percentage > 0:
        # Log scale gives better resolution for small objects
        # Map from roughly 0.01% to 50% of screen area
        value = int(min(127, max(0, 42.5 * math.log10(percentage * 2 + 1))))
    else:
        value = 0
        
    return value

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
    
    # Try to get the highest resolution supported by the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get actual frame dimensions for area calculations
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_area = image_width * image_height
    aspect_ratio = image_width / image_height
    
    print(f"Camera resolution: {image_width}x{image_height}, Aspect ratio: {aspect_ratio:.2f}")
    
    # Model
    model = YOLO("yolo-Weights/yolov8n.pt")
    
    # Generate unique colors for each class
    class_colors = generate_unique_colors(len(classNames))
    
    # Store previously detected classes to track when objects appear/disappear
    prev_detected_classes = set()
    
    # Detection persistence tracker
    # Dictionary to keep track of how many frames each object has been missing
    # Key: class_name, Value: count of frames since last detection
    missing_frames = {}
    
    # Persistence threshold - number of frames to wait before sending note_off
    persistence_frames = args.persistence
    print(f"Detection persistence: {persistence_frames} frames")
    
    # Window name
    window_name = f'YOLO Detection with MIDI (Camera {camera_idx})'
    
    # Create window with aspect ratio preservation
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set initial window size with proper aspect ratio
    initial_width = 1280
    initial_height = int(initial_width / aspect_ratio)
    cv2.resizeWindow(window_name, initial_width, initial_height)
    
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
                    
                    # Get control channels for this class
                    x_cc, y_cc, size_cc = get_control_channels(class_name)
                    
                    # Calculate CC values for x, y positions and size
                    x_pos_value, y_pos_value = calculate_position_cc_values(
                        x_center, y_center, image_width, image_height
                    )
                    size_value = calculate_size_cc_value(box_area, image_area)
                    
                    # Send MIDI note_on message for newly detected objects
                    # Only send note_on once when the object first appears
                    if class_name not in prev_detected_classes:
                        # Use a fixed velocity for note_on - we'll use CC for size instead
                        port.send(Message('note_on', note=note, velocity=100))
                        print(f"MIDI Note ON: Class={class_name}, Note={note}")
                    
                    # Send CC messages for X and Y positions and size
                    port.send(Message('control_change', control=x_cc, value=x_pos_value))
                    port.send(Message('control_change', control=y_cc, value=y_pos_value))
                    port.send(Message('control_change', control=size_cc, value=size_value))
                    print(f"MIDI CC: Class={class_name}, X(CC{x_cc})={x_pos_value}, Y(CC{y_cc})={y_pos_value}, Size(CC{size_cc})={size_value}")
                    
                    # Confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    
                    # Get color for this class
                    color = class_colors[cls]
                    
                    # Put box in cam with class-specific color
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center point
                    cv2.circle(img, (x_center, y_center), 5, (0, 255, 0), -1)
                    
                    # Create label with class name, confidence and MIDI info
                    label = f"{class_name}: {confidence:.2f} | Note: {note}"
                    pos_label = f"X: {x_pos_value}(CC{x_cc}), Y: {y_pos_value}(CC{y_cc}), Size: {size_value}(CC{size_cc})"
                    
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
            
            # Update all currently detected classes to have 0 missing frames
            for class_name in current_detected_classes:
                missing_frames[class_name] = 0
            
            # Handle objects that disappeared
            # Instead of immediately sending note_off, increment missing frame count
            classes_to_remove = []
            
            for class_name in prev_detected_classes:
                if class_name not in current_detected_classes:
                    # Increment missing frames counter
                    missing_frames[class_name] = missing_frames.get(class_name, 0) + 1
                    
                    # If object has been missing for more than persistence_frames, remove it
                    if missing_frames[class_name] >= persistence_frames:
                        # Send note_off
                        note = calculate_note(class_name)
                        port.send(Message('note_off', note=note, velocity=0))
                        print(f"MIDI Note OFF: Class={class_name}, Note={note} (after {persistence_frames} missing frames)")
                        
                        # Mark for removal from prev_detected_classes
                        classes_to_remove.append(class_name)
                    else:
                        # Object is temporarily missing but still within persistence threshold
                        # Show in UI that the object is being kept alive
                        print(f"Keeping {class_name} alive: missing for {missing_frames[class_name]} frames")
            
            # Remove classes that have been missing for too long
            for class_name in classes_to_remove:
                prev_detected_classes.remove(class_name)
                # Also remove from missing_frames tracker
                if class_name in missing_frames:
                    del missing_frames[class_name]
            
            # Update previous detections with current detections
            prev_detected_classes.update(current_detected_classes)
            
            # Draw status of all tracked objects (including those being kept alive)
            status_y = 30
            cv2.putText(img, "Tracked Objects:", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            status_y += 25
            
            for idx, class_name in enumerate(sorted(prev_detected_classes)):
                missing_count = missing_frames.get(class_name, 0)
                if missing_count > 0:
                    # Yellow text for objects being kept alive but not currently detected
                    status_color = (0, 255, 255)
                    status_text = f"{class_name}: Keeping alive ({missing_count}/{persistence_frames})"
                else:
                    # Green text for actively detected objects
                    status_color = (0, 255, 0)
                    status_text = f"{class_name}: Active"
                
                cv2.putText(img, status_text, (10, status_y + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, status_color, 2)
            
            # Show frame (maintain aspect ratio)
            cv2.imshow(window_name, img)
            
            # Check for keypresses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('f'):  # Toggle fullscreen on 'f' key
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    # Save current screen dimensions to calculate proper fullscreen size
                    screen_w = cv2.getWindowImageRect(window_name)[2]
                    screen_h = cv2.getWindowImageRect(window_name)[3]
                    
                    # Set to fullscreen while maintaining aspect ratio
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    # Return to normal window with proper aspect ratio
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    # Reset window size with proper aspect ratio
                    resized_width = 1280
                    resized_height = int(resized_width / aspect_ratio)
                    cv2.resizeWindow(window_name, resized_width, resized_height)
    
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