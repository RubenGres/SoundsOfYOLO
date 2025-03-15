from ultralytics import YOLO

def print_yolo_classes():
    """
    Print all class names that the YOLO model can recognize
    """
    # Load the YOLO model
    model = YOLO("yolov8n.pt")
    
    # Get the class names dictionary
    class_names = model.names
    
    # Print all classes with their indices
    print("YOLO Pre-trained Classes:")
    print("-" * 40)
    for idx, name in class_names.items():
        print(f"{idx}: {name}")
    print("-" * 40)
    print(f"Total Classes: {len(class_names)}")

if __name__ == "__main__":
    print_yolo_classes()