"""
Simple script to extract class names from YOLO models without using OpenCV.
This avoids the OpenCV dependency issues.
"""

import sys
from ultralytics.utils.downloads import attempt_download
import torch
import yaml
import json

def get_yolo_model_classes(model_name="yolov8n.pt"):
    """
    Extract class names from a YOLO model without loading the full model
    or using OpenCV dependencies.
    
    Args:
        model_name (str): The name of the YOLO model file
        
    Returns:
        dict: A dictionary mapping class IDs to class names
    """
    print(f"Attempting to extract class names from {model_name}")
    
    try:
        # Download the model if it doesn't exist
        model_path = attempt_download(model_name)
        print(f"Model located at: {model_path}")
        
        # Load just the model metadata using torch.load
        # This avoids loading the entire model architecture
        data = torch.load(model_path, map_location='cpu')
        
        # Extract class names - different models may store this differently
        if 'model' in data and hasattr(data['model'], 'names'):
            names = data['model'].names
            print(f"Found class names in model.names")
        elif 'names' in data:
            names = data['names']
            print(f"Found class names directly in data['names']")
        elif 'model' in data and isinstance(data['model'], dict) and 'names' in data['model']:
            names = data['model']['names']  
            print(f"Found class names in model dictionary")
        else:
            # Try to find a YAML file with the same name
            yaml_path = model_path.replace('.pt', '.yaml')
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        names = yaml_data['names']
                        print(f"Found class names in associated YAML file")
                    else:
                        names = None
            except FileNotFoundError:
                names = None
                
        # If we still don't have names, use COCO classes as fallback
        if names is None:
            print("Could not find class names, using COCO classes as fallback")
            # Standard COCO classes
            names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        
        # Create a dictionary mapping class IDs to names
        class_dict = {i: name for i, name in enumerate(names)}
        return class_dict
        
    except Exception as e:
        print(f"Error extracting class names: {e}")
        return None

def main():
    print(f"YOLO Classes Extractor")
    print(f"Python version: {sys.version}")
    
    # List of models to check
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    
    for model_name in models:
        print(f"\nExtracting classes from {model_name}:")
        classes = get_yolo_model_classes(model_name)
        if classes:
            print(f"Found {len(classes)} classes:")
            for class_id, class_name in classes.items():
                print(f"  {class_id}: {class_name}")
            
            # Save classes to JSON file
            output_file = f"{model_name.replace('.pt', '')}_classes.json"
            with open(output_file, 'w') as f:
                json.dump(classes, f, indent=2)
            print(f"Saved classes to {output_file}")
        else:
            print(f"Failed to extract classes from {model_name}")

if __name__ == "__main__":
    main()