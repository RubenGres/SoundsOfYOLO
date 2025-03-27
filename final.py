import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from gradio_webrtc import WebRTC
import os
import time

# COCO classes used by PyTorch models
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Dictionary of available models
AVAILABLE_MODELS = {
    "faster_rcnn": {
        "name": "Faster R-CNN (Accurate)",
        "description": "Good balance of accuracy and speed"
    },
    "ssd": {
        "name": "SSD300 (Fast)",
        "description": "Faster but less accurate"
    },
    "retinanet": {
        "name": "RetinaNet (High Accuracy)",
        "description": "Higher accuracy, slower processing"
    }
}

class ObjectDetector:
    def __init__(self, model_name="faster_rcnn"):
        """Initialize the object detector with a PyTorch model"""
        self.model_name = model_name
        self.model_info = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS["faster_rcnn"])
        print(f"Loading model: {self.model_info['name']}")
        
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load appropriate model
        if model_name == "faster_rcnn":
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
            self.transforms = weights.transforms()
        elif model_name == "ssd":
            weights = SSD300_VGG16_Weights.DEFAULT
            self.model = ssd300_vgg16(weights=weights, box_score_thresh=0.5)
            self.transforms = weights.transforms()
        elif model_name == "retinanet":
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
            self.transforms = weights.transforms()
        else:
            # Default to Faster R-CNN
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
            self.transforms = weights.transforms()
        
        # Move model to appropriate device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Model loaded successfully: {self.model_info['name']}")
    
    def detect(self, image, confidence_threshold=0.5):
        """
        Detect objects in an image
        
        Args:
            image: NumPy array of shape [height, width, 3]
            confidence_threshold: Minimum confidence to include a detection
            
        Returns:
            List of detected objects with class, confidence, bbox
        """
        # Convert image to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            # If the image has 4 channels, take only the first 3 (RGB)
            if image.shape[2] == 4:
                image = image[:, :, :3]
            
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Input must be a NumPy array or PIL Image")
        
        # Apply transforms for the model
        tensor_image = self.transforms(pil_image).to(self.device)
        
        # Add batch dimension if not already present
        if tensor_image.dim() == 3:
            tensor_image = tensor_image.unsqueeze(0)
        
        # Get original image dimensions for coordinate conversion
        original_height, original_width = image.shape[0], image.shape[1]
        
        with torch.no_grad():
            # Perform detection
            predictions = self.model(tensor_image)
        
        detections = []
        for i in range(len(predictions[0]['boxes'])):
            confidence = predictions[0]['scores'][i].item()
            
            # Only keep detections above the confidence threshold
            if confidence >= confidence_threshold:
                box = predictions[0]['boxes'][i].detach().cpu().numpy()
                label_idx = predictions[0]['labels'][i].item()
                
                # Get class name
                class_name = COCO_CLASSES[label_idx]
                
                # Convert coordinates
                x1, y1, x2, y2 = box
                
                # Calculate center point and area
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                detections.append({
                    "class": class_name,
                    "confidence": float(confidence),
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "area": float(area),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return detections

# Global detector instance to avoid reloading for each frame
detector = None

def detection(image, conf_threshold=0.3, model_name="faster_rcnn"):
    """Process a single frame with object detection"""
    global detector
    
    # Initialize the detector if it doesn't exist
    if detector is None:
        detector = ObjectDetector(model_name)
    
    if image is None:
        return None
    
    print("before inference")

    # Run inference
    detected_objects = detector.detect(image, conf_threshold)
    
    # Sort by confidence
    detected_objects.sort(key=lambda x: x["confidence"], reverse=True)

    print(detect_objects)
    
    return image
    
    # Convert numpy array to PIL Image for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw results on frame
    for obj in detected_objects:
        # Extract bbox coordinates
        x1, y1, x2, y2 = obj["bbox"]
        
        # Draw bounding box (green)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        
        # Draw center point (red)
        center_x, center_y = int(obj["center_x"]), int(obj["center_y"])
        draw.ellipse((center_x-5, center_y-5, center_x+5, center_y+5), fill=(255, 0, 0))
        
        # Create label text
        label = f"{obj['class']}: {obj['confidence']:.2f}"
        
        # Put text above bounding box (blue)
        draw.text((x1, y1-20), label, fill=(0, 0, 255), font=font)
    
    # Add object count
    draw.text((10, 30), f"Objects: {len(detected_objects)}", fill=(255, 255, 0), font=font)
    
    # Convert back to numpy array
    result_image = np.array(pil_image)
    
    return result_image

# WebRTC configuration
rtc_configuration = None

# CSS for styling
css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Object Detection with WebRTC ⚡️
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Real-time object detection using PyTorch models
        </h3>
        """
    )
    
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream", rtc_configuration=rtc_configuration)
            
            with gr.Row():
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.30,
                )
                
                model_dropdown = gr.Dropdown(
                    choices=[(AVAILABLE_MODELS[k]["name"], k) for k in AVAILABLE_MODELS.keys()],
                    value="faster_rcnn",
                    label="Detection Model"
                )
        
        # Stream function with multiple inputs
        def stream_function(image, conf_threshold, model_name):
            return detection(image, conf_threshold, model_name)
        
        image.stream(
            fn=stream_function, 
            inputs=[image, conf_threshold, model_dropdown], 
            outputs=[image],
            time_limit=10
        )

if __name__ == "__main__":
    demo.launch()