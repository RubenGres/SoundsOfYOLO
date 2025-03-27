import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
import gradio as gr
import time
import sys

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

def process_frame(frame, confidence_threshold, top_k, model_name):
    """Process a single frame with PyTorch object detection model"""
    print(f"[process_frame] Processing frame with model: {model_name}, confidence: {confidence_threshold}, top_k: {top_k}")
    start_time = time.time()
    
    # Load model (note: in a real app, you'd want to cache this)
    print(f"[process_frame] Loading detection model: {model_name}")
    detector = ObjectDetector(model_name)
    print(f"[process_frame] Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Run inference
    print(f"[process_frame] Running inference on frame of shape {frame.shape}")
    inference_start = time.time()
    detected_objects = detector.detect(frame, confidence_threshold)
    inference_time = time.time() - inference_start
    print(f"[process_frame] Inference completed in {inference_time:.4f} seconds")
    
    # Sort by confidence and take top k
    print(f"[process_frame] Sorting {len(detected_objects)} objects by confidence")
    detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
    top_objects = detected_objects[:top_k]
    print(f"[process_frame] Selected top {len(top_objects)} objects")
    
    # Convert numpy array to PIL Image for drawing
    # Ensure frame is in correct format for PIL (uint8)
    if isinstance(frame, np.ndarray):
        # Convert to uint8 if not already
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Convert RGB to BGR if needed (PIL uses RGB)
        if frame.shape[2] == 3:  # RGB or BGR format
            pil_image = Image.fromarray(frame)
        else:
            pil_image = Image.fromarray(frame)
    else:
        pil_image = frame
        
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw results on frame
    print(f"[process_frame] Drawing results on frame")
    for i, obj in enumerate(top_objects):
        # Extract bbox coordinates
        x1, y1, x2, y2 = obj["bbox"]
        
        print(f"[process_frame] Drawing object #{i}: {obj['class']} at [{x1},{y1},{x2},{y2}]")
        
        # Draw bounding box (green)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        
        # Draw center point (red)
        center_x, center_y = int(obj["center_x"]), int(obj["center_y"])
        draw.ellipse((center_x-5, center_y-5, center_x+5, center_y+5), fill=(255, 0, 0))
        
        # Create label text
        label = f"{obj['class']}: {obj['confidence']:.2f}"
        area_text = f"Area: {obj['area']:.0f}"
        pos_text = f"Pos: ({center_x}, {center_y})"
        
        # Put text above bounding box (blue)
        draw.text((x1, y1-50), label, fill=(0, 0, 255), font=font)
        draw.text((x1, y1-30), area_text, fill=(0, 0, 255), font=small_font)
        draw.text((x1, y1-10), pos_text, fill=(0, 0, 255), font=small_font)
    
    # Add object count (yellow)
    draw.text((10, 30), f"Objects: {len(top_objects)}/{len(detected_objects)}", fill=(255, 255, 0), font=font)
    
    # Convert back to numpy array for Gradio
    frame_array = np.array(pil_image)
    
    # Prepare output text
    output_text = f"Detected {len(detected_objects)} objects, showing top {len(top_objects)}:\n"
    for i, obj in enumerate(top_objects, 1):
        output_text += f"{i}. {obj['class']} ({obj['confidence']:.2f})\n"
    
    total_time = time.time() - start_time
    print(f"[process_frame] Frame processing completed in {total_time:.4f} seconds")
    print(f"[process_frame] Returning frame with {len(top_objects)}/{len(detected_objects)} objects drawn")
    
    return frame_array, output_text

def object_detector_webcam(confidence_threshold=0.5, top_k=5, model_name="faster_rcnn"):
    """
    Gradio interface function for webcam-based object detection
    """
    print(f"[object_detector_webcam] Initializing with confidence={confidence_threshold}, top_k={top_k}, model={model_name}")
    
    # Cache the model to avoid reloading it for every frame
    print(f"[object_detector_webcam] Loading detection model: {model_name}")
    model_load_start = time.time()
    detector = ObjectDetector(model_name)
    print(f"[object_detector_webcam] Model loaded in {time.time() - model_load_start:.2f} seconds")
    
    def process_webcam(image):
        print(f"\n[process_webcam] Received webcam frame: {'None' if image is None else f'shape={image.shape}'}")
        if image is None:
            print("[process_webcam] No image received, returning")
            return None, "No webcam feed available"
        
        frame_start = time.time()
        
        # Run inference
        print(f"[process_webcam] Running inference with model {model_name}")
        inference_start = time.time()
        detected_objects = detector.detect(image, confidence_threshold)
        inference_time = time.time() - inference_start
        print(f"[process_webcam] Inference completed in {inference_time:.4f} seconds")
        
        # Sort by confidence and take top k
        print(f"[process_webcam] Sorting {len(detected_objects)} objects by confidence")
        detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
        top_objects = detected_objects[:top_k]
        print(f"[process_webcam] Selected top {len(top_objects)} objects")
        
        # Create a copy of the image for drawing
        print("[process_webcam] Creating a copy of the image for annotations")
        
        # Convert numpy array to PIL Image
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
        print(f"[process_webcam] Drawing results on frame")
        for i, obj in enumerate(top_objects):
            # Extract bbox coordinates
            x1, y1, x2, y2 = obj["bbox"]
            
            print(f"[process_webcam] Drawing object #{i}: {obj['class']} at [{x1},{y1},{x2},{y2}]")
            
            # Draw bounding box (green)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            
            # Draw center point (red)
            center_x, center_y = int(obj["center_x"]), int(obj["center_y"])
            draw.ellipse((center_x-5, center_y-5, center_x+5, center_y+5), fill=(255, 0, 0))
            
            # Create label text
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            area_text = f"Area: {obj['area']:.0f}"
            pos_text = f"Pos: ({center_x}, {center_y})"
            
            # Put text above bounding box (blue)
            draw.text((x1, y1-50), label, fill=(0, 0, 255), font=font)
            draw.text((x1, y1-30), area_text, fill=(0, 0, 255), font=small_font)
            draw.text((x1, y1-10), pos_text, fill=(0, 0, 255), font=small_font)
        
        # Add object count (yellow)
        draw.text((10, 30), f"Objects: {len(top_objects)}/{len(detected_objects)}", fill=(255, 255, 0), font=font)
        
        # Convert back to numpy array for Gradio
        annotated_image = np.array(pil_image)
        
        # Prepare output text
        output_text = f"Detected {len(detected_objects)} objects, showing top {len(top_objects)}:\n"
        for i, obj in enumerate(top_objects, 1):
            output_text += f"{i}. {obj['class']} ({obj['confidence']:.2f})\n"
        
        frame_time = time.time() - frame_start
        print(f"[process_webcam] Frame processing completed in {frame_time:.4f} seconds")
        print(f"[process_webcam] Returning frame with {len(top_objects)}/{len(detected_objects)} objects drawn")
        
        return annotated_image, output_text
    
    return process_webcam

# Create the Gradio interface
def create_interface():
    print("[create_interface] Setting up Gradio interface components")
    with gr.Blocks(title="Object Detection") as interface:
        gr.Markdown("# Object Detection with Webcam")
        
        with gr.Row():
            with gr.Column(scale=2):
                print("[create_interface] Creating webcam input")
                webcam_input = gr.Image(sources=["webcam"], streaming=False, label="Webcam Feed")
                print("[create_interface] Creating detection output image")
                detection_output = gr.Image(label="Detection Results")
            
            with gr.Column(scale=1):
                print("[create_interface] Creating UI controls")
                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    label="Confidence Threshold"
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Number of Objects to Show"
                )
                
                # Create a dropdown for model selection
                model_choices = [(model_info["name"], model_name) for model_name, model_info in AVAILABLE_MODELS.items()]
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value="faster_rcnn",
                    label="Detection Model"
                )
                
                detection_text = gr.Textbox(label="Detection Results", lines=10)
                
                # Adding a large button to trigger processing
                detection_button = gr.Button("Run Detection", elem_id="run-detection-button", size="lg")
        
        # Explain the models
        gr.Markdown("""
        ## Available Models:
        - **Faster R-CNN**: Good balance of accuracy and speed
        - **SSD300**: Faster but less accurate
        - **RetinaNet**: Higher accuracy, slower processing
        """)
        
        # Create a processing function that uses the webcam feed and current slider values
        def process_feed(image, conf, top_k, model):
            print(f"[process_feed] Called with conf={conf}, top_k={top_k}, model={model}")
            print(f"[process_feed] Image received: {'None' if image is None else f'shape={image.shape}'}")
            
            if image is None:
                print("[process_feed] No image available, returning early")
                return None, "No webcam feed available"
            
            print("[process_feed] Calling process_frame function")
            result = process_frame(image, conf, top_k, model)
            print("[process_feed] Returning results from process_frame")
            return result
        
        print("[create_interface] Setting up event handlers")
        # Set up the event trigger for the button
        detection_button.click(
            process_feed,
            inputs=[webcam_input, confidence_slider, top_k_slider, model_dropdown],
            outputs=[detection_output, detection_text]
        )
        
        print("[create_interface] Gradio interface setup complete")
    
    return interface


if __name__ == "__main__":
    print(f"[main] Starting Object Detection Gradio Interface")
    print(f"[main] Python version: {sys.version}")
    
    try:
        # Check PyTorch version
        print(f"[main] PyTorch version: {torch.__version__}")
        print(f"[main] Torchvision version: {torchvision.__version__}")
        print(f"[main] CUDA available: {torch.cuda.is_available()}")
        
        # Check PIL version
        import PIL
        print(f"[main] PIL version: {PIL.__version__}")
        
        # Print installed packages for debugging
        import pkg_resources
        print(f"[main] Installed packages:")
        packages = [f"{d.project_name}=={d.version}" for d in pkg_resources.working_set]
        for package in sorted(packages):
            print(f"  - {package}")
    except Exception as e:
        print(f"[main] Error listing packages: {e}")
    
    interface = create_interface()
    interface.launch(share=True)  # Set share=False if you don't want a public link