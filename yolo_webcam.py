import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
import time
import sys

def process_frame(frame, confidence_threshold, top_k, model_path):
    """Process a single frame with YOLO model"""
    print(f"[process_frame] Processing frame with model: {model_path}, confidence: {confidence_threshold}, top_k: {top_k}")
    start_time = time.time()
    
    # Load model (note: in a real app, you'd want to cache this)
    print(f"[process_frame] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print(f"[process_frame] Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Run inference
    print(f"[process_frame] Running inference on frame of shape {frame.shape}")
    inference_start = time.time()
    results = model(frame, conf=confidence_threshold)[0]
    inference_time = time.time() - inference_start
    print(f"[process_frame] Inference completed in {inference_time:.4f} seconds")
    
    # Process results
    print(f"[process_frame] Processing detection results")
    detected_objects = []
    
    for i, detection in enumerate(results.boxes.data.tolist()):
        # Extract data
        x1, y1, x2, y2, conf, class_id = detection
        class_name = results.names[int(class_id)]
        print(f"[process_frame] Detection #{i}: {class_name} (conf: {conf:.4f}) at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
        
        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate area
        area = (x2 - x1) * (y2 - y1)
        
        detected_objects.append({
            "class": class_name,
            "confidence": conf,
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "bbox": [x1, y1, x2, y2]
        })
    
    # Sort by confidence and take top k
    print(f"[process_frame] Sorting {len(detected_objects)} objects by confidence")
    detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
    top_objects = detected_objects[:top_k]
    print(f"[process_frame] Selected top {len(top_objects)} objects")
    
    # Draw results on frame
    print(f"[process_frame] Drawing results on frame")
    for i, obj in enumerate(top_objects):
        # Extract bbox coordinates
        x1, y1, x2, y2 = obj["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        print(f"[process_frame] Drawing object #{i}: {obj['class']} at [{x1},{y1},{x2},{y2}]")
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw center point
        center_x, center_y = int(obj["center_x"]), int(obj["center_y"])
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Create label text
        label = f"{obj['class']}: {obj['confidence']:.2f}"
        area_text = f"Area: {obj['area']:.0f}"
        pos_text = f"Pos: ({center_x}, {center_y})"
        
        # Put text above bounding box
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2)
        cv2.putText(frame, area_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2)
        cv2.putText(frame, pos_text, (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2)
    
    # Add object count
    cv2.putText(frame, f"Objects: {len(top_objects)}/{len(detected_objects)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Prepare output text
    output_text = f"Detected {len(detected_objects)} objects, showing top {len(top_objects)}:\n"
    for i, obj in enumerate(top_objects, 1):
        output_text += f"{i}. {obj['class']} ({obj['confidence']:.2f})\n"
    
    total_time = time.time() - start_time
    print(f"[process_frame] Frame processing completed in {total_time:.4f} seconds")
    print(f"[process_frame] Returning frame with {len(top_objects)}/{len(detected_objects)} objects drawn")
    
    return frame, output_text

def yolo_webcam(confidence_threshold=0.5, top_k=5, model_path="yolov8n.pt"):
    """
    Gradio interface function for webcam-based object detection
    """
    print(f"[yolo_webcam] Initializing with confidence={confidence_threshold}, top_k={top_k}, model={model_path}")
    
    # Cache the model to avoid reloading it for every frame
    print(f"[yolo_webcam] Loading YOLO model: {model_path}")
    model_load_start = time.time()
    model = YOLO(model_path)
    print(f"[yolo_webcam] Model loaded in {time.time() - model_load_start:.2f} seconds")
    
    def process_webcam(image):
        print(f"\n[process_webcam] Received webcam frame: {'None' if image is None else f'shape={image.shape}'}")
        if image is None:
            print("[process_webcam] No image received, returning")
            return None, "No webcam feed available"
        
        frame_start = time.time()
        
        # Run inference
        print(f"[process_webcam] Running inference with model {model_path}")
        inference_start = time.time()
        results = model(image, conf=confidence_threshold)[0]
        inference_time = time.time() - inference_start
        print(f"[process_webcam] Inference completed in {inference_time:.4f} seconds")
        
        # Process results
        print(f"[process_webcam] Processing detection results")
        detected_objects = []
        
        for i, detection in enumerate(results.boxes.data.tolist()):
            # Extract data
            x1, y1, x2, y2, conf, class_id = detection
            class_name = results.names[int(class_id)]
            print(f"[process_webcam] Detection #{i}: {class_name} (conf: {conf:.4f}) at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate area
            area = (x2 - x1) * (y2 - y1)
            
            detected_objects.append({
                "class": class_name,
                "confidence": conf,
                "center_x": center_x,
                "center_y": center_y,
                "area": area,
                "bbox": [x1, y1, x2, y2]
            })
        
        # Sort by confidence and take top k
        print(f"[process_webcam] Sorting {len(detected_objects)} objects by confidence")
        detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
        top_objects = detected_objects[:top_k]
        print(f"[process_webcam] Selected top {len(top_objects)} objects")
        
        # Create a copy of the image to draw on
        print("[process_webcam] Creating a copy of the image for annotations")
        annotated_image = image.copy()
        
        # Draw results on frame
        print(f"[process_webcam] Drawing results on frame")
        for i, obj in enumerate(top_objects):
            # Extract bbox coordinates
            x1, y1, x2, y2 = obj["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            print(f"[process_webcam] Drawing object #{i}: {obj['class']} at [{x1},{y1},{x2},{y2}]")
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = int(obj["center_x"]), int(obj["center_y"])
            cv2.circle(annotated_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Create label text
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            area_text = f"Area: {obj['area']:.0f}"
            pos_text = f"Pos: ({center_x}, {center_y})"
            
            # Put text above bounding box
            cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2)
            cv2.putText(annotated_image, area_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2)
            cv2.putText(annotated_image, pos_text, (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2)
        
        # Add object count
        cv2.putText(annotated_image, f"Objects: {len(top_objects)}/{len(detected_objects)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
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
    with gr.Blocks(title="YOLO Object Detection") as interface:
        gr.Markdown("# YOLO Object Detection with Webcam")
        
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
                model_dropdown = gr.Dropdown(
                    choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                    value="yolov8n.pt",
                    label="YOLOv8 Model"
                )
                detection_text = gr.Textbox(label="Detection Results", lines=10)
                
                # Adding a large button to trigger processing
                detection_button = gr.Button("Run Detection", elem_id="run-detection-button", size="lg")
        
        # Explain the models
        gr.Markdown("""...""")  # (Your explanation about models remains the same)
        
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
    print(f"[main] Starting YOLO Webcam Gradio Interface")
    print(f"[main] Python version: {sys.version}")
    print(f"[main] OpenCV version: {cv2.__version__}")
    
    try:
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
