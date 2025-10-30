"""
Traffic Sign Detection Demo
A Gradio-based web interface for real-time traffic sign detection using YOLOv8
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

# Model path configuration
MODEL_PATH = "yolov8/runs/detect/train3/weights/best.pt"

# Load model
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print(f"Model loaded successfully! Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

def detect_traffic_signs(image, confidence_threshold=0.685):
    """
    Detect traffic signs in the input image

    Args:
        image: Input image (numpy array)
        confidence_threshold: Minimum confidence for detection (default: 0.685 based on report)

    Returns:
        Annotated image with bounding boxes and labels
    """
    if image is None:
        return None

    # Run inference
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        imgsz=768,  # Using 768 as specified in training
        device='0' if torch.cuda.is_available() else 'cpu'
    )

    # Get annotated image
    annotated_image = results[0].plot()

    # Convert BGR to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return annotated_image

def detect_with_stats(image, confidence_threshold=0.685):
    """
    Detect traffic signs and return both image and statistics

    Args:
        image: Input image (numpy array)
        confidence_threshold: Minimum confidence for detection

    Returns:
        Tuple of (annotated image, statistics text)
    """
    if image is None:
        return None, "No image provided"

    # Run inference
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        imgsz=768,
        device='0' if torch.cuda.is_available() else 'cpu'
    )

    # Get annotated image
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Extract statistics
    boxes = results[0].boxes
    num_detections = len(boxes)

    if num_detections == 0:
        stats_text = "No traffic signs detected"
    else:
        # Count detections by class
        class_counts = {}
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(confidence)

        # Build statistics text
        stats_text = f"Total detections: {num_detections}\n\n"
        stats_text += "Detected signs:\n"
        for class_name, confidences in sorted(class_counts.items()):
            avg_conf = np.mean(confidences)
            count = len(confidences)
            stats_text += f"  - {class_name}: {count} (avg conf: {avg_conf:.3f})\n"

    return annotated_image, stats_text

# Create Gradio interface
def create_demo():
    """Create and configure the Gradio demo interface"""

    with gr.Blocks(title="Traffic Sign Detection Demo - Group 26") as demo:
        gr.Markdown("""
        # Traffic Sign Detection System
        ### ELEC5308 Intelligent Information Engineering Practice - Project 1
        **Group 26** | YOLOv8s Model | mAP@0.5: 96.3%

        Upload an image or use your webcam to detect traffic signs in real-time.
        The model can recognize 58 different types of traffic signs.
        """)

        with gr.Tab("Image Upload"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload Image",
                        type="numpy"
                    )
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.685,
                        step=0.05,
                        label="Confidence Threshold (optimal: 0.685)"
                    )
                    detect_btn = gr.Button("Detect Traffic Signs", variant="primary")

                with gr.Column():
                    output_image = gr.Image(label="Detection Results")
                    output_stats = gr.Textbox(
                        label="Detection Statistics",
                        lines=10,
                        max_lines=20
                    )

            detect_btn.click(
                fn=detect_with_stats,
                inputs=[input_image, confidence_slider],
                outputs=[output_image, output_stats]
            )

        with gr.Tab("Webcam Detection"):
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(
                        label="Webcam",
                        sources=["webcam"],
                        type="numpy"
                    )
                    webcam_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.685,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    webcam_btn = gr.Button("Detect from Webcam", variant="primary")

                with gr.Column():
                    webcam_output = gr.Image(label="Detection Results")
                    webcam_stats = gr.Textbox(
                        label="Detection Statistics",
                        lines=10,
                        max_lines=20
                    )

            webcam_btn.click(
                fn=detect_with_stats,
                inputs=[webcam_input, webcam_confidence],
                outputs=[webcam_output, webcam_stats]
            )

        with gr.Tab("Batch Processing"):
            gr.Markdown("""
            ### Process Multiple Images
            Upload multiple images to process them in batch.
            """)
            batch_input = gr.Files(
                label="Upload Multiple Images",
                file_types=["image"]
            )
            batch_confidence = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.685,
                step=0.05,
                label="Confidence Threshold"
            )
            batch_btn = gr.Button("Process Batch", variant="primary")
            batch_gallery = gr.Gallery(
                label="Batch Results",
                columns=3,
                height="auto"
            )

            def process_batch(files, conf):
                if files is None or len(files) == 0:
                    return []

                results = []
                for file in files:
                    img = cv2.imread(file.name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    annotated = detect_traffic_signs(img, conf)
                    results.append(annotated)
                return results

            batch_btn.click(
                fn=process_batch,
                inputs=[batch_input, batch_confidence],
                outputs=batch_gallery
            )

        with gr.Tab("Model Information"):
            gr.Markdown("""
            ## Model Details

            - **Architecture**: YOLOv8s (Small)
            - **Input Size**: 768x768 pixels
            - **Classes**: 58 traffic sign types
            - **Performance Metrics**:
                - Precision: 0.917
                - Recall: 0.932
                - mAP@0.5: **96.3%**
                - Optimal Confidence Threshold: 0.685
                - Best F1 Score: 0.91

            ### Training Configuration
            - Epochs: 100
            - Optimizer: AdamW
            - Learning Rate: 0.01 with cosine annealing
            - Data Augmentation: Mosaic, Mixup, HSV adjustments, Flipping

            ### Dataset
            - Total Images: 6,164
            - Training Set: 4,170 images
            - Test Set: 1,994 images

            ### Supported Traffic Signs
            The model can detect the following categories:
            - Speed limit signs (5-90 km/h)
            - Prohibition signs (No entry, No turn, No overtaking, etc.)
            - Mandatory signs (Turn only, Keep left/right, etc.)
            - Warning signs (Pedestrian crossing, Road works, Curves, etc.)
            - Information signs (Roundabout, Stop, Give way, etc.)

            ### Known Limitations
            Through practical testing with this demo, we discovered important insights:

            **Dataset Format Constraints:**
            - Training signs typically show only numbers (e.g., "30" in a circle)
            - Signs with additional text like "km/h" or "km" may not be detected
            - This reflects dataset coverage rather than model capability

            **Implications:**
            - The model performs excellently on signs matching training format
            - Format variations not in training data may result in low confidence or missed detections
            - Highlights the importance of diverse datasets for real-world deployment

            **Recommendations:**
            - For best results, test with signs similar to training dataset style
            - Lower confidence threshold (0.3-0.5) for signs with format variations
            - Future work: expand dataset to include regional sign variations

            This demo enabled discovery of these practical limitations - demonstrating
            the value of interactive testing tools in AI deployment.

            For more details, please refer to the project report.
            """)

        gr.Markdown("""
        ---
        ### Project Information
        **Course**: ELEC5308 Intelligent Information Engineering Practice
        **University**: The University of Sydney
        **Team Members**: Shuhuai Wang, Kun Chen, Yanlong Feng, Zihan Wang, Boxi Chen
        """)

    return demo

if __name__ == "__main__":
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the trained model exists at the specified path.")
        exit(1)

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",  # Localhost access
        server_port=7860,
        show_error=True,
        inbrowser=True  # Automatically open browser
    )
