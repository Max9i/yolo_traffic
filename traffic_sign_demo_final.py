"""
Traffic Sign Detection Demo - Final Version
Practical demo comparing YOLOv8 detection vs ResNet50 classification
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image

# Model paths
YOLO_MODEL_PATH = "yolov8/runs/detect/train3/weights/best.pt"
RESNET_MODEL_PATH = "resnet/parameters_Resnet50.cpt"

# Load YOLOv8 model
print("Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print(f"YOLOv8 loaded! Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ResNet50 model definition
class ResNet50Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50()
        self.net.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 58)
        )

    def forward(self, x):
        return self.net(x)

# Load ResNet50 model
print("Loading ResNet50 model...")
resnet_model = ResNet50Classifier()
resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location='cpu'))
resnet_model.eval()
if torch.cuda.is_available():
    resnet_model = resnet_model.cuda()
print("ResNet50 loaded!")

# Class names
CLASS_NAMES = [
    '5kilometer', '15kilometer', '30kilometer', '40kilometer',
    '50kilometer', '60kilometer', '70kilometer', '90kilometer',
    'No Left Turn or Straight Ahead', 'No Right Turn or Straight Ahead',
    'No Straight Ahead', 'No Left Turn', 'No Left or Right Turn',
    'No Right Turn', 'No Overtaking', 'No U-turn',
    'No Entry for Motor Vehicles', 'No Horn', 'End Speed Limit 40',
    'End Speed Limit 50', 'Turn Right or Go Straight Ahead', 'Ahead Only',
    'Left Turn Only', 'Left or Right Turn Only', 'Right Turn Only',
    'Keep Left', 'Keep Right', 'Roundabout', 'Motor Vehicles Only',
    'Sound Horn', 'Bicycles Only', 'U-turn Only', 'Divided Road Ahead',
    'Traffic Signals Ahead', 'General Warning', 'Pedestrian Crossing Ahead',
    'Cyclists Ahead', 'Children Crossing Ahead', 'Right Curve Ahead',
    'Left Curve Ahead', 'Steep Descent', 'Steep Ascent', 'SLOW',
    'Side Road Junction Ahead', 'Side Road Junction (left) Ahead',
    'Built-up Area Warning', 'Winding Road Ahead', 'train ahead',
    'Road Works Ahead', 'Continuous sharp turn sign', 'Railway level crossing',
    'Rear End Collision', 'STOP', 'No Entry for Vehicles', 'No Stopping',
    'No Entry', 'Give Way', 'Stop - Police'
]

# ResNet transforms
resnet_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_yolo(image, confidence=0.685):
    """YOLOv8 detection with bounding boxes"""
    if image is None:
        return None, "No image provided"

    results = yolo_model.predict(
        source=image,
        conf=confidence,
        imgsz=768,
        device='0' if torch.cuda.is_available() else 'cpu'
    )

    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    if len(boxes) == 0:
        return annotated, "No traffic signs detected"

    stats = f"Detected {len(boxes)} sign(s):\n\n"
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        stats += f"- {yolo_model.names[cls_id]}: {conf:.3f}\n"

    return annotated, stats

def classify_resnet(image):
    """ResNet50 classification (whole image)"""
    if image is None:
        return "No image provided", 0.0

    try:
        pil_img = Image.fromarray(image).convert('RGB')
        tensor = resnet_transform(pil_img).unsqueeze(0)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        with torch.no_grad():
            output = resnet_model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(dim=1)

        pred_class = CLASS_NAMES[pred.item()]
        confidence = conf.item()

        result = f"**Classification Result:**\n\n"
        result += f"Predicted Class: **{pred_class}**\n"
        result += f"Confidence: **{confidence:.3f}**\n\n"

        # Top 3 predictions
        top3_probs, top3_indices = probs.topk(3, dim=1)
        result += "**Top 3 Predictions:**\n"
        for i in range(3):
            idx = top3_indices[0][i].item()
            prob = top3_probs[0][i].item()
            result += f"{i+1}. {CLASS_NAMES[idx]}: {prob:.3f}\n"

        return result, confidence

    except Exception as e:
        return f"Error: {str(e)}", 0.0

def compare_models(image, yolo_conf=0.685):
    """Compare YOLOv8 and ResNet50 side by side"""
    if image is None:
        return None, "No image provided", "No image provided"

    # YOLOv8 detection
    yolo_results = yolo_model.predict(
        source=image,
        conf=yolo_conf,
        imgsz=768,
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    yolo_annotated = yolo_results[0].plot()
    yolo_annotated = cv2.cvtColor(yolo_annotated, cv2.COLOR_BGR2RGB)

    yolo_stats = "**YOLOv8 Detection:**\n"
    boxes = yolo_results[0].boxes
    if len(boxes) == 0:
        yolo_stats += "No signs detected"
    else:
        yolo_stats += f"Detected {len(boxes)} sign(s):\n"
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            yolo_stats += f"- {yolo_model.names[cls_id]}: {conf:.3f}\n"

    # ResNet50 classification
    resnet_result, _ = classify_resnet(image)

    return yolo_annotated, yolo_stats, resnet_result

def create_demo():
    """Create Gradio interface"""

    # Custom CSS
    custom_css = """
    footer {display: none !important;}
    .gradio-container {min-height: 0px !important;}
    """

    with gr.Blocks(
        title="Traffic Sign Recognition Demo",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:

        gr.Markdown("""
        # Traffic Sign Recognition System
        ### ELEC5308 Intelligent Information Engineering Practice - Project 2
        **Group 26** | Comparing YOLOv8 (Detection) vs ResNet50 (Classification)
        """)

        with gr.Tab("YOLOv8 Detection"):
            gr.Markdown("""
            **YOLOv8: Object Detection**
            - Detects and localizes multiple traffic signs
            - Draws bounding boxes around each sign
            - Returns position + class for each detection
            - **Performance:** mAP@0.5 = 96.3%
            """)

            with gr.Row():
                with gr.Column():
                    yolo_input = gr.Image(label="Upload Image", type="numpy")
                    yolo_conf = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.685, step=0.05,
                        label="Confidence Threshold (optimal: 0.685)"
                    )
                    yolo_btn = gr.Button("Detect Signs", variant="primary")

                with gr.Column():
                    yolo_output = gr.Image(label="Detection Results")
                    yolo_stats = gr.Textbox(label="Detection Details", lines=10)

            yolo_btn.click(
                fn=detect_yolo,
                inputs=[yolo_input, yolo_conf],
                outputs=[yolo_output, yolo_stats]
            )

        with gr.Tab("ResNet50 Classification"):
            gr.Markdown("""
            **ResNet50: Image Classification**
            - Classifies the entire image as one traffic sign
            - No localization - assumes one sign per image
            - Returns only the class label
            - **Best for:** Pre-cropped single sign images
            """)

            with gr.Row():
                with gr.Column():
                    resnet_input = gr.Image(label="Upload Image", type="numpy")
                    resnet_btn = gr.Button("Classify Sign", variant="primary")

                with gr.Column():
                    resnet_output = gr.Markdown(label="Classification Result")
                    resnet_conf = gr.Number(label="Confidence Score", precision=3)

            resnet_btn.click(
                fn=classify_resnet,
                inputs=resnet_input,
                outputs=[resnet_output, resnet_conf]
            )

        with gr.Tab("Model Comparison"):
            gr.Markdown("""
            **Compare Both Models Side-by-Side**

            Upload the same image to see how YOLOv8 (detection) and ResNet50 (classification) perform differently.

            **Key Differences:**
            - **YOLOv8:** Finds and locates multiple signs - Real-world road scenarios
            - **ResNet50:** Classifies whole image as one sign - Pre-cropped images only
            """)

            with gr.Row():
                with gr.Column():
                    compare_input = gr.Image(label="Upload Image", type="numpy")
                    compare_conf = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.685, step=0.05,
                        label="YOLOv8 Confidence"
                    )
                    compare_btn = gr.Button("Compare Models", variant="primary")

            with gr.Row():
                with gr.Column():
                    compare_yolo_img = gr.Image(label="YOLOv8 Detection")
                    compare_yolo_text = gr.Textbox(label="YOLOv8 Results", lines=8)

                with gr.Column():
                    gr.Markdown("**ResNet50 Classification**")
                    compare_resnet_text = gr.Markdown(label="ResNet50 Results")

            compare_btn.click(
                fn=compare_models,
                inputs=[compare_input, compare_conf],
                outputs=[compare_yolo_img, compare_yolo_text, compare_resnet_text]
            )

        with gr.Tab("About"):
            gr.Markdown("""
            ## Model Information

            ### YOLOv8s - Object Detection
            - **Architecture:** CSPDarknet53 backbone + PAN-FPN neck + Decoupled head
            - **Input Size:** 768x768 pixels
            - **Output:** Bounding boxes + class labels + confidence scores
            - **Metrics:** Precision 0.917, Recall 0.932, mAP@0.5 **96.3%**
            - **Use Case:** Real-world road scenes with multiple signs

            ### ResNet50 - Image Classification
            - **Architecture:** ResNet50 backbone + Custom FC layers (2048->1024->58)
            - **Input Size:** 224x224 pixels
            - **Output:** Single class label + confidence
            - **Use Case:** Pre-cropped single sign images

            ### Method Comparison

            | Aspect | YOLOv8 | ResNet50 |
            |--------|--------|----------|
            | Task | Detection | Classification |
            | Localization | Yes | No |
            | Multi-object | Yes | No |
            | Real-time | Yes | Yes |
            | Best for | Road scenes | Cropped signs |

            ### Known Limitations

            **WARNING - Dataset Format Constraint Discovered:**
            - Training data contains signs with numbers only (e.g., "30")
            - Real-world signs often include units (e.g., "30 km/h")
            - This format mismatch can cause detection failures
            - Example: 40km/h (260 samples) detected | 30km/h (80 samples) often missed

            **Recommendation:** For best results, use signs matching the training format,
            or lower the confidence threshold (0.3-0.5) for format variations.

            ---

            ### Training Configuration
            - **YOLOv8:** 100 epochs, AdamW optimizer, Cosine LR, Mosaic augmentation
            - **ResNet50:** 100 epochs, Adam optimizer, Cosine LR, Standard augmentation

            ### Dataset
            - Total: 6,464 images (4,372 train / 2,092 validation)
            - Classes: 58 traffic sign categories
            - Severe class imbalance: 223:1 ratio (most frequent vs least frequent)

            ---

            **Course:** ELEC5308 Intelligent Information Engineering Practice
            **University:** The University of Sydney
            **Team:** Shuhuai Wang, Kun Chen, Yanlong Feng, Zihan Wang, Boxi Chen
            """)

    return demo

if __name__ == "__main__":
    # Check models exist
    if not Path(YOLO_MODEL_PATH).exists():
        print(f"Error: YOLOv8 model not found at {YOLO_MODEL_PATH}")
        exit(1)
    if not Path(RESNET_MODEL_PATH).exists():
        print(f"Error: ResNet50 model not found at {RESNET_MODEL_PATH}")
        exit(1)

    demo = create_demo()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True,
        show_api=False
    )
