# Traffic Sign Recognition Project

ELEC5308 Intelligent Information Engineering Practice - Group 26

This repository contains two projects:
- **Project 1:** YOLOv8 Traffic Sign Detection
- **Project 2:** Model Comparison (YOLOv8 vs ResNet50) with Interactive Web Demo

## Table of Contents

- [Quick Start - Web Demo](#quick-start---web-demo)
- [Installation and Environment Configuration](#installation-and-environment-configuration)
- [Dataset Preparation](#dataset-preparation)
- [Models](#models)
  - [YOLOv8 (Object Detection)](#yolov8-object-detection)
  - [ResNet50 (Image Classification)](#resnet50-image-classification)
- [Training](#training)
- [Web Demo Usage](#web-demo-usage)
- [Project Structure](#project-structure)
- [Results](#results)

---

## Quick Start - Web Demo

Launch the interactive web demo to test both models:

**Windows:**
```bash
launch_demo.bat
```

**Linux/Mac:**
```bash
pip install -r requirements_demo.txt
python traffic_sign_demo_final.py
```

Then open your browser at: `http://127.0.0.1:7861`


## Installation and Environment Configuration

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Miniconda/Anaconda

### Environment Setup

1. Check Conda installation:
```bash
conda --version
```

2. Check CUDA availability (if using GPU):
```bash
nvidia-smi
```

3. Create Python environment:
```bash
conda create -n traffic python=3.8
conda activate traffic
```

4. Install PyTorch (visit [pytorch.org](https://pytorch.org) for your specific CUDA version):
```bash
pip install torch torchvision
```

5. Install project dependencies:
```bash
pip install -r requirements_demo.txt
```

### Required Packages

- torch, torchvision - Deep learning framework
- ultralytics - YOLOv8 implementation
- gradio - Web demo interface
- opencv-python - Image processing
- numpy, pillow - Numerical operations
- plotly, pandas - Data visualization 

## Dataset Preparation

### Dataset Structure

The YOLO dataset format should be organized as follows:
```
datasets/
|
+-- traffic/
    +-- images/
    |   +-- train/
    |   +-- val/
    |
    +-- labels/
        +-- train/
        +-- val/
```

### Dataset Statistics

- **Total images:** 6,464
- **Training set:** 4,372 images
- **Validation set:** 2,092 images
- **Classes:** 58 traffic sign categories
- **Format:** YOLO format (class x_center y_center width height, normalized)

### Data Processing Tools

- `transform_to_yolo.py` - Convert original annotations to YOLO format
  - Original format: filename; width; height; x; y; w; h; class
  - YOLO format: class x_center y_center width height (normalized)
- `format.ipynb` - Convert various image formats to JPG
- `nums.ipynb` - Display class distribution statistics

### Known Dataset Characteristics

- **Severe class imbalance:** 223:1 ratio between most and least frequent classes
- **Format constraint:** Training data contains signs with numbers only (e.g., "30")
  - Real-world signs with units (e.g., "30 km/h") may have reduced detection accuracy
  - This is a dataset limitation, not a model architecture issue

## Models

### YOLOv8 (Object Detection)

**Why YOLOv8:**
- Higher accuracy than older versions (YOLOv3/v5)
- Faster inference speed and greater task scalability
- Compared to newer architectures (YOLOv9/Transformer), accuracy difference is minimal (3-8%)
- Fewer parameters, lower deployment cost, and lower GPU requirements
- Optimal balance between performance and efficiency

**Architecture:**
- Backbone: CSPDarknet53
- Neck: PAN-FPN
- Head: Decoupled detection head
- Input size: 768x768 pixels

**Model Configuration:**
```python
model = YOLO("yolov8s.pt")
```

**Performance:**
- Precision: 0.917
- Recall: 0.932
- mAP@0.5: **96.3%**
- Optimal confidence threshold: 0.685

**Source:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

### ResNet50 (Image Classification)

**Architecture:**
- Backbone: ResNet50 (pretrained)
- Custom FC layers: 2048 -> 1024 -> 58 classes
- Dropout: 0.5
- Input size: 224x224 pixels

**Model Definition:**
```python
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
```

**Use Case:**
- Pre-cropped single sign images
- Whole-image classification (no localization)
- Fast inference on preprocessed data

---

### Model Comparison

| Aspect | YOLOv8 | ResNet50 |
|--------|--------|----------|
| Task | Object Detection | Image Classification |
| Localization | Yes (bounding boxes) | No |
| Multi-object | Yes | No |
| Input Size | 768x768 | 224x224 |
| Best For | Road scenes | Cropped signs |
| Output | Boxes + classes + confidence | Single class + confidence |


## Training

### YOLOv8 Training

1. **Configure dataset in `traffic.yaml`:**
```yaml
path: ./datasets/traffic
train: images/train
val: images/val
nc: 58
names: ['5kilometer', '15kilometer', ...]
```

2. **Training code (`traffic_train.py`):**
```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="traffic.yaml",
    epochs=100,
    imgsz=768,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    mosaic=1.0
)
```

3. **Hyperparameter configuration:**
- Edit `ultralytics/cfg/default.yaml` for training parameters
- Edit `ultralytics/cfg/models/v8/yolov8.yaml` for network architecture

4. **Training output location:**
```
yolov8/runs/detect/train3/weights/
+-- best.pt         # Best model checkpoint
+-- last.pt         # Final epoch checkpoint
```

### ResNet50 Training

Training configuration:
- Epochs: 100
- Optimizer: Adam
- Learning rate: 0.001 with cosine decay
- Batch size: 32
- Augmentation: Standard torchvision transforms

Model checkpoint: `resnet/parameters_Resnet50.cpt`

---

## Web Demo Usage

The interactive web demo provides three main features:

### 1. YOLOv8 Detection Tab
- Upload image with traffic signs
- Adjust confidence threshold (default: 0.685)
- View detected signs with bounding boxes
- See detection details (class + confidence for each sign)

### 2. ResNet50 Classification Tab
- Upload single pre-cropped sign image
- Get classification result with confidence
- View top-3 predictions

### 3. Model Comparison Tab
- Upload same image to both models
- Compare detection vs classification approaches
- Understand the difference between localization and classification

### Demo Features
- Professional UI with custom styling
- Real-time inference
- Side-by-side model comparison
- Detailed model information and limitations

---

## Project Structure

```
yolo_traffic/
+-- datasets/                    # Dataset directory
+-- yolov8/                      # YOLOv8 training code and results
|   +-- runs/detect/train3/
|       +-- weights/best.pt      # Trained YOLOv8 model
+-- resnet/                      # ResNet50 code and model
|   +-- parameters_Resnet50.cpt  # Trained ResNet50 model
+-- traffic_sign_demo_final.py   # Web demo application
+-- requirements_demo.txt        # Demo dependencies
+-- launch_demo.bat              # Windows launcher
+-- traffic.yaml                 # Dataset configuration
+-- traffic_train.py             # YOLOv8 training script
+-- traffic_test.py              # Testing script
+-- transform_to_yolo.py         # Data format conversion
+-- PROJECT2_INNOVATION.md       # Project 2 documentation
+-- README.md                    # This file
```

---

## Results

### YOLOv8 Performance
- **Precision:** 0.917
- **Recall:** 0.932
- **mAP@0.5:** 96.3%
- **F1-Score:** 0.924

### Training Output
Results are saved in `yolov8/runs/detect/train3/`:
- Trained models (best.pt, last.pt)
- Confusion matrix
- F1-score curve
- Precision-Recall curve
- Training/validation loss curves
- Example detection results

### Testing
Test the trained model:
```python
from ultralytics import YOLO
model = YOLO("yolov8/runs/detect/train3/weights/best.pt")
results = model.predict(source="test_image.jpg", conf=0.685)
```

---

## Team

**Group 26**
- Shuhuai Wang
- Kun Chen
- Yanlong Feng
- Zihan Wang
- Boxi Chen

**Course:** ELEC5308 Intelligent Information Engineering Practice
**University:** The University of Sydney


