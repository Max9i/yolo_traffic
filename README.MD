# Traffic Sign Recognition Project

This repository contains multiple components related to **traffic sign dataset preparation, classification, and detection**.

## Repository Structure

### 📂 crawl/
- Contains scripts for **web crawling and dataset preparation**.  
- Main features:
  - Automatically download traffic sign images from the web  
  - Organize and save images by category  
  - Prepare datasets for classification and detection tasks  

### 📂 resnet/
- Implements a **traffic sign classification model** based on **PyTorch + ResNet50**.  
- Features:
  - Custom `fc` layer supporting 58 traffic sign classes  
  - Data augmentation (flip, rotation, translation, color jitter, etc.)  
  - Dropout to mitigate overfitting  
  - Training with Adam + CosineAnnealingLR  
  - TensorBoard logging and best checkpoint saving  
- Output: **Class label** for each input image (e.g., speed limit, no entry, stop sign).  

### 📂 yolov8/
- Implements a **traffic sign detection model** using **Ultralytics YOLOv8**.  
- Features:
  - Supports custom dataset (train/validation/test split)  
  - Detects multiple traffic sign classes with bounding boxes  
  - Outputs trained weights and detection results  
- Output: **Bounding boxes + labels + confidence scores** for traffic signs in images/videos.  

---

## How to Use
1. **crawl/**  
   - Run the crawler scripts to collect and organize traffic sign images.  

2. **resnet/**  
   - Navigate into the folder and run `resnet50_traffic.ipynb` for classification training and evaluation.  

3. **yolov8/**  
   - Navigate into the folder and run YOLOv8 training scripts (requires `ultralytics`) for detection tasks.  

---

## Dependencies
- Python 3.8+
- PyTorch / torchvision
- Ultralytics YOLOv8
- NumPy, Pillow, TensorBoard
- Requests / Selenium (for crawling)  