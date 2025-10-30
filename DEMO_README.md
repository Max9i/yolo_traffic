# Traffic Sign Detection Demo

An interactive web-based demo for real-time traffic sign detection using YOLOv8.

## Overview

This demo provides a user-friendly interface for testing our trained YOLOv8s traffic sign detection model. The model achieved **96.3% mAP@0.5** on the test dataset and can recognize 58 different types of traffic signs.

## Features

- **Image Upload**: Upload images for traffic sign detection
- **Webcam Detection**: Real-time detection using your webcam
- **Batch Processing**: Process multiple images at once
- **Adjustable Confidence**: Fine-tune detection threshold (optimal: 0.685)
- **Detection Statistics**: View detailed statistics for each detection
- **Model Information**: Complete model details and performance metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)

### Setup Steps

1. **Navigate to the project directory**
   ```bash
   cd yolo_traffic
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_demo.txt
   ```

   Or install manually:
   ```bash
   pip install torch torchvision ultralytics gradio opencv-python numpy pillow
   ```

3. **Verify model file exists**
   Make sure the trained model is located at:
   ```
   yolov8/runs/detect/train3/weights/best.pt
   ```

## Usage

### Starting the Demo

Run the demo application:
```bash
python traffic_sign_demo.py
```

The application will start a local web server. You should see output like:
```
Loading YOLOv8 model...
Model loaded successfully! Device: CUDA
Running on local URL:  http://127.0.0.1:7860
```

Open your web browser and navigate to the provided URL (default: `http://127.0.0.1:7860`)

### Using the Interface

#### 1. Image Upload Tab
- Click "Upload Image" to select an image file
- Adjust the confidence threshold slider (default: 0.685)
- Click "Detect Traffic Signs" to run detection
- View the annotated image and detection statistics

#### 2. Webcam Detection Tab
- Click "Webcam" to activate your camera
- Capture an image
- Adjust confidence threshold if needed
- Click "Detect from Webcam" to process
- View results and statistics

#### 3. Batch Processing Tab
- Upload multiple images at once
- Set confidence threshold
- Click "Process Batch"
- View all results in a gallery format

#### 4. Model Information Tab
- View detailed model specifications
- Check performance metrics
- Review supported traffic sign categories

### Confidence Threshold

The confidence threshold determines the minimum confidence score for a detection to be displayed:

- **Default (0.685)**: Optimal balance between precision and recall (F1=0.91)
- **Higher (0.8-0.9)**: Fewer detections, higher precision
- **Lower (0.4-0.6)**: More detections, may include false positives

## Model Performance

| Metric | Value |
|--------|-------|
| Precision | 0.917 |
| Recall | 0.932 |
| mAP@0.5 | **96.3%** |
| F1 Score | 0.91 |
| Input Size | 768x768 |
| Classes | 58 |

## Supported Traffic Signs

The model can detect 58 types of traffic signs including:

### Speed Limits
5, 15, 30, 40, 50, 60, 70, 90 km/h

### Prohibition Signs
- No Left/Right Turn
- No Entry
- No Overtaking
- No U-turn
- No Horn
- And more...

### Mandatory Signs
- Turn directions
- Keep Left/Right
- Roundabout
- Specific vehicle types

### Warning Signs
- Pedestrian/Children crossing
- Road works
- Curves and bends
- Railway crossing
- And more...

### Information Signs
- Stop
- Give Way
- Traffic signals ahead

## Troubleshooting

### Model Not Found Error
```
Error: Model file not found at yolov8/runs/detect/train3/weights/best.pt
```
**Solution**: Ensure the trained model exists at the specified path. You may need to train the model first using `traffic_train.py`.

### CUDA Out of Memory
**Solution**: The demo will automatically fall back to CPU if GPU memory is insufficient. You can also close other GPU-intensive applications.

### Slow Inference on CPU
**Solution**: CPU inference is slower than GPU. For faster processing:
- Use a smaller image resolution
- Process fewer images in batch mode
- Consider using a machine with GPU support

### Import Errors
**Solution**: Reinstall dependencies:
```bash
pip install --upgrade -r requirements_demo.txt
```

## Deployment Options

### Local Network Access
To allow access from other devices on your network, the demo is configured with:
```python
server_name="0.0.0.0"  # Allow external access
server_port=7860
```

Access from other devices using: `http://[YOUR_IP]:7860`

### Public Sharing
To create a temporary public URL (valid for 72 hours), modify `traffic_sign_demo.py`:
```python
demo.launch(share=True)
```

This will generate a public Gradio link you can share.

## Technical Details

### Architecture
- **Model**: YOLOv8s (Small variant)
- **Backbone**: CSPDarknet53
- **Neck**: PAN-FPN
- **Head**: Anchor-free decoupled head

### Training Configuration
- **Epochs**: 100
- **Optimizer**: AdamW
- **Image Size**: 768x768
- **Batch Size**: 16
- **Augmentation**: Mosaic, Mixup, HSV, Affine transformations

## Project Information

**Course**: ELEC5308 Intelligent Information Engineering Practice
**University**: The University of Sydney
**Project**: Traffic Sign Recognition (Project 1)

**Team Members - Group 26**:
- Shuhuai Wang (550145574)
- Kun Chen (490485686)
- Yanlong Feng (530738637)
- Zihan Wang (530732417)
- Boxi Chen (550462471)

## References

For detailed methodology, results, and analysis, please refer to:
- Project Report: `ELEC5308_project1_report_group26.pdf`
- Training Script: `yolov8/traffic_train.py`
- Dataset Config: `yolov8/traffic.yaml`

## License

This project is created for academic purposes as part of ELEC5308 coursework at The University of Sydney.

## Acknowledgments

- YOLOv8 by Ultralytics
- Gradio for the web interface framework
- ELEC5308 course staff for guidance and support
