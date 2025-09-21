import torch
from ultralytics import YOLO

# print(f"CUDA available: {torch.cuda.is_available()}")  # 检查CUDA是否可用
# print(f"CUDA device count: {torch.cuda.device_count()}")  # 检查可用GPU数量
# print(f"Current device: {torch.cuda.current_device()}")  # 当前使用的GPU索引

if __name__ == '__main__':
    model = YOLO(r"D:\python\yolov8\yolov8s.pt")
    model.train(
        data=r"D:\python\yolov8\traffic.yaml",
        epochs=1,
        imgsz=768,                  # 分辨率略升，细节更好
        batch=16,                   # 先不加，稳一点
        workers=4,
        device='0',
        optimizer='AdamW',
        lr0=0.01,                   # 仍按16基准
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=3,
        cos_lr=True,
        patience=25,
        close_mosaic=10,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.1
)