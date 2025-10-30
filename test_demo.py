"""
Test script for Traffic Sign Detection Demo
Verifies that all dependencies are installed and the model can be loaded
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")

    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'gradio': 'Gradio',
        'ultralytics': 'Ultralytics YOLOv8',
        'PIL': 'Pillow'
    }

    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name} found")
        except ImportError:
            print(f"  ✗ {name} NOT found")
            missing.append(name)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements_demo.txt")
        return False

    print("\n✓ All required packages are installed")
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"  - Device count: {torch.cuda.device_count()}")
            print(f"  - Current device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ⚠ CUDA not available, will use CPU")
            print("  (This is OK but inference will be slower)")
            return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def test_model():
    """Test if model file exists and can be loaded"""
    print("\nTesting model file...")

    model_path = Path("yolov8/runs/detect/train3/weights/best.pt")

    if not model_path.exists():
        print(f"  ✗ Model file not found at {model_path}")
        print("  Please ensure the model is trained and saved at the correct location")
        return False

    print(f"  ✓ Model file found at {model_path}")

    try:
        from ultralytics import YOLO
        print("  Loading model...")
        model = YOLO(str(model_path))
        print("  ✓ Model loaded successfully")
        print(f"  - Classes: {len(model.names)}")
        return True
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False

def test_gradio():
    """Test if Gradio can be imported and is working"""
    print("\nTesting Gradio...")

    try:
        import gradio as gr
        print(f"  ✓ Gradio version: {gr.__version__}")
        return True
    except Exception as e:
        print(f"  ✗ Error with Gradio: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Traffic Sign Detection Demo - System Test")
    print("=" * 60)
    print()

    tests = [
        ("Package Imports", test_imports),
        ("CUDA Support", test_cuda),
        ("Model File", test_model),
        ("Gradio Interface", test_gradio)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append(False)
        print()

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for (name, _), result in zip(tests, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! You can now run the demo:")
        print("  python traffic_sign_demo.py")
        print("  or double-click launch_demo.bat (Windows)")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above before running the demo.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
