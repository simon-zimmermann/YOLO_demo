# Live demo of YOLO object detection using a webcam

There are two versions available: YOLOv7-tiny and YOLOv8.

YOLOv7-tiny is a smaller and more optimized/compressed version of YOLOv7, and can be used on very low-performance devices. However, its accuracy is quite poor.

YOLOv8 is a more accurate (and more recent) version of YOLO, but it requires a CUDA compatible GPU to run at resonable performance.

## Requirements

- Python 3.10 or later
- A CUDA compatible GPU (only for the yolov8.py)
- A webcam accessible by OpenCV

## Install

Make sure to have python 3 installed.
Create a virtual environment and activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the following packages using pip:
Install torch first to use the correct CUDA package.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python ultralytics
```
