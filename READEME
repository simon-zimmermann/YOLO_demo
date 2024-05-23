# Live demo of YOLO object detection using a webcam

There are two versions available: YOLOv7-tiny and YOLOv8.

YOLOv7-tiny is a smaller and more optimized/compressed version of YOLOv7, and can be used on very low-performance devices. However, its accuracy is quite poor.

YOLOv8 is a more accurate (and more recent) version of YOLO, but it requires a CUDA compatible GPU to run at resonable performance.
This version is tailored to run on a Jetson Nano, but it can be run on any other CUDA compatible GPU.

## Requirements

- Python 3.10 or later
- A CUDA compatible GPU (only for the yolov8.py)
- A webcam accessible by OpenCV

## Install (on a generic system)

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

## Install (on a Jetson Nano)

Stolen from [https://i7y.org/en/yolov8-on-jetson-nano/](https://i7y.org/en/yolov8-on-jetson-nano/)

Install required packages:

```bash
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip \
libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev
```

Clone the YOLOv8 repository.

```bash
cd ~
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
```

Update Python packages.

```bash
python3.8 -m pip install -U pip wheel gdown
```

Download and install the pre-built PyTorch, TorchVision package. This package was built using the method described in [this](https://i7y.org/en/building-jetson-nano-libraries-on-host-pc/) article. [This](https://i7y.org/en/jetson-nano-yolov5-with-csi-2-camera/) article also uses the pre-built package.

```bash
# pytorch 1.11.0
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
# torchvision 0.12.0
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV
python3.8 -m pip install torch-*.whl torchvision-*.whl
```

Install the Python package for YOLOv8.

```bash
python3.8 -m pip install .
```

You MAY have to reinstall matplotlib.

```bash
python3.8 -m pip install matplotlib --force-reinstall
```

And if you use a 16GB SD-card, you will have to delete some files to make space.

## Run

To run the YOLOv7-tiny version, use the following command:

```bash
python yolov7-tiny.py
```

To run the YOLOv8 version on a Jetson Nano using the above install method, use the following command:

```bash
python3.8 yolov8.py
```
