import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time
import torch

# Make sure to actually use the GPU for inference
torch.cuda.set_device(0)  # use the GPU for inference
print("Using torch version: ", torch.__version__)
print("CUDA Support: ", torch.cuda.is_available())

# Model options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
model = YOLO("yolov8s-seg.pt")  # segmentation model, see https://docs.ultralytics.com/tasks/segment/#models
names = model.model.names  # use predefined class names

# Read background image
bg = cv2.imread("yolov8_BG.png")

# Create window in fullscreen mode
cv2.namedWindow('ML_DEMO', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('ML_DEMO', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# (win_x, win_y, win_w, win_h) = cv2.getWindowImageRect('ML_DEMO')
win_w = 1920  # Automatic window size detection does not work on the jetson nano
win_h = 1080
print("Window size: ", win_w, win_h)

cap = cv2.VideoCapture(0)  # Get capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)  # try to set some insane resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
# get actual maximum resolution
capture_w, capture_h = (int(cap.get(x))
                        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
print("Capture size: ", capture_w, capture_h)

# Calculate scaling for captured stream
# scale the webcam stream to a factor of the window height
# Factor determined by trial and error
scaled_height = int(0.754 * win_h)
scaled_width = int((capture_w / capture_h) * scaled_height)
print("Scaled size: ", scaled_width, scaled_height)

# Calculate start position for the scaled stream on the background image
start_y = int((win_h - scaled_height) / 2)
start_x = int((win_w - scaled_width) / 2)
print("Start position: ", start_x, start_y)

# Define the image size the inference engine shall use. Impacts performance!
# HAS to be a multiple of 32!
predict_w = 32 * 18
predict_h = 32 * 10
print("Inference image size: ", predict_w, predict_h)

while True:
    (grabbed, frame_cap) = cap.read()
    if not grabbed:
        exit()
    frame_cap = cv2.flip(frame_cap, 1)

    # AI inference, measure time, see https://docs.ultralytics.com/tasks/segment/#inference
    start_inference = time.time()
    results = model.predict(frame_cap, half=True, verbose=False, imgsz=(predict_w, predict_h))
    end_inference = time.time()

    # Draw stream, bg image and fps
    start_drawing = time.time()
    annotator = Annotator(frame_cap, line_width=2, pil=False)
    if results[0].masks is not None:
        # Classes of detected objects
        det_classes = results[0].boxes.cls.cpu().tolist()
        # Masks of detected objects
        det_masks = results[0].masks.xy
        # Confidence of detected objects
        det_confidence = results[0].boxes.conf.cpu().tolist()

        # Draw bounding boxes and labels for each detected object
        for mask, cls, confid in zip(det_masks, det_classes, det_confidence):
            label_str = "%s: %d%%" % (names[int(cls)], confid * 100)
            # Annotate image
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               det_label=label_str)

    # Create display frame, start with background image
    frame = cv2.resize(bg, (win_w, win_h), interpolation=cv2.INTER_AREA)
    # Resize stream to fit window
    resized = cv2.resize(frame_cap, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    # Add scaled stream to background image
    frame[start_y:start_y + scaled_height, start_x:start_x + scaled_width] = resized

    end_drawing = time.time()
    # Add FPS to display frame
    inference_time = (end_inference - start_inference) * 1000
    fps = 1 / (end_drawing - start_inference)  # overall FPS
    drawing_time = (end_drawing - start_drawing) * 1000
    fps_label = "FPS: %.2f (inference: %.2fms, drawing: %.2fms)" % (fps, inference_time, drawing_time)
    cv2.putText(img=frame,
                text=fps_label,
                org=(start_x + 5, start_y + 19),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)

    cv2.imshow("ML_DEMO", frame)

    # to quit, press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
