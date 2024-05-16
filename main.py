import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time

model = YOLO("yolov8n-seg.pt")  # segmentation model, see https://docs.ultralytics.com/tasks/segment/#models
names = model.model.names  # use predefined class names

# Read background image
bg = cv2.imread('YOLO demonstrator BG.png')

# Create window in fullscreen mode
cv2.namedWindow('ML_DEMO', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('ML_DEMO', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
(win_x, win_y, win_w, win_h) = cv2.getWindowImageRect('ML_DEMO')
print("Window size: ", win_w, win_h)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Get capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)  # try to set some insane resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
# get actual maximum resolution
capture_w, capture_h = (int(cap.get(x))
                        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
print("Capture size: ", capture_w, capture_h)

# Calculate scaling for captured stream
# scale the webcam stream to a factor of the window height
# Factor determined by trial and error
scaled_height = int(0.755 * win_h)
scaled_width = int((capture_w / capture_h) * scaled_height)
print("Scaled size: ", scaled_width, scaled_height)

# Calculate start position for the scaled stream on the background image
start_y = int((win_h - scaled_height) / 2)
start_x = int((win_w - scaled_width) / 2)
print("Start position: ", start_x, start_y)

while True:
    (grabbed, frame_cap) = cap.read()
    if not grabbed:
        exit()
    frame_cap = cv2.flip(frame_cap, 1)

    # AI inference, measure time, see https://docs.ultralytics.com/tasks/segment/#inference
    start_inference = time.time()
    results = model.predict(frame_cap)
    end_inference = time.time()

    # Draw stream, bg image and fps
    start_drawing = time.time()
    annotator = Annotator(frame_cap, line_width=2)
    if results[0].masks is not None:
        cls = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, cls):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               det_label=names[int(cls)])

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
