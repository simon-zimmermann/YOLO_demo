import cv2
import time

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("yolov7_tiny.classes", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Read background image
bg = cv2.imread("yolov7_BG.png")

# Create window in fullscreen mode
cv2.namedWindow('ML_DEMO', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('ML_DEMO', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
(win_x, win_y, win_w, win_h) = cv2.getWindowImageRect('ML_DEMO')
win_w = 1920  # For some reason auto-detection of window size does not work
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
scaled_height = int(0.7545 * win_h)
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

    start_inference = time.time()
    classes, scores, boxes = model.detect(frame_cap, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end_inference = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        if score > 0.5:
            color = COLORS[int(classid) % len(COLORS)]
            if class_names[classid] == "refrigerator":
                continue
            label = "{}: {:.0%}".format(class_names[classid], score)
            cv2.rectangle(frame_cap, (int(box[0]), int(box[1]),
                          int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame_cap, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            end_drawing = time.time()

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
