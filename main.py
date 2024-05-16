import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time
import numpy as np

model = YOLO("yolov8n-seg.pt")  # segmentation model, see https://docs.ultralytics.com/tasks/segment/#models
names = model.model.names
cap = cv2.VideoCapture(0)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

scaled_height = 800
scaled_width = (w / h) * capture_height

bg = cv2.imread('ml_demo.jpg')

cv2.namedWindow('ML_DEMO', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('ML_DEMO', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#scaling_factor = 800 / 416


while True:
    (grabbed, frame) = cap.read()
    if not grabbed:
        exit()
    frame = cv2.flip(frame, 1)

    start = time.time()
    results = model.predict(frame)
    end = time.time()

    start_drawing = time.time()
    annotator = Annotator(frame, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               det_label=names[int(cls)])
    end_drawing = time.time()

    # fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    # cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    #resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    #blank_image = np.zeros((1080, 1920, 3), np.uint8)

#    frame = bg

    # height=1080
    # width = 1920
 #   start_y = int((h - resized_h) / 2)
  #  start_x = int((w - resized_w) / 2)
   # frame[start_y:start_y+resized_h, start_x:start_x+resized_w] = resized

    # frame[110:910, (width - 800)//2:((width-800)//2)+800] = frame

    #msg = "Zentrum Industrie 4.0 ML Demo"
   # cv2.putText(frame, msg, (((width)//2)-8*len(msg), 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ML_DEMO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# out.release()
cap.release()
cv2.destroyAllWindows()
