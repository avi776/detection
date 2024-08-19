import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import numpy as np

input_video_path = 'C:\\Users\\avipo\\PycharmProjects\\wrong way tracking\\ssvid.net - Drivers caught on camera while crossing red signals_1080p (online-video-cutter.com).mp4'
try:
    model = YOLO('yolov8s.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Load video
cap = cv2.VideoCapture(input_video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Load class list
try:
    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")
except Exception as e:
    print(f"Error loading class list: {e}")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = 'output.avi'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

count = 0
tracker = Tracker()

# New coordinates for area1
area1 = [
    (43, 157),
    (26, 236),
    (854, 218),
    (584, 158)
]

# Select ROI for color detection manually
ret, frame = cap.read()
if not ret:
    print("Error reading video frame")
    exit()

roi = cv2.selectROI("Select ROI for color detection", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI for color detection")

wup = {}
wrongway = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # Extract ROI for color detection
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]

    # Traffic light detection in the ROI
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskr = cv2.add(mask1, mask2)

    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    is_red_light = False
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))
        for i in r_circles[0, :]:
            cv2.circle(roi_frame, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
            cv2.putText(roi_frame, 'RED', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            is_red_light = True

    frame[y:y+h, x:x+w] = roi_frame

    if not is_red_light:
        continue

    try:
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
    except Exception as e:
        print(f"Error in object detection: {e}")
        continue

    bbox_list = []

    for _, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = class_list[d]

        # Check for cars, trucks, and vans within area1
        if 'car' in c or 'truck' in c or 'van' in c:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0:
                bbox_list.append([x1, y1, x2, y2])

    try:
        bbox_list = tracker.update(bbox_list)
    except Exception as e:
        print(f"Error in tracker update: {e}")
        continue

    for bbox in bbox_list:
        if len(bbox) != 5:
            print(f"Unexpected bbox format: {bbox}")
            continue

        x3, y3, x4, y4, id = bbox

        cx, cy = x3, y4

        result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)

        if result >= 0:
            wup[id] = (cx, cy)
            cv2.circle(frame, (cx, cy), 9, (255, 0, 0), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
            if id not in wrongway:
                wrongway.add(id)

        try:
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        except Exception as e:
            print(f"Error in cvzone.putTextRect: {e}")

    print(wup)
    w = len(wrongway)
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 255, 255), 2)
    cvzone.putTextRect(frame, f'signal violation: {w}', (60, 100), 1, 1)

    cv2.imshow("RGB", frame)

    # Write the frame to the output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
