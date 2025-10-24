
# Import Required Libraries
import cv2
import math
import torch
from ultralytics import YOLO
from utils.object_counter import ObjectCounter

# Initialize the app
objectCounter = ObjectCounter()
deepsort = objectCounter.initialize_deepsort()
model = YOLO("yolo11n.pt") # Choose our YOLO model
classes = [0,1,2,3,4,5,6,7] # Only care about these classes (from the COCO dataset)
cap = cv2.VideoCapture("resources/videos/video.mp4") # Create a Video Capture Object
count = 0 # Loop counter

while True:
    xywh_bbox = []  # [x, y, width, height]
    confs = []  # Confidence scores
    oids = []  # Object id's
    outputs = []

    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        count += 1
        print("Frame No:", count)
        results = model.predict(frame, classes=classes)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Top Left Corner: x1, y1 ---- Bottom Right Corner: x2, y2
                x1, y1, x2, y2 = box.xyxy[0] # coordinates
                # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                # Convert the floating point tensors into integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Find the Center Coordinates of the bounding box for each detected object
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                # Find the Width and Height of the Bounding Box
                bbox_width = abs(x1 - x2)
                bbox_height = abs(y1 - y2)
                xcycwh = [cx, cy, bbox_width, bbox_height]
                xywh_bbox.append(xcycwh)
                conf = math.ceil(box.conf[0]*100)/100
                confs.append(conf)
                classNameInt=int(box.cls[0])
                oids.append(classNameInt)
        xywhs = torch.tensor(xywh_bbox)
        confss = torch.tensor(confs)
        outputs = deepsort.update(xywhs, confss, oids, frame)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            objectCounter.draw_boxes(frame, bbox_xyxy, identities, object_id)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
