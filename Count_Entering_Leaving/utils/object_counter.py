# Import All the Requried Libraries
import cv2
from collections import deque
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

class ObjectCounter:

    def __init__(self):
        self.line = [(0, 250), (1920, 250)]
        self.entering = {}
        self.leaving = {}
        self.data_deque = {}
    
    def classNames(self):
        cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", 
                          "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", 
                          "bench", "bird", "cat","dog", "horse", "sheep", "cow", "elephant", "bear", 
                          "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
                          "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
                          "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", 
                          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
                          "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", 
                          "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
                          "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        return cocoClassNames

    def intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def get_direction(self, point1, point2):
        direction_str = ""
        # calculate y axis direction
        if point1[1] > point2[1]:
            direction_str += "South"
        elif point1[1] < point2[1]:
            direction_str += "North"
        else:
            direction_str += ""
        return direction_str

    def compute_color_labels(self, label):
        if label == 0:  # person
            color = (85, 45, 255)
        elif label == 2:  # car
            color = (222, 82, 175)
        elif label == 3:  # Motorbike
            color = (0, 204, 255)
        elif label == 5:  # Bus
            color = (0, 149, 255)
        else:
            color = (200, 100, 0)
        return tuple(color)

    def initialize_deepsort(self):
        # Create the Deep SORT configuration object and load settings from the YAML file
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

        # Initialize the DeepSort tracker
        deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                            # min_confidence  parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
                            min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            # nms_max_overlap specifies the maximum allowed overlap between bounding boxes during non-maximum suppression (NMS)
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                            # max_iou_distance parameter defines the maximum intersection-over-union (IoU) distance between object detections
                            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            # Max_age: If an object's tracking ID is lost (i.e., the object is no longer detected), this parameter determines how many frames the tracker should wait before assigning a new id
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                            # nn_budget: It sets the budget for the nearest-neighbor search.
                            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True
                            )

        return deepsort

    def draw_boxes(self, frame, bbox_xyxy, identities=None, categories=None, offset=(0,0)):
        height, width, _ = frame.shape
        cv2.line(frame, self.line[0], self.line[1], (46, 162, 112),3)
        for key in list(self.data_deque):
          if key not in identities:
            self.data_deque.pop(key)

        for i, box in enumerate(bbox_xyxy):

            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            y1 += offset[0]
            x2 += offset[0]
            y2 += offset[0]
            # Find the center point of the bounding box
            center = int((x1+x2)/2), int((y1+y2)/2)
            cat = int(categories[i]) if categories is not None else 0
            color = self.compute_color_labels(cat)
            id = int(identities[i]) if identities is not  None else 0
            # Create new buffer for new object
            if id not in self.data_deque:
              self.data_deque[id] = deque(maxlen= 64)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            className = self.classNames()
            name = className[cat]
            label = str(id) + ":" + name
            text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + text_size[0], y1 - text_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(frame,center, 2, (0,255,0), cv2.FILLED)
            self.data_deque[id].appendleft(center)
            if len(self.data_deque[id]) >= 2:
                direction = self.get_direction(self.data_deque[id][0], self.data_deque[id][1])
                if self.intersect(self.data_deque[id][0], self.data_deque[id][1], self.line[0], self.line[1]):
                    cv2.line(frame, self.line[0], self.line[1], (255, 255, 255), 3)
                    if "South" in direction:
                        if name not in self.leaving:
                            self.leaving[name] = 1
                        else:
                            self.leaving[name] += 1
                    if "North" in direction:
                        if name not in self.entering:
                            self.entering[name] = 1
                        else:
                            self.entering[name] += 1

            # print("Entering Count", self.entering)
            # print("Leaving Count", self.leaving)
            # Display the Entering Count in the Top-Right Corner and Leaving Count in the Top Left Corner
            for idx, (key, value) in enumerate(self.entering.items()):
                entering_count = str(key) + ":" + str(value)
                cv2.line(frame, (width - 300, 25), (width, 25), [100,100, 255], 40)
                cv2.putText(frame, f"Entering Count", (width-300, 35), 0,1,[255,255,255], thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, (width - 180, 65 + (idx*40)), (width, 65 + (idx*40)), [100,100, 255], 30)
                cv2.putText(frame, entering_count, (width - 180, 75 + (idx*40)), 0,1, [255,255,255], thickness=2, lineType=cv2.LINE_AA)

            for idx, (key, value) in enumerate(self.leaving.items()):
                entering_count = str(key) + ":" + str(value)
                cv2.line(frame, (20, 25), (250, 25), [100,100, 255], 40)
                cv2.putText(frame, f"Leaving Count", (20, 35), 0,1,[255,255,255], thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, (20, 65 + (idx*40)), (150, 65 + (idx*40)), [100,100, 255], 30)
                cv2.putText(frame, entering_count, (20, 75 + (idx*40)), 0,1, [255,255,255], thickness=2, lineType=cv2.LINE_AA)

        return frame
