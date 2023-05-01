#!/usr/bin/env python3
"""
File:    main.py
Author:  Tauno Erik
Started: 01.05.2023
Edited:  01.05.2023
"""

import argparse
import sys
from ultralytics import YOLO  # pip install ultralytics
import cv2
import math
import cvzone
from datetime import datetime
from liiklus_const import *

# Settings
APPNAME = "Liiklus"
LANG = EST # or ENG
MODEL_FILE = './models/yolov8l.pt'
CONF_TH = 0.6
WIDTH = 1280
HEIGHT = 720

is_active = True # while loop

# All model class names
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

detect_names = [ "person", "bicycle", "car",
                 "motorbike", "bus", "truck",
                 "dog", "horse", "cat",
                 "bird"]

colors = [(199,55,255), (56,56,255), (134,219,61),
          (168,153,44),(151,157,255), (52,147,26),
          (31,112,255), (29,178,255), (49,210,207),
          (10,249,72), (23,204,146)]

counter_line_l = [0, 0, 0, 0]
counter_line_r = [0, 0, 0, 0]


def main(argv):
  parser = argparse.ArgumentParser()

  is_webcam = True

  # Optional arguments:
  parser.add_argument("-v","--video", type=str, help="Input video file.")

  args = parser.parse_args()

  if args.video:
     is_webcam = False

  model = YOLO(MODEL_FILE)
  model.fuse()  # Mida teeb?

  cap = cv2.VideoCapture(0)  # webcam
  cap.set(3, WIDTH)
  cap.set(4, HEIGHT)

  if not cap.isOpened():
    raise IOError("Cannot open webcam")

  while is_active:
    is_cap, frame = cap.read()

    if is_cap:
      results = model(frame, stream=True)

      for r in results:
        boxes = r.boxes

        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          w, h = x2-x1, y2-y1

          conf = math.ceil((box.conf[0]*100))/100 # round
          class_id = int(box.cls[0])
          class_name = model.names[class_id]

          if class_name in detect_names:
            color_index = detect_names.index(class_name)
            cv2.rectangle(frame, (x1,y1), (x2,y2), colors[color_index], 3)
            cvzone.putTextRect(frame, 
                               f'{class_name}:{conf}',
                               (max(0,x1)+8, max(30, y1)-8),
                               colorB=colors[color_index]
                               )

    else:
       print("Bad cap")

    # Display video
    cv2.imshow(APPNAME, frame)
    
    # Exit window
    c = cv2.waitKey(1)
    if c == 27:        # ESC
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    print(f'Started: {start_time}')
    main(sys.argv)