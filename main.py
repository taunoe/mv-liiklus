#!/usr/bin/env python3
"""
File:    main.py
Author:  Tauno Erik
Started: 01.05.2023
Edited:  02.05.2023
"""

import argparse
import sys
from ultralytics import YOLO  # pip install ultralytics
import cv2
import math
import cvzone
import os
import mimetypes
from datetime import datetime
from liiklus_const import *

# Settings
APPNAME = "Liiklus"
LANG = EST # or ENG
MODEL_FILE = './models/yolov8l.pt'
CONF_TH = 0.6
DATA_DIR = '/data/'
WIDTH = 1280
HEIGHT = 720

is_save_video = False
is_active = True      # Main while loop


def is_video(file):
  '''
  Returns: True or False
  '''
  if mimetypes.guess_type(file)[0].startswith('video'):
    return True
  return False


def process_file_input(filename):
  """
  Param video filename. Videos in DATA_DIR!
  Returns full path to video
  """
  filepath = os.getcwd() + DATA_DIR + filename  # Absolute path to input
  print(filepath)

  if os.path.isfile(filepath):
    if not is_video(filepath):
      print("Not a video!")
    else:
      return filepath
  else:
    print("{} - is not a file!".format(filename))

def calculate_count_line(cap, name):
  w = cap.get(3)
  h = cap.get(4)
  y1 = 50
  y2 = int(h-50)
  if name == 'right':
    x1 = int((w/2)+50)
    x2 = int((w/2)+50)
  if name == 'left':
    x1 = int((w/2)-50)
    x2 = int((w/2)-50)
  return [x1, y1, x2, y2]


def main(argv):
  parser = argparse.ArgumentParser()

  is_webcam = True

  # Optional arguments:
  parser.add_argument("-v","--video", type=str, help="Input video file name.")

  args = parser.parse_args()

  if args.video:
     is_webcam = False
     video_name = args.video
     video_path = process_file_input(video_name)
     print(video_path)

  model = YOLO(MODEL_FILE)
  model.fuse()  # Mida teeb?

  if is_webcam:
    # Open webcamera
    cap = cv2.VideoCapture(0)  # webcam
    #cap.set(3, WIDTH)
    #cap.set(4, HEIGHT)
    if not cap.isOpened():
      raise IOError("Cannot open webcam!")
  else:
    # Open video fail
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise IOError("Cannot open video!")
  
  # Loendus joon
  counter_l = calculate_count_line(cap, 'left')
  counter_r = calculate_count_line(cap, 'right')

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

          #conf = math.ceil((box.conf[0]*100))/100 # round
          class_id = int(box.cls[0])
          class_name = model.names[class_id]

          if class_name in detect_names: 
            index = detect_names.index(class_name)
            cv2.rectangle(frame, (x1,y1), (x2,y2), colors[index], 3)
            cvzone.putTextRect(frame, 
                               f'{lang_txt[LANG][index]}',
                               (max(0,x1)+8, max(30, y1)-8),
                               colorB=colors[index],
                               colorR=colors[index],
                               colorT=(10,10,10)
                               )
      # Draw counting lines
      cv2.line(frame, (counter_l[0], counter_l[1]), (counter_l[2], counter_l[3]), colors[10], 3)
      cv2.line(frame, (counter_r[0], counter_r[1]), (counter_r[2], counter_r[3]), colors[10], 3)
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