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
from sort import *  # Download sort.py https://github.com/abewley/sort


# Settings
APPNAME = "Liiklus"
LANG = EST # or ENG
MODEL_FILE = './models/yolov8l.pt'
CONF_TH = 0.5
DATA_DIR = '/data/'
MASKS_DIR = '/masks/'
OUT_DIR = './output/'
WIDTH = 1280  #1920
HEIGHT = 720  #1080


is_save_video = False
is_active = True      # Main while loop

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
total_counter = []  # Kokku kõiki erinevaid

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
  #print(filepath)

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


def select_mask_file(cap):
  w = int(cap.get(3))
  h = int(cap.get(4))
  file_name = 'mask_'+str(w)+'_'+str(h)+'.png'
  filepath = os.getcwd() + MASKS_DIR + file_name  # Absolute path to input
  if os.path.isfile(filepath):
    mask = cv2.imread(filepath)
    return mask
  else:
    raise IOError(f"Cannot open mask file: {file_name}")

def generate_img(cap):
  w = int(cap.get(3))
  h = int(cap.get(4))
  img = np.zeros((w,h,3), dtype=np.uint8)
  return img

def main(argv):
  parser = argparse.ArgumentParser()
  start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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

  # Mask
  mask_file = select_mask_file(cap)

  # Store last good frmae
  last_frame = generate_img(cap) # emty image

  while is_active:
    is_cap, frame = cap.read()
    if is_cap:
      frame_region = cv2.bitwise_and(frame, mask_file)
      results = model(frame_region, stream=True)
      detections = np.empty((0, 5))  # Trackeri jaoks

      for r in results:
        boxes = r.boxes
        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          w, h = x2-x1, y2-y1

          conf = math.ceil((box.conf[0]*100))/100 # round
          class_id = int(box.cls[0])
          class_name = model.names[class_id]

          if conf > CONF_TH:
            if class_name in detect_names: 
              index = detect_names.index(class_name)
              # Track
              current_array = np.array([x1, y1, x2, y2, conf])
              detections = np.vstack((detections, current_array)) # add to tracker
              # Label
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
      # Update tracker
      results_tracker = tracker.update(detections)
      for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2-x1, y2-y1
        # Object center
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 5, (255,255,0), cv2.FILLED)
        # Write id
        #cvzone.putTextRect(frame,f'id{id}',(max(0,x1),max(30, y1)),scale=1.5,thickness=1,offset=5)

        # Count all cars
        if counter_l[1] < cy < counter_l[3] and counter_l[0]-20 < cx < counter_l[2]+20:
          if total_counter.count(id) == 0: # kui ei ole juba loetud
            total_counter.append(id) # count all cojeckts
            # Blink line color to red
            cv2.line(frame, (counter_l[0], counter_l[1]), (counter_l[2], counter_l[3]), (0,0,200), 3)

    else:
       print("Bad cap")

    # Total conter
    cvzone.putTextRect(frame,
                       f'Kokku: {len(total_counter)}',
                       (WIDTH-500, HEIGHT-50),
                       scale=4,
                       thickness=2,
                       offset=10,
                       colorR=(20,20,20)
                    )

    # Display video
    #time.sleep(0.1)
    try:
      cv2.imshow(APPNAME, frame)
      # store last good frmae
      last_frame = frame
    except:
      print("Error, bad frame")
      # Write last frame
      end_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
      savefile = OUT_DIR+start_time+'_'+end_time+'.jpg'
      cv2.imwrite(savefile, last_frame)
    
    # Exit window
    c = cv2.waitKey(1)
    if c == 27:        # ESC
      break

  
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    main(sys.argv)