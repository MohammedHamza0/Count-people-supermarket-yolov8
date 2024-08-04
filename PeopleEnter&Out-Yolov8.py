from ultralytics import YOLO
import cv2
from tracker import *
import numpy as np
import pandas as pd
import os

os.chdir(r"F:\YOLO Projects\People tracking")

# Load the labels file once
with open("coco.txt", "r") as file:
    Labels = file.read().splitlines()

cap = cv2.VideoCapture("peoplecount1.mp4")

# Define the entry and exit areas
areaOut = np.array([[339, 542], [319, 545], [533, 667], [559, 657]])
areaEnter = np.array([[307, 553], [280, 555], [483, 675], [510, 671]])

peopleEnterId = {}
peopleOutId = {}

entering = set()
exiting = set()

tracker = Tracker()
targetLabel = ["person"]
model = YOLO("yolov8n.pt", task="detect")

def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Background rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  background_color, 
                  cv2.FILLED)
    # Border rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  border_color, 
                  thickness)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't read the frame")
        break
    
    frame = cv2.resize(frame, (1100, 700))
    prediction = model.predict(source=frame, conf=0.7)
    
    result = prediction[0].boxes.data.cpu().numpy()
    detection = pd.DataFrame(result)
    
    cv2.polylines(frame, [areaEnter], True, [255, 0, 0], 2)
    cv2.polylines(frame, [areaOut], True, [255, 0, 0], 2)
    
    lst = []
    for index, row in detection.iterrows():
        xmin = int(row[0])
        ymin = int(row[1])
        xmax = int(row[2])
        ymax = int(row[3])
        conf = float(row[4])
        Label = int(row[5])
        
        if Labels[Label] in targetLabel:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
            draw_text_with_background(frame, 
                                      f"{Labels[Label].capitalize()}, conf:{(conf)*100:0.2f}%", 
                                      (xmin, ymin - 10), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),  # White text
                                      (0, 0, 0),  # Black background
                                      (0, 0, 255))  # Red border
            
            lst.append([xmin, ymin, xmax, ymax])
    
    personsIDs = tracker.update(lst)
    for personID in personsIDs:
        x, y, w, h, ID = personID
        
        # Check if person is in the entry area
        if cv2.pointPolygonTest(areaEnter, ((w, h)), False) >= 0:
            peopleEnterId[ID] = (w, h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)
            cv2.polylines(frame, [areaEnter], True, [0, 255, 0], 2)
        
        # Check if person moves to the exit area after being in the entry area
        if ID in peopleEnterId and cv2.pointPolygonTest(areaOut, ((w, h)), False) >= 0:
            entering.add(ID)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)
            cv2.polylines(frame, [areaOut], True, [0, 255, 0], 2)
        
        # Check if person is in the exit area
        if cv2.pointPolygonTest(areaOut, ((w, h)), False) >= 0:
            peopleOutId[ID] = (w, h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)
            cv2.polylines(frame, [areaOut], True, [0, 255, 0], 2)
            
        # Check if person moves to the entry area after being in the exit area
        if ID in peopleOutId and cv2.pointPolygonTest(areaEnter, ((w, h)), False) >= 0:
            exiting.add(ID)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)
            cv2.polylines(frame, [areaEnter], True, [0, 255, 0], 2)
            
    countsEnterPersons = len(entering)
    countsOutPersons = len(exiting)
    
    # Improved display of counts
    draw_text_with_background(frame, 
                              f"Number of persons who entered the supermarket: {countsEnterPersons}", 
                              (30, 50), 
                              cv2.FONT_HERSHEY_COMPLEX, 
                              0.8, 
                              (255, 255, 255),  # White text
                              (0, 0, 0),  # Black background
                              (72, 61, 139))  # Border color
    
    draw_text_with_background(frame, 
                              f"Number of persons who left the supermarket: {countsOutPersons}", 
                              (30, 100), 
                              cv2.FONT_HERSHEY_COMPLEX, 
                              0.8, 
                              (255, 255, 255),  # White text
                              (0, 0, 0),  # Black background
                              (72, 61, 139))  # Border color
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()