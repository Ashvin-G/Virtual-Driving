import cv2
import numpy as np
from PIL import ImageGrab
from math import trunc
from net_config import *

def screen_stream():
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 91, 716, 413)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        return screen

def draw_center_line(frame):
    frame_height, frame_width, channel = frame.shape
    cv2.line(frame, (0, int(frame_height/2)), (frame_width, int(frame_height/2)), (0, 255, 0), 1)

def mask_roi(frame):
    frame_height, frame_width, channel = frame.shape
    roi = frame[int(frame_height/2):frame_height, :]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 50, 58], dtype = "uint8") 
    upper_skin = np.array([30, 255, 255], dtype = "uint8")
    
    mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def find_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = []
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            cs.append(contour)
    return cs

def draw_centroids(contours, frame):
    frame_height, frame_width, channel = frame.shape
    for contour in contours:
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(frame[int(frame_height/2):frame_height, :], (cx, cy), 2, (0, 0, 255), -1)
                 
def detector(frame):
    frame_resized = cv2.resize(frame,(300,300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            class_id = int(detections[0, 0, i, 1])
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)

            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0

            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0), 2)

            if class_id in classNames:
                label = classNames[class_id] + ": " + str(trunc(confidence * 100)) + "%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
