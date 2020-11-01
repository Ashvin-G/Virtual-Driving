import cv2
import numpy as np
from PIL import ImageGrab
from math import trunc
from net_config import *
from math import sqrt
from math import atan
from math import degrees

def screen_stream():
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 91, 780, 600)))
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
            if class_id == 7 or class_id == 6 or class_id == 19:
                flag = 1
                xLeftTop = int(detections[0, 0, i, 3] * cols) 
                yLeftTop = int(detections[0, 0, i, 4] * rows)
                xRightBottom   = int(detections[0, 0, i, 5] * cols)
                yRightBottom   = int(detections[0, 0, i, 6] * rows)

                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0

                xLeftTop = int(widthFactor * xLeftTop) 
                yLeftTop = int(heightFactor * yLeftTop)
                xRightBottom   = int(widthFactor * xRightBottom)
                yRightBottom   = int(heightFactor * yRightBottom)

                

                cv2.rectangle(frame, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom),
                            (0, 255, 0), 2)

                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(trunc(confidence * 100)) + "%"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftTop, labelSize[1])
                    cv2.rectangle(frame, (xLeftTop, yLeftTop - labelSize[1]),
                                        (xLeftTop + labelSize[0], yLeftTop + baseLine),
                                        (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftTop, yLeftTop),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    if flag == 1:
                        return xLeftTop, yLeftTop, xRightBottom, yRightBottom
    return 0, 0, 0, 0
                
    



def compute_distance(xLeftTop, yLeftTop, xRightBottom, yRightBottom, game_frame,  game_frame_height, game_frame_width):
    if xLeftTop != 0 and yLeftTop !=0 and xRightBottom !=0 and yRightBottom != 0:
        x_mid_bbox = int((xLeftTop + xRightBottom)/2)
        y_mid_bbox = int((yLeftTop + yRightBottom)/2)

        cv2.circle(game_frame, (x_mid_bbox, y_mid_bbox), 1, (0, 0, 255), -1)

        cv2.line(game_frame, (x_mid_bbox, y_mid_bbox), (0, game_frame_height), (0, 0, 255), 3)
        cv2.line(game_frame, (x_mid_bbox, y_mid_bbox), (game_frame_width, game_frame_height), (0, 0, 255), 3)
        cv2.line(game_frame, (x_mid_bbox, y_mid_bbox), (int(game_frame_width/2), game_frame_height), (0, 0, 255), 3)

        slope_L_m1 = (y_mid_bbox - game_frame_height)/(x_mid_bbox)
        slope_L_m2 = 0


        slope_R_m1 = (y_mid_bbox - game_frame_height)/(x_mid_bbox - game_frame_width)
        slope_R_m2 = 0

        angR_L = atan((slope_L_m1 - slope_L_m2)/(1 + slope_L_m1*slope_L_m2))
        angD_L = abs(round(degrees(angR_L)))

        angR_R = atan((slope_R_m1 - slope_R_m2)/(1 + slope_R_m1*slope_L_m2))
        angD_R = abs(round(degrees(angR_R)))

        
        if angD_L > angD_R:
            cv2.putText(game_frame, "Vehicle on Left", (int(game_frame_width/2), game_frame_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif (angD_R > angD_L):
            cv2.putText(game_frame, "Vehicle on Right", (int(game_frame_width/2), game_frame_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            pass



            


def lane(game_frame):
    game_frame_height, game_frame_width, channel = game_frame.shape
    mask = np.zeros_like(game_frame)
    vertices = np.array([[0, 410], [0, 320], [348, 290], [518, 290], [game_frame_width, 315], [game_frame_width, 395], [600, 350], [245, 350]])
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    masked = cv2.bitwise_and(game_frame, mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=200, threshold2=300)
    lower_vertices = np.array([[0, game_frame_height], [0, 410], [245, 350], [600, 350], [game_frame_width, 395], [game_frame_width, game_frame_height]])
    cv2.fillPoly(edges, [lower_vertices], (0, 0, 0))
    cv2.line(edges, (0, 320), (348, 290), (0, 0, 0), 2)
    cv2.line(edges, (348, 290), (518, 290), (0, 0, 0), 2)
    cv2.line(edges, (518, 290), (game_frame_width, 395), (0, 0, 0), 2)
    return edges