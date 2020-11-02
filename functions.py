import cv2
import numpy as np
import os
from os import path
from PIL import ImageGrab
from math import trunc
from net_config import *
from math import sqrt
from math import atan
from math import degrees
import win32gui, win32ui, win32con, win32api


def helpWindow(frame):
    help_window = np.ones_like(frame)
    help_window = help_window*255

    text = "* Place you fist in respective region. \n \n* For best result avoid similar skin colour interfering\nin the region.\n\n* Adjust Lower and Upper HSV such that\n fist's have maximum white area.\n\n* Press Esc to exit."
    y0, dy = 125, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(help_window, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.imshow('Help Window', help_window)



def grab_screen(region=None):
    #Function created by Frannecklp
    
    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.resize(img, (800, 600))

def nothing(x):
    pass

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

    lh = 0
    ls = 50
    lv = 58

    uh = 30
    us = 255
    uv = 255

    hsv = []
    if path.exists("hsv.txt"):
        file_hsv = open("hsv.txt", "r")
        lines = file_hsv.readlines()

        for line in lines:
            hsv.append(line.strip())

        lh = hsv[0]
        ls = hsv[1]
        lv = hsv[2]

        uh = hsv[3]
        us = hsv[4]
        uv = hsv[5]
        file_hsv.close()
        os.remove("hsv.txt")

    lower_skin = np.array([lh, ls, lv], dtype = "uint8") 
    upper_skin = np.array([uh, us, uv], dtype = "uint8")
    
    mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    hands = cv2.bitwise_and(roi, roi, mask=mask)

    
    return mask, hands

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
                 
def detect_vehicles(frame):
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
                        coords = xLeftTop, yLeftTop, xRightBottom, yRightBottom
                        return coords
    return (0, 0, 0, 0)
                
    



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
            cv2.putText(game_frame, "Vehicle on Left", (int(game_frame_width/2) - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif (angD_R > angD_L):
            cv2.putText(game_frame, "Vehicle on Right", (int(game_frame_width/2) - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            pass


def lane_region(game_frame):
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


def make_coordinates(game_frame, line_parameters):
    try:
        if line_parameters is not None:
            slope, intercept = line_parameters
            y1 = game_frame.shape[0]
            y2 = int(y1 * (3/5))
            x1 = int((y1 - intercept)/slope)
            x2 = int((y2 - intercept)/slope)


            cv2.line(game_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    except:
        pass

def average_slope_intercept(game_frame, lines):
    try:
        left_fit = []
        right_fit = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)

        left_line = make_coordinates(game_frame, left_fit_average)
        right_line = make_coordinates(game_frame, right_fit_average)
    except:
        pass

def draw_lane_lines(edges, game_frame):
    try:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 160, np.array([]), 40, 5)

        averaged_lines = average_slope_intercept(game_frame, lines)
    except:
        pass