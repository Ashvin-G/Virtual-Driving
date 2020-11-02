import cv2
import numpy as np
import pyautogui

from functions import draw_center_line
from functions import mask_roi
from functions import find_contours
from functions import draw_centroids
from functions import screen_stream
from functions import detect_vehicles
from functions import compute_distance
from functions import lane_region
from functions import draw_lane_lines
from functions import grab_screen


from directKeys import W, A, S, D, PressKey, ReleaseKey


# cameraPort = 0
# cap = cv2.VideoCapture(cameraPort, cv2.CAP_DSHOW)

while True:
    # ret, webcam_frame = cap.read()

    # webcam_frame = cv2.flip(webcam_frame, 1)

    # draw_center_line(webcam_frame)
    # mask, hands = mask_roi(webcam_frame)
    # contours = find_contours(mask)
    # draw_centroids(contours, webcam_frame)

    game_frame = grab_screen(region=(0, 91, 780, 600))
    
    
    #coords = detect_vehicles(game_frame)
    #compute_distance(coords[0], coords[1], coords[2], coords[3], game_frame, game_frame.shape[0], game_frame.shape[1])

    # edges = lane_region(game_frame)
    # draw_lane_lines(edges, game_frame)
    
    
    #cv2.imshow('webcam_frame', webcam_frame)
    #cv2.imshow('hands', hands)
    cv2.imshow('game_frame', game_frame)

    

    if cv2.waitKey(1) == 27:
        break

# cap.release()
cv2.destroyAllWindows()
