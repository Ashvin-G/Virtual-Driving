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

cameraPort = 0
cap = cv2.VideoCapture(cameraPort, cv2.CAP_DSHOW)

while True:
    ret, webcam_frame = cap.read()

    webcam_frame = cv2.flip(webcam_frame, 1)

    draw_center_line(webcam_frame)
    mask = mask_roi(webcam_frame)
    contours = find_contours(mask)
    draw_centroids(contours, webcam_frame)

    game_frame = screen_stream()
    
    
    coords = detect_vehicles(game_frame)
    compute_distance(coords[0], coords[1], coords[2], coords[3], game_frame, game_frame.shape[0], game_frame.shape[1])
    
    cv2.imshow('webcam_frame', webcam_frame)
    #cv2.imshow('game_frame', game_frame)

    

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
