import cv2
import numpy as np


cameraPort = 0

cap = cv2.VideoCapture(cameraPort, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame_height, frame_width, channel = frame.shape
    cv2.line(frame, (0, int(frame_height/2)), (frame_width, int(frame_height/2)), (0, 255, 0), 1)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == 27:
        break




cap.release()
cv2.destroyAllWindows()