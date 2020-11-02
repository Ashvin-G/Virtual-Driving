import cv2
import time
import numpy as np
from functions import nothing

TIMER = int(5)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

flash = np.ones_like(frame)
flash = flash * 255

help_window = np.ones_like(frame)
help_window = help_window*255

text = "* Place you fist in respective region. \n \n* For best result avoid similar skin colour interfering\nin the region.\n\n* Adjust Lower and Upper HSV such that\n fist's have maximum white area.\n\n* Press Esc to exit."
y0, dy = 125, 30
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(help_window, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)


flag = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press 'q' to start Autocapture", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'h' for Help", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(125)

    if key == ord('h'):
        cv2.imshow('help', help_window)

    
    elif key == ord('q'):
        prev = time.time()

        while TIMER >=0:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, channel = frame.shape
            cv2.putText(frame, str(TIMER), (int(frame_width/2) - 70, 210), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 4)
            cv2.line(frame, (0, int(frame_height/2)), (frame_width, int(frame_height/2)), (0, 255, 0), 2)
            cv2.line(frame, (int(frame_width/2), int(frame_height/2)), (int(frame_width/2), (int(frame_height))), (0, 255, 0), 2)
            
            cv2.putText(frame, "LEFT FIST", (88, 350), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(frame, "RIGHT FIST", (408, 350), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            cv2.waitKey(125)

            cur = time.time()

            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1

        else:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, channel = frame.shape
            captured = cv2.addWeighted(frame, 0.3, flash, 0.7, 0)
            cv2.putText(captured, "CAPTURED", (int(frame_width/2) - 80, int(frame_height/2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
            cv2.imshow('frame', captured)

            cv2.waitKey(2000)
            flag = 1
            break
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()



if flag == 1:
    cv2.namedWindow("HSV_Adjuster")
    cv2.createTrackbar("LH", "HSV_Adjuster", 0, 255, nothing)
    cv2.createTrackbar("LS", "HSV_Adjuster", 0, 255, nothing)
    cv2.createTrackbar("LV", "HSV_Adjuster", 0, 255, nothing)

    cv2.setTrackbarPos("LH", "HSV_Adjuster", 0)
    cv2.setTrackbarPos("LS", "HSV_Adjuster", 10)
    cv2.setTrackbarPos("LV", "HSV_Adjuster", 60)

    cv2.createTrackbar("UH", "HSV_Adjuster", 0, 255, nothing)
    cv2.createTrackbar("US", "HSV_Adjuster", 0, 255, nothing)
    cv2.createTrackbar("UV", "HSV_Adjuster", 0, 255, nothing)

    cv2.setTrackbarPos("UH", "HSV_Adjuster", 20)
    cv2.setTrackbarPos("US", "HSV_Adjuster", 150)
    cv2.setTrackbarPos("UV", "HSV_Adjuster", 255)

    while True:
        roi = frame[int(frame_height/2):frame_height, :]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        

        lh = cv2.getTrackbarPos("LH", "HSV_Adjuster")
        ls = cv2.getTrackbarPos("LS", "HSV_Adjuster")
        lv = cv2.getTrackbarPos("LV", "HSV_Adjuster")

        uh = cv2.getTrackbarPos("UH", "HSV_Adjuster")
        us = cv2.getTrackbarPos("US", "HSV_Adjuster")
        uv = cv2.getTrackbarPos("UV", "HSV_Adjuster")

        lower_skin = np.array([lh, ls, lv], dtype = "uint8") 
        upper_skin = np.array([uh, us, uv], dtype = "uint8")


        mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)

        cv2.imshow('mask', mask)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
            
