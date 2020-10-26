import cv2
import numpy as np

def nothing(x):
    pass


cameraPort = 0

cap = cv2.VideoCapture(cameraPort, cv2.CAP_DSHOW)

cv2.namedWindow("HSV Adjuster")
cv2.createTrackbar("LH", "HSV Adjuster", 0, 255, nothing)
cv2.createTrackbar("LS", "HSV Adjuster", 0, 255, nothing)
cv2.createTrackbar("LV", "HSV Adjuster", 0, 255, nothing)
cv2.createTrackbar("UH", "HSV Adjuster", 0, 255, nothing)
cv2.createTrackbar("US", "HSV Adjuster", 0, 255, nothing)
cv2.createTrackbar("UV", "HSV Adjuster", 0, 255, nothing)


cv2.setTrackbarPos("LH", "HSV Adjuster", 0)
cv2.setTrackbarPos("LS", "HSV Adjuster", 50)
cv2.setTrackbarPos("LV", "HSV Adjuster", 58)

cv2.setTrackbarPos("UH", "HSV Adjuster", 30)
cv2.setTrackbarPos("US", "HSV Adjuster", 255)
cv2.setTrackbarPos("UV", "HSV Adjuster", 255)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    lh = cv2.getTrackbarPos("LH", "HSV Adjuster")
    ls = cv2.getTrackbarPos("LS", "HSV Adjuster")
    lv = cv2.getTrackbarPos("LV", "HSV Adjuster")

    uh = cv2.getTrackbarPos("UH", "HSV Adjuster")
    us = cv2.getTrackbarPos("US", "HSV Adjuster")
    uv = cv2.getTrackbarPos("UV", "HSV Adjuster")


    lower_skin = np.array([lh, ls, lv], dtype = "uint8") 
    upper_skin = np.array([uh, us, uv], dtype = "uint8")

    frame_height, frame_width, channel = frame.shape
    cv2.line(frame, (0, int(frame_height/2)), (frame_width, int(frame_height/2)), (0, 255, 0), 1)
    
    roi = frame[int(frame_height/2):frame_height, :]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)

    

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    c = []
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            c.append(contour)
    #cv2.drawContours(frame[int(frame_height/2):frame_height, :], c, -1, (0, 255, 0), 1)

    try:
        hands = cv2.bitwise_and(roi, roi, mask=mask)
        cv2.imshow("hands", hands)
    except:
        pass
    

    

    cv2.imshow('Frame', frame)
    
   

    if cv2.waitKey(1) == 27:
        break




cap.release()
cv2.destroyAllWindows()
