import cv2
import numpy as np
from screen import screen
from math import trunc

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

classNames = {0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

prototxt = "model_data/MobileNetSSD_deploy.prototxt"
weights = "model_data/MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame_game = screen()
    
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
    cs = []
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            cs.append(contour)
    

    for c in cs:
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cv2.circle(frame[int(frame_height/2):frame_height, :], (cx, cy), 2, (0, 0, 255), -1)

    try:
        hands = cv2.bitwise_and(roi, roi, mask=mask)
        #cv2.imshow("hands", hands)
    except:
        pass
    
    
    frame_resized = cv2.resize(frame_game,(300,300))
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

            cv2.rectangle(frame_game, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0), 2)

            if class_id in classNames:
                label = classNames[class_id] + ": " + str(trunc(confidence * 100)) + "%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame_game, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame_game, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    

    cv2.imshow('Frame', frame)
    cv2.imshow('Screen', frame_game)
    
   

    if cv2.waitKey(1) == 27:
        break
        
    




cap.release()
cv2.destroyAllWindows()

