from PIL import ImageGrab
import numpy as np
import cv2

def screen():
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 91, 716, 413)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        return screen
