import cv2
import numpy as np
from screen import screen

while True:
    game_frame = screen()

    cv2.imshow('game_frame', game_frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()