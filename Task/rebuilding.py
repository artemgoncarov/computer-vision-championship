from Auxiliary.utils import *
from time import time

# Init
car = Car()

cap = cv2.VideoCapture(0)
time_start = time()

while time_start - time() > 1:
    status, frame = cap.read()
    if status:
        angle = turn(frame, coef=2)
        car.drive(100, angle, 0.2)

