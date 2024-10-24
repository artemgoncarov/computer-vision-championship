from Auxiliary.utils import *

# Init
car = Car()

cap = cv2.VideoCapture(0)

while True:
    status, frame = cap.read()
    if status:
        angle = turn(frame, coef=2)
        car.drive(100, angle, 0.2)
