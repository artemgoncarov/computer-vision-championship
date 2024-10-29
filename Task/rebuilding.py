from Auxiliary.utils import *
from time import time

# Init
car = Car()

cap = cv2.VideoCapture(0)
time_start = time()

while time_start - time() > 1:
    status, frame = cap.read()
    if status:
        # Создаем экземпляр класса LaneDetection
        lane_detector = ImageTransform(frame)

        # Обрабатываем кадр
        lane_detector.visualize_images(True)

