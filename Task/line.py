from Auxiliary.utils import *

# Init
car = Car()

cap = cv2.VideoCapture(0)

while True:
    status, frame = cap.read()
    if status:
        # Создаем экземпляр класса LaneDetection
        lane_detector = ImageTransform(frame)

        # Обрабатываем кадр
        angle = lane_detector.calculate_turn_angle()
        car.drive(100 - abs(angle // 10), angle, 0.05)
