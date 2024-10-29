import cv2
import numpy as np
import time
from Auxiliary.utils import *
import os
import time


# Инициализация видеопотока
cap = cv2.VideoCapture(0)  # Замените 0 на путь к видеофайлу, если нужно
car = Car()

def detect_green_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_lower = (58, 194, 205)
    green_upper = (96, 255, 254)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    _, binary_mask = cv2.threshold(green_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

# Параметры для мигания и таймера
min_pixel_count = 50  # Порог зеленых пикселей для обнаружения света
blink_check_interval = 1  # Интервал проверки на мигание (в секундах)

last_blink_check_time = time.time()
green_detected_in_interval = False
non_green_detected_in_interval = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    binary_mask = detect_green_light(frame)
    green_count = cv2.countNonZero(binary_mask)

    # Проверяем, есть ли достаточное количество зеленых пикселей
    if green_count > min_pixel_count:
        green_detected_in_interval = True
    else:
        non_green_detected_in_interval = True

    # Проверяем, прошла ли 1 секунда с последней проверки
    current_time = time.time()
    if current_time - last_blink_check_time >= blink_check_interval:
        # Если в течение интервала был и зеленый, и его отсутствие, значит он мигает
        if green_detected_in_interval and non_green_detected_in_interval:
            print("Красный или мигающий свет. Остановка.")
            # Код для остановки
            # drone.stop()
        elif green_detected_in_interval:
            print("Зеленый свет!!! Можно двигаться.")
            car.drive(100, 90, 1)
            break
            # control(pi, ESC, 1550, STEER, 90)
            # time.sleep(0.5)
            # Код для движения
            # drone.move_forward()
        else:
            print("Нет зеленого света. Остановка.")
            # control(pi, ESC, 1500, STEER, 90)
            # time.sleep(0.5)
            # car.drive(0, 90, 1)
            # Код для остановки
            # drone.stop()

        # Сбрасываем значения для нового интервала
        last_blink_check_time = current_time
        green_detected_in_interval = False
        non_green_detected_in_interval = False

    # Отображение кадров
    # cv2.imshow('Original Frame', frame)
    # cv2.imshow('Binary Mask', binary_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()