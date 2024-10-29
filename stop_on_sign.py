import cv2
import numpy as np
# from Auxiliary.utils import *

# Задаем пороги для бинаризации
lower_threshold = np.array([0, 39, 143])
upper_threshold = np.array([130, 114, 185])

# Открываем видеопоток с веб-камеры
cap = cv2.VideoCapture(0)
# car = Car()

while True:
    # Читаем кадр из видеопотока
    ret, frame = cap.read()
    if not ret:
        break

    # Применяем бинаризацию по заданным порогам
    mask = cv2.inRange(frame, lower_threshold, upper_threshold)

    # Считаем количество пикселей, удовлетворяющих бинаризации
    pixel_sum = np.sum(mask) // 255  # Считаем количество белых пикселей

    # Проверяем условие
    # print(pixel_sum)
    if pixel_sum >= 2000:
        print("STOP")
        # car.drive(-100, 0, 0.5)
        # car.drive(0, 0, 1)
        # input()
    else:
        # car.drive(92, 0, 0.05)
        print('DRIVE')

    # Отображаем оригинальный кадр и маску
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Выходим из цикла при нажатии клавиши 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Освобождаем ресурсы
# cap.release()
# cv2.destroyAllWindows()