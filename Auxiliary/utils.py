from time import sleep

import numpy as np
import pigpio
import cv2
import os


def calculate_angle(left_x, right_x, img_width):
    """Вычисляем угол поворота от -100 до 100."""
    lane_center = (left_x + right_x) / 2
    image_center = img_width / 2
    offset = lane_center - image_center
    max_offset = img_width / 2
    angle = (offset / max_offset) * 100
    return int(np.clip(angle, -100, 100))


def turn(frame, show=False, coef=1):
    """Вычисляет угол поворота на основе кадра."""

    # Параметры для перспективного преобразования и размера изображения
    src = np.float32([[20, 200], [350, 200], [275, 120], [85, 120]])
    img_size = [200, 360]
    dst = np.float32([[0, img_size[0]], [img_size[1], img_size[0]], [img_size[1], 0], [0, 0]])

    resized = cv2.resize(frame, (img_size[1], img_size[0]))

    # Извлекаем красный канал и создаем бинарное изображение
    r_channel = resized[:, :, 2]
    binary = np.zeros_like(r_channel)
    binary[(r_channel > 200)] = 1

    # Преобразуем в HLS и извлекаем S-канал
    hls = cv2.cvtColor(resized, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary2 = np.zeros_like(s_channel)
    binary2[(s_channel > 160)] = 1

    # Объединяем бинарные изображения
    allBinary = np.zeros_like(binary)
    allBinary[(binary == 1) | (binary2 == 1)] = 255

    # Перспективное преобразование
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(allBinary, M, (img_size[1], img_size[0]))

    if show:
        cv2.imshow("Binary", allBinary)
        cv2.imshow("Warped", warped)

    # Гистограмма для поиска полос
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

    midpoint = histogram.shape[0] // 2
    IndWhitestColumnsL = np.argmax(histogram[:midpoint])
    IndWhitestColumnsR = np.argmax(histogram[midpoint:]) + midpoint

    if IndWhitestColumnsL > 0 and IndWhitestColumnsR > 0:
        # Вычисляем угол поворота
        return calculate_angle(IndWhitestColumnsL, IndWhitestColumnsR, img_size[1]) * coef
    else:
        # Если не удалось найти полосы, возвращаем 0
        return 0


def scale_value(value, original_range, target_range):
    # Распаковка кортежей диапазонов
    original_min, original_max = original_range
    target_min, target_max = target_range

    # Преобразование значения из одного диапазона в другой
    scaled_value = target_min + (value - original_min) * (target_max - target_min) / (original_max - original_min)

    return scaled_value


class Car:
    MOTOR_PIN = 17
    SERVO_PIN = 18

    def __init__(self):
        os.system("sudo pigpiod")
        sleep(1)

        self.__pi = pigpio.pi()

        self.__pi.set_servo_pulsewidth(self.SERVO_PIN, 0)
        self.__pi.set_servo_pulsewidth(self.MOTOR_PIN, 0)
        sleep(1)

        self.drive(0, 0, 2)

    def drive(self, speed, angle=0, time=1):
        print(f"Drive for {time}s:")
        print(f" - speed: {speed}")
        print(f" - angle: {angle}\n")

        self.__control(*self.transform_data(speed, angle))
        sleep(time)

    def __control(self, speed, angle):
        self.__pi.set_servo_pulsewidth(self.MOTOR_PIN, speed)
        self.__pi.set_servo_pulsewidth(self.SERVO_PIN, int(11.5 * angle + 500))

    def exit(self):
        print("EXIT")

        self.__pi.set_servo_pulsewidth(self.MOTOR_PIN, 0)
        self.__pi.set_servo_pulsewidth(self.SERVO_PIN, 0)

    @staticmethod
    def transform_data(speed, angle):

        if speed < -100:
            speed = -100

        elif speed > 100:
            speed = 100

        if angle < -100:
            angle = -100

        elif angle > 100:
            angle = 100

        start_range = (-100, 100)
        speed_range = (1450, 1550)
        angle_range = (70, 110)

        speed = scale_value(speed, start_range, speed_range)
        angle = scale_value(-angle, start_range, angle_range)

        return int(speed), int(angle)
