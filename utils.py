from time import sleep
import os
import numpy as np
import pigpio
import cv2

def turn(binary_warped, coef=1.0):
    # Получаем размеры изображения
    height, width = binary_warped.shape

    # Определяем середину изображения
    midpoint = width // 2

    # Находим пиксели для левой и правой разметки
    left_half = binary_warped[:, :midpoint]
    right_half = binary_warped[:, midpoint:]

    # Найти позиции пикселей, которые принадлежат линиям разметки
    left_x = np.mean(np.nonzero(left_half)[1]) if np.any(left_half) else 0
    right_x = np.mean(np.nonzero(right_half)[1]) if np.any(right_half) else width

    # Пересчитываем координату правой разметки относительно всего изображения
    right_x += midpoint

    # Центр между найденными линиями разметки
    lane_center = (left_x + right_x) / 2

    # Вычисляем смещение от центра изображения
    center_offset = lane_center - midpoint

    # Нормализуем смещение в диапазон от -100 до 100
    turn_value = (center_offset / midpoint) * 100

    # Обрезаем значение, чтобы не выходило за пределы [-100, 100]
    turn_value = max(min(turn_value, 100), -100)

    return round(turn_value * coef)


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
