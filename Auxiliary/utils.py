from time import sleep
import os

import pigpio
import cv2

from Auxiliary.params import *


def scale_value(value, original_range, target_range):
    # Распаковка кортежей диапазонов
    original_min, original_max = original_range
    target_min, target_max = target_range

    # Преобразование значения из одного диапазона в другой
    scaled_value = target_min + (value - original_min) * (target_max - target_min) / (original_max - original_min)

    return scaled_value


class ImageTransform:
    # Остаточное накопление
    residual_accumulation_WhitestColumns = None

    def __init__(self, frame):
        """
        Инициализация класса LaneDetector.

        Параметры:
            frame (ndarray): Исходный кадр изображения.
        """

        # Исходный кадр
        self.frame = frame

        # Инициализация атрибутов
        self.resized = None
        self.r_channel = None
        self.binary_r = None
        self.hls = None
        self.s_channel = None
        self.binary_s = None
        self.allBinary = None
        self.allBinary_visual = None
        self.M = None
        self.warped = None
        self.histogram = None
        self.left_fit = None
        self.right_fit = None
        self.center_fit = None
        self.turn_angle = None
        self.out_img = None
        self.warped_visual = None
        self.WhitePixelIndX = None
        self.WhitePixelIndY = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.y_eval = None

    def resize_frame(self):
        """
        Изменяет размер исходного кадра до заданных параметров.

        Возвращает:
            ndarray: Измененный по размеру кадр.
        """
        if self.resized is None:
            self.resized = cv2.resize(self.frame, (img_size[1], img_size[0]))
        return self.resized

    def compute_binary(self):
        """
        Преобразует изображение в бинарное, используя красный канал и S-канал в пространстве HLS.

        Возвращает:
            ndarray: Бинарное изображение.
        """
        if self.allBinary is None:
            if self.resized is None:
                self.resize_frame()
            # Извлекаем красный канал
            self.r_channel = self.resized[:, :, 2]
            self.binary_r = np.array(self.r_channel > binary_threshold_r).astype(np.uint8)

            # Переводим изображение в HLS и извлекаем S-канал
            self.hls = cv2.cvtColor(self.resized, cv2.COLOR_BGR2HLS)
            self.s_channel = self.hls[:, :, 2]
            self.binary_s = np.array(self.s_channel > binary_threshold_s).astype(np.uint8)

            # Объединяем бинарные изображения
            self.allBinary = np.zeros_like(self.binary_r)
            self.allBinary[(self.binary_r == 1) | (self.binary_s == 1)] = 255

            # Копируем для визуализации многоугольника
            self.allBinary_visual = self.allBinary.copy()
            cv2.polylines(self.allBinary_visual, [src.astype('int32')], True, 255)
        return self.allBinary

    def perspective_transform(self):
        """
        Применяет перспективное преобразование к бинарному изображению.

        Возвращает:
            ndarray: Трансформированное изображение.
        """
        if self.warped is None:
            if self.allBinary is None:
                self.compute_binary()
            self.M = cv2.getPerspectiveTransform(src, dst)
            self.warped = cv2.warpPerspective(self.allBinary, self.M, (img_size[1], img_size[0]))
        return self.warped

    def compute_histogram(self):
        """
        Вычисляет гистограмму нижней половины трансформированного изображения для поиска пиков линий.

        Возвращает:
            ndarray: Гистограмма изображения.
        """
        if self.histogram is None:
            if self.warped is None:
                self.perspective_transform()
            self.histogram = np.sum(self.warped[self.warped.shape[0] // 2:, :], axis=0)
        return self.histogram

    def detect_lane_lines(self):
        """
        Обнаруживает линии дороги с помощью метода скользящего окна и аппроксимирует их полиномом.

        Возвращает:
            tuple: Коэффициенты полиномов для левой и правой линии.
        """
        if self.left_fit is None or self.right_fit is None:
            if self.histogram is None:
                self.compute_histogram()

            midpoint = self.histogram.shape[0] // 2
            IndWhitestColumnsL = np.argmin(np.abs(self.histogram[:midpoint] - self.histogram[:midpoint].mean()))
            IndWhitestColumnsR = np.argmin(np.abs(self.histogram[midpoint:] - self.histogram[midpoint:].mean())) + midpoint

            if ImageTransform.residual_accumulation_WhitestColumns is None:
                ImageTransform.residual_accumulation_WhitestColumns = (IndWhitestColumnsL, IndWhitestColumnsR)
            else:
                IndWhitestColumnsL = int(IndWhitestColumnsL * residual_accumulation_WhitestColumns_coef +
                                         (1 - residual_accumulation_WhitestColumns_coef) *
                                         ImageTransform.residual_accumulation_WhitestColumns[0])

                IndWhitestColumnsR = int(IndWhitestColumnsR * residual_accumulation_WhitestColumns_coef +
                                         (1 - residual_accumulation_WhitestColumns_coef) *
                                         ImageTransform.residual_accumulation_WhitestColumns[1])

                ImageTransform.residual_accumulation_WhitestColumns = (IndWhitestColumnsL, IndWhitestColumnsR)

            # Визуализация найденных столбцов
            self.warped_visual = self.warped.copy()
            cv2.line(self.warped_visual, (IndWhitestColumnsL, 0),
                     (IndWhitestColumnsL, self.warped_visual.shape[0]), 110, 2)
            cv2.line(self.warped_visual, (IndWhitestColumnsR, 0),
                     (IndWhitestColumnsR, self.warped_visual.shape[0]), 110, 2)

            # Настройки скользящего окна
            nwindows = 9
            window_height = np.int32(self.warped.shape[0] / nwindows)
            window_half_width = 25

            XCenterLeftWindow = IndWhitestColumnsL
            XCenterRightWindow = IndWhitestColumnsR

            left_lane_inds = []
            right_lane_inds = []

            self.out_img = np.dstack((self.warped, self.warped, self.warped))

            # Индексы ненулевых пикселей
            nonzero = self.warped.nonzero()
            WhitePixelIndY = np.array(nonzero[0])
            WhitePixelIndX = np.array(nonzero[1])
            self.WhitePixelIndY = WhitePixelIndY
            self.WhitePixelIndX = WhitePixelIndX

            for window in range(nwindows):
                win_y_low = self.warped.shape[0] - (window + 1) * window_height
                win_y_high = self.warped.shape[0] - window * window_height

                win_xleft_low = XCenterLeftWindow - window_half_width
                win_xleft_high = XCenterLeftWindow + window_half_width
                win_xright_low = XCenterRightWindow - window_half_width
                win_xright_high = XCenterRightWindow + window_half_width

                # Отображение окон
                cv2.rectangle(self.out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (50 + window * 21, 0, 0), 2)
                cv2.rectangle(self.out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 0, 50 + window * 21), 2)

                good_left_inds = ((WhitePixelIndY >= win_y_low) & (WhitePixelIndY <= win_y_high) &
                                  (WhitePixelIndX >= win_xleft_low) & (WhitePixelIndX <= win_xleft_high)).nonzero()[0]
                good_right_inds = ((WhitePixelIndY >= win_y_low) & (WhitePixelIndY <= win_y_high) &
                                   (WhitePixelIndX >= win_xright_low) & (WhitePixelIndX <= win_xright_high)).nonzero()[
                    0]

                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # Обновление центра окна с использованием good_left_inds
                if len(good_left_inds) > 50:
                    XCenterLeftWindow = np.int32(np.mean(WhitePixelIndX[good_left_inds]))
                if len(good_right_inds) > 50:
                    XCenterRightWindow = np.int32(np.mean(WhitePixelIndX[good_right_inds]))

            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            self.left_lane_inds = left_lane_inds
            self.right_lane_inds = right_lane_inds

            if len(left_lane_inds) > 0 and len(right_lane_inds) > 0:
                # Извлечение позиций пикселей линий
                leftx = WhitePixelIndX[left_lane_inds]
                lefty = WhitePixelIndY[left_lane_inds]
                rightx = WhitePixelIndX[right_lane_inds]
                righty = WhitePixelIndY[right_lane_inds]

                # Отображение найденных пикселей
                self.out_img[WhitePixelIndY[left_lane_inds], WhitePixelIndX[left_lane_inds]] = [255, 0, 0]
                self.out_img[WhitePixelIndY[right_lane_inds], WhitePixelIndX[right_lane_inds]] = [0, 0, 255]

                # Аппроксимация полиномом
                self.left_fit = np.polyfit(lefty, leftx, 2)
                self.right_fit = np.polyfit(righty, rightx, 2)

                self.ploty = np.linspace(0, self.out_img.shape[0] - 1, self.out_img.shape[0])
                self.y_eval = np.max(self.ploty)
            else:
                self.left_fit = None
                self.right_fit = None

    def compute_center_line(self):
        """
        Вычисляет центральную линию между левой и правой линиями.

        Возвращает:
            ndarray: Коэффициенты полинома центральной линии.
        """
        if self.center_fit is None:
            if self.left_fit is None or self.right_fit is None:
                self.detect_lane_lines()
            if self.left_fit is not None and self.right_fit is not None:
                self.center_fit = (self.left_fit + self.right_fit) / 2
            else:
                self.center_fit = None
        return self.center_fit

    def calculate_turn_angle(self):
        """
        Вычисляет угол поворота на основе наклона средней линии дороги относительно вертикали.
        Возвращает:
            int: Угол поворота от -100 до 100.
        """
        if self.left_fit is None or self.right_fit is None:
            self.detect_lane_lines()  # Детекция линий при необходимости

        if self.left_fit is not None and self.right_fit is not None:
            # Задаем две точки для расчета наклона средней линии
            y1 = img_size[0] * 0.75  # Точка ниже центра изображения
            y2 = img_size[0] - 1  # Нижняя точка изображения

            # Вычисление x координат левой и правой линии в этих точках
            left_x1 = self.left_fit[0] * y1 ** 2 + self.left_fit[1] * y1 + self.left_fit[2]
            left_x2 = self.left_fit[0] * y2 ** 2 + self.left_fit[1] * y2 + self.left_fit[2]
            right_x1 = self.right_fit[0] * y1 ** 2 + self.right_fit[1] * y1 + self.right_fit[2]
            right_x2 = self.right_fit[0] * y2 ** 2 + self.right_fit[1] * y2 + self.right_fit[2]

            # Вычисляем координаты средней линии в этих точках
            midpoint_x1 = (left_x1 + right_x1) / 2
            midpoint_x2 = (left_x2 + right_x2) / 2

            # Вычисляем наклон средней линии
            delta_x = midpoint_x2 - midpoint_x1
            delta_y = y2 - y1
            slope = delta_x / delta_y if delta_y != 0 else 0  # Защита от деления на ноль

            # Переводим наклон в угол относительно вертикали
            raw_angle = np.arctan(slope) * (180 / np.pi)  # Угол в градусах
            angle = int(np.clip(-raw_angle * angle_coef, -100, 100))
        else:
            # Если линии не найдены, угол поворота 0
            angle = 0

        return angle

    def visualize_images(self, all_image=False):
        """
        Визуализирует различные этапы обработки изображения.

        Параметры:
            all_image (bool): Если True, вычисляет и отображает все изображения.
        """
        # Проверка и вычисление необходимых изображений
        if self.resized is None and all_image:
            self.resize_frame()
        if self.allBinary is None and all_image:
            self.compute_binary()
        if self.warped is None and all_image:
            self.perspective_transform()
        if (self.left_fit is None or self.right_fit is None) and all_image:
            self.detect_lane_lines()
        if self.center_fit is None and all_image:
            self.compute_center_line()

        # Визуализация изображений
        if self.resized is not None:
            cv2.imshow('Resized Frame', self.resized)
        if self.allBinary is not None:
            cv2.imshow('Binary Image', self.allBinary)
        if self.allBinary_visual is not None:
            cv2.imshow('Binary with Polygon', self.allBinary_visual)
        if self.warped is not None:
            cv2.imshow('Warped Image', self.warped)
        if self.warped_visual is not None:
            cv2.imshow('Whitest Columns', self.warped_visual)
        if self.out_img is not None and self.center_fit is not None:
            # Отображение центральной линии
            ploty = self.ploty
            center_fitx = self.center_fit[0] * ploty ** 2 + self.center_fit[1] * ploty + self.center_fit[2]
            for i in range(len(ploty)):
                cv2.circle(self.out_img, (int(center_fitx[i]), int(ploty[i])), 2, (255, 0, 255), 1)
            cv2.imshow('Lane Lines and Center', self.out_img)
        elif self.out_img is not None:
            cv2.imshow('Lane Lines and Center', self.out_img)

        cv2.waitKey(1)


class Car:
    MOTOR_PIN = 17
    SERVO_PIN = 18

    def __init__(self):
        os.system("sudo pigpiod")
        sleep(1)

        self.__pi = pigpio.pi()
        self.__pi.set_servo_pulsewidth(self.MOTOR_PIN, 0)
        self.__pi.set_servo_pulsewidth(self.SERVO_PIN, 0)

        sleep(1)

        self.drive(0, 0, 2)

        input("Включите двигатели:")
        print()

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

    @classmethod
    def transform_data(cls, speed, angle):

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
