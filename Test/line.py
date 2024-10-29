from Auxiliary.utils import *

cap = cv2.VideoCapture("output1280.avi")  # "output1280.avi"

while cv2.waitKey(1) != 27:
    status, frame = cap.read()
    if not status:
        print('END of video')
        break

    # Создаем экземпляр класса LaneDetection
    lane_detector = ImageTransform(frame)

    # Обрабатываем кадр
    angle = lane_detector.calculate_turn_angle()
    lane_detector.visualize_images(True)

    # Добавляем вывод информации о возможности поворота
    print(f'Angle: {angle}, Speed: {100 - abs(angle // 5)}')

# Освобождаем захват видео и закрываем окна
cap.release()
cv2.destroyAllWindows()
