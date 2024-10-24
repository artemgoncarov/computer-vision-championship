from utils import *

cap = cv2.VideoCapture("../output1280.avi")

while cv2.waitKey(1) != 27:
    status, frame = cap.read()
    if not status:
        print('END of video')
        break

    angle = turn(cv2.flip(frame, 1), True, 2)

    # Добавляем вывод информации о возможности поворота
    print(f'Angle: {angle}')

# Освобождаем захват видео и закрываем окна
cap.release()
cv2.destroyAllWindows()
