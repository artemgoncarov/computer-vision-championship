from Auxiliary.utils import *

# Открываем видеофайл
cap = cv2.VideoCapture('output1280.avi')

# Проверяем, удалось ли открыть видеофайл
if not cap.isOpened():
    print('Error opening video file')
    exit()

# Основной цикл обработки кадров
while cv2.waitKey(1) != 27:  # Продолжаем, пока не нажата клавиша 'Esc'
    ret, frame = cap.read()
    if not ret:  # Проверяем, есть ли следующий кадр
        print('END of video')
        break

    # Изменяем размер кадра до заданных параметров
    resized = cv2.resize(frame, (img_size[1], img_size[0]))
    cv2.imshow('frame', resized)

    # Извлекаем красный канал
    r_channel = resized[:, :, 2]
    binary = (r_channel > binary_threshold_r).astype(np.uint8)

    # Переводим изображение в цветовое пространство HLS и извлекаем S-канал
    hls = cv2.cvtColor(resized, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary2 = (s_channel > binary_threshold_s).astype(np.uint8)

    # Объединяем два бинарных изображения
    allBinary = np.maximum(binary, binary2) * 255
    cv2.imshow("binary", allBinary)

    # Копируем бинарное изображение для отображения многоугольника
    allBinary_visual = allBinary.copy()
    cv2.polylines(allBinary_visual, [src.astype('int32')], True, 255)
    cv2.imshow('polygon', allBinary_visual)

    # Применяем перспективное преобразование
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(allBinary, M, (img_size[1], img_size[0]))
    cv2.imshow('warped', warped)

    # Создаем гистограмму для поиска полос
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

    # Определяем самые светлые области для левой и правой полосы
    midpoint = histogram.shape[0] // 2
    IndWhitestColumnsL = np.argmax(histogram[:midpoint])
    IndWhitestColumnsR = np.argmax(histogram[midpoint:]) + midpoint

    # Отображаем найденные полосы на изображении
    warped_visual = warped.copy()
    cv2.line(warped_visual, (IndWhitestColumnsL, 0), (IndWhitestColumnsL, warped_visual.shape[0]), 110, 2)
    cv2.line(warped_visual, (IndWhitestColumnsR, 0), (IndWhitestColumnsR, warped_visual.shape[0]), 110, 2)
    cv2.imshow("Whitestcolumns", warped_visual)

    # Настройки для поиска окон
    nwindows = 9  # Количество окон
    window_height = warped.shape[0] // nwindows
    window_half_width = 25  # Половина ширины окна

    XCenterLeftWindow = IndWhitestColumnsL
    XCenterRightWindow = IndWhitestColumnsR

    # Индексы пикселей для левой и правой полосы
    left_lane_inds = np.array([], dtype=np.int16)
    right_lane_inds = np.array([], dtype=np.int16)

    # Создаем изображение для отображения результатов
    out_img = np.dstack((warped, warped, warped))

    # Координаты ненулевых пикселей
    nonzero = warped.nonzero()
    WhitePixelIndY = np.array(nonzero[0])
    WhitePixelIndX = np.array(nonzero[1])

    # Поиск пикселей полосы по окнам
    for window in range(nwindows):
        win_y1 = warped.shape[0] - (window + 1) * window_height
        win_y2 = warped.shape[0] - window * window_height

        left_win_x1 = XCenterLeftWindow - window_half_width
        left_win_x2 = XCenterLeftWindow + window_half_width
        right_win_x1 = XCenterRightWindow - window_half_width
        right_win_x2 = XCenterRightWindow + window_half_width

        # Отображаем окна
        cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0))
        cv2.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 0, 50 + window * 21))
        cv2.imshow('windows', out_img)

        # Находим индексы белых пикселей в окнах
        good_left_inds = ((WhitePixelIndY >= win_y1) & (WhitePixelIndY <= win_y2) &
                          (WhitePixelIndX >= left_win_x1) & (WhitePixelIndX <= left_win_x2)).nonzero()[0]
        good_right_inds = ((WhitePixelIndY >= win_y1) & (WhitePixelIndY <= win_y2) &
                           (WhitePixelIndX >= right_win_x1) & (WhitePixelIndX <= right_win_x2)).nonzero()[0]

        left_lane_inds = np.concatenate((left_lane_inds, good_left_inds))
        right_lane_inds = np.concatenate((right_lane_inds, good_right_inds))

        # Если найдено достаточно пикселей, обновляем центр окна
        if len(good_left_inds) > 50:
            XCenterLeftWindow = np.int32(np.mean(WhitePixelIndX[left_lane_inds]))
        if len(good_right_inds) > 50:
            XCenterRightWindow = np.int32(np.mean(WhitePixelIndX[right_lane_inds]))

    # Отображаем найденные пиксели для полос
    out_img[WhitePixelIndY[left_lane_inds], WhitePixelIndX[left_lane_inds]] = [255, 0, 0]
    out_img[WhitePixelIndY[right_lane_inds], WhitePixelIndX[right_lane_inds]] = [0, 0, 255]

    cv2.imshow('LANe', out_img)

    # Полиномы для левой и правой полосы
    leftx = WhitePixelIndX[left_lane_inds]
    lefty = WhitePixelIndY[left_lane_inds]
    rightx = WhitePixelIndX[right_lane_inds]
    righty = WhitePixelIndY[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Рассчитываем полином для центральной линии
    center_fit = (left_fit + right_fit) / 2

    # Отображаем центральную линию
    for ver_ind in range(out_img.shape[0]):
        gor_ind = int(center_fit[0] * (ver_ind ** 2) + center_fit[1] * ver_ind + center_fit[2])
        cv2.circle(out_img, (gor_ind, ver_ind), 2, (255, 0, 255), 1)

    cv2.imshow('Center', out_img)

# Освобождаем захват видео и закрываем окна
cap.release()
cv2.destroyAllWindows()
