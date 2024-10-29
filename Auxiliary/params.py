import numpy as np

# Размер изображения после изменения (высота, ширина)
img_size = [200, 360]

distancing = 120
narrowing = 100

# Задаем исходные точки для перспективного преобразования (выборка региона интереса)
src = np.float32([[0, img_size[0]],
                  [img_size[1], img_size[0]],
                  [img_size[1] - narrowing, img_size[0] - distancing],
                  [narrowing, img_size[0] - distancing]]
                 )

# Задаем точки назначения для перспективного преобразования
dst = np.float32([[0, img_size[0]], [img_size[1], img_size[0]], [img_size[1], 0], [0, 0]])

# Коэффициент угла поворота
angle_coef = 4

# Порог бинаризации
binary_threshold_s = 170
binary_threshold_r = 200

# Коэффициент остатка накопления линий дороги
residual_accumulation_WhitestColumns_coef = 0.1
