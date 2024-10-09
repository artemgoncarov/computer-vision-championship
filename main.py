from utils import Car, turn
import cv2
import numpy as np

# Init
car = Car()

# Initializing video
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Fail open file")
    exit()

# Image size
image_size = (200, 360)
warped_image_size = (300, 200)

# Tracking coordinates
coords = np.array([
    [20, 200],
    [350, 200],
    [275, 120],
    [85, 120]
])

# Allocating a plot data
dst_coords = np.array([[0, 0], [warped_image_size[0], 0],
                       [warped_image_size[0], warped_image_size[1]], [0, warped_image_size[1]]])

# Reading
status, frame = video.read()

while status:
    image = cv2.resize(frame, image_size[::-1])

    # Binary
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.inRange(gray_image, 200, 255)

    # Allocating a plot
    matrix, status = cv2.findHomography(coords, dst_coords)
    binary_warped = cv2.warpPerspective(binary_image, matrix, warped_image_size)

    car.drive(100, turn(binary_warped), 0.75)

    status, frame = video.read()

car.drive(100, 100)

# Exit
car.exit()
