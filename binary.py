import cv2
import numpy as np

def nott(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("result")
cv2.createTrackbar("minB", "result", 0, 255, nott)
cv2.createTrackbar("minG", "result", 0, 255, nott)
cv2.createTrackbar("minR", "result", 0, 255, nott)

cv2.createTrackbar("maxB", "result", 0, 255, nott)
cv2.createTrackbar("maxG", "result", 0, 255, nott)
cv2.createTrackbar("maxR", "result", 0, 255, nott)

ESCAPE = 27
while True:
    # frame = img.copy()
    ret, frame = cap.read()
    if ret is False:
        print('END')
        break

    # print(frame.shape)
    w, h = frame.shape[:2]
    # frame = cv2.resize(frame, (h//2, w//2))
    cv2.imshow("Frame", frame)

    frame = cv2.blur(frame, (3, 3), 3)
    cv2.imshow("Blured_Frame", frame)

    minb = cv2.getTrackbarPos("minB", "result")
    ming = cv2.getTrackbarPos("minG", "result")
    minr = cv2.getTrackbarPos("minR", "result")

    maxb = cv2.getTrackbarPos("maxB", "result")
    maxg = cv2.getTrackbarPos("maxG", "result")
    maxr = cv2.getTrackbarPos("maxR", "result")

    binary = cv2.inRange(frame, (minb, ming, minr), (maxb, maxg, maxr))
    cv2.imshow("Binary", binary)

    result = cv2.bitwise_and(frame, frame, mask=binary)
    cv2.imshow("result", result)

    key = cv2.waitKey(10)
    if key == ESCAPE:
        break

cv2.destroyAllWindows()