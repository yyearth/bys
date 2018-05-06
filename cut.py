#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time    : 2018/4/30 9:22
# @Author  : yy


import cv2
import numpy as np

cap = cv2.VideoCapture('capt20180430_093218.avi')
while True:
    # get a frame
    ret, frame = cap.read()
    # show a frame
    save = frame.copy()
    cv2.rectangle(frame,(200, 400), (1050, 500), (0, 255, 0), 2)
    cv2.imshow("capture", frame)
    key = cv2.waitKey()
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('frame4.jpg', save)
cap.release()
cv2.destroyAllWindows()
