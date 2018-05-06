#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time    : 2018/5/3 16:05
# @Author  : yy


import numpy as np
import cv2


def solveline(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    A = np.array([[x1, 1], [x2, 1]])
    b = np.array([y1, y2])
    sov = np.linalg.solve(A, b)
    return sov


def detectlane(img, drt=None):
    roi = img[400:500, 200:1050]
    flag = False
    cp = roi.copy()
    ta, tb = drt
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, roi = cv2.threshold(roi, 210, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img', roi)
    # cv2.waitKey()
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=10)

    if lines is None:
        return img
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(cp, (x1, y1), (x2, y2), (0, 255, 0), 1)
        try:
            la, lb = solveline((x1, y1), (x2, y2))
        except np.linalg.linalg.LinAlgError as e:
            # flag = True
            print(e, end=' ')
            print(x1, y1, x2, y2)
            # cv2.waitKey()
        # print('------------------------------------------')
        # print(la, lb)

        A = np.array([[-la, 1], [-ta, 1]])
        b = np.array([lb, tb])
        encounter = np.linalg.solve(A, b)

        # print('encounter:', encounter)
        # print('------------------------------------------')

        if abs(la) < 1 or abs(la) > 1e+10:
            continue
        if encounter[1] > -20:
            flag = True
        if lb > 0:
            cv2.line(cp, (0, int(lb)), (int(-lb / la), 0), (0, 0, 255), 1)
        elif lb < 0:
            cv2.line(cp, (int((100 - lb) / la), 100), (int(-lb / la), 0), (0, 0, 255), 1)
        # cv2.imshow('img', cp)
        # cv2.waitKey()

    if flag:
        h, w, ch = img.shape
        cv2.putText(img, 'Danger!', (w // 2, (h - 30) // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    else:
        h, w, ch = img.shape
        cv2.putText(img, 'Safe!', (w // 2, (h - 30) // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 179, 7), 2)

    img[400:500, 200:1050] = cp
    return img


if __name__ == '__main__':
    ta, tb = solveline((434, 0), (415, 100))
    # img = cv2.imread('frame2.jpg')
    # show = detectlane(img, (ta, tb))
    # cv2.imshow('img', show)
    # cv2.waitKey()

    # cap = cv2.VideoCapture('capt20180430_093218.avi')
    cap = cv2.VideoCapture('Driving in Los Angeles Interstate 405.mp4')
    ret, frame = cap.read()
    while ret:
        # get a frame
        save = frame.copy()
        show = detectlane(frame, (ta, tb))
        cv2.imshow("capture", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite('frame4.jpg', save)

        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
