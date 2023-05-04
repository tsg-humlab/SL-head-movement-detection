import cv2
import numpy as np


def draw_opaque_box(frame, bbox, alpha=0.5, gamma=0):
    beta = 1 - alpha
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    sub_img = frame[y1:y2, x1:x2]
    green_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    green_rect[:, :, 1] = 255
    res = cv2.addWeighted(sub_img, alpha, green_rect, beta, gamma)
    frame[y1:y2, x1:x2] = res
