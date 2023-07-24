import cv2
import numpy as np
import seaborn as sns


def draw_opaque_box(image, bbox, alpha=0.5, gamma=0):
    """Uses bounding box coordinates to draw an opaque box into an image.

    :param image: Any image that can be manipulated using index assignment
    :param bbox: Bounding box coordinates
    :param alpha: Weight of the image (0.0-1.0) (the higher, the more opaque the box)
    :param gamma: Value to add to the product
    """
    beta = 1 - alpha

    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    sub_img = image[y1:y2, x1:x2]
    green_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    green_rect[:, :, 1] = 255

    res = cv2.addWeighted(sub_img, alpha, green_rect, beta, gamma)
    image[y1:y2, x1:x2] = res


def set_seaborn_theme():
    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_context('paper')
