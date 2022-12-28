import cv2
import random
import numpy as np
from PIL import Image

def ran(range_=0.1):
    """generate random value"""
    return random.random() * range_ * 2 - range_ + 1


def resize_image(img, level=0.1):
    """resize image in a bbox"""
    assert level < 1
    h, w = img.shape[0], img.shape[1]
    shrink = cv2.resize(img, (int(w * ran(range_=level)), int(h * ran(range_=level))), interpolation=cv2.INTER_AREA)
    return shrink


def resize_bbox(img, bbox, level=0.1):
    bbox_set = set(tuple(_) for _ in bbox)
    new_bbox = []
    bbox_dict = {}
    img = np.asarray(img)
    max_y, max_x = img.shape[0], img.shape[1]

    def change_scale(x, factor):
        return int(x * factor / 1000)

    for bbox_ in bbox_set:
        x1_, y1_, x3_, y3_ = bbox_

        x1 = change_scale(x1_, max_x)
        x3 = change_scale(x3_, max_x)
        y1 = change_scale(y1_, max_y)
        y3 = change_scale(y3_, max_y)

        if y1 != y3:
            new_img = resize_image(img[y1:y3, x1:x3], level=level)
            h, w = new_img.shape[0], new_img.shape[1]
            if y1 + h < max_y and x1 + w < max_x:
                img[y1:y3, x1:x3] = 255
                img[y1:y1 + h, x1:x1 + w] = new_img
                bbox_dict[bbox_] = [x1_, y1_, int(x1_ + w / max_x * 1000), int(y1_ + h / max_y * 1000)]
    for bbox_ in bbox:
        if bbox_dict.get(tuple(bbox_)) is None:
            new_bbox.append(bbox_)
        else:
            new_bbox.append(bbox_dict.get(tuple(bbox_)))
    return Image.fromarray(np.uint8(img)), new_bbox
