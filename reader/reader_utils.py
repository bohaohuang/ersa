import numpy as np


def image_rotating(img):
    """
    randomly rotate images by 0/90/180/270 degrees
    :param img: input image
    :return:rotated images
    """
    rot_time = np.random.randint(low=0, high=4)
    img = np.rot90(img, rot_time, (0, 1))
    return img


def image_flipping(img):
    """
    randomly flips images left-right and up-down
    :param img: input image
    :return:flipped images
    """
    v_flip = np.random.randint(0, 1)
    h_flip = np.random.randint(0, 1)
    if v_flip == 1:
        img = img[::-1, :, :]
    if h_flip == 1:
        img = img[:, ::-1, :]
    return img