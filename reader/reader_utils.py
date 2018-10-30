import skimage.transform
import numpy as np
import ersa_utils


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
    img = image_flipping_hori(img)
    img = image_flipping_vert(img)
    return img


def image_flipping_hori(img):
    """
        randomly flips images left-right
        :param img: input image
        :return:flipped images
        """
    h_flip = np.random.randint(0, 1)
    if h_flip == 1:
        img = img[:, ::-1, :]
    return img


def image_flipping_vert(img):
    """
    randomly flips images up-down
    :param img: input image
    :return:flipped images
    """
    v_flip = np.random.randint(0, 1)
    if v_flip == 1:
        img = img[::-1, :, :]
    return img


def resize_image(img, new_size, preserve_range=False):
    """
    Resize the input image, can preserve the original data range if given ground truth
    :param img: the image to be resized
    :param new_size: new image size
    :param preserve_range: keep the original data range or not
    :return:
    """
    if preserve_range:
        return skimage.transform.resize(img, new_size, order=0, preserve_range=True, mode='reflect')
    else:
        return skimage.transform.resize(img, new_size, mode='reflect')


def image_scaling_with_label(img):
    """
    Random scale images, assume the last channel is the label
    Resize the rgb part with bilinear interpolation, the label part with nearest neighbor
    :param img: image data cube, the last channel is the label
    :param rescale: if True, the image and label will be rescaled to the original shape
    :return: rescaled image
    """
    ftr = img[:, :, :-1]
    lbl = img[:, :, -1]
    scale = np.random.uniform(low=0.5, high=2.0)
    h, w = ftr.shape[:2]
    h_new = int(h * scale)
    w_new = int(w * scale)
    ftr = resize_image(ftr, (h_new, w_new))
    lbl = np.expand_dims(resize_image(lbl, (h_new, w_new), preserve_range=True), axis=-1)
    img = np.dstack([ftr, lbl])
    img = random_pad_crop_image_with_label(img, (h, w))

    return img


def random_crop(img, h_target, w_target):
    """
    Random crop the image to the target size
    :param img: image to be cropped
    :param h_target: target height
    :param w_target: target width
    :return: random cropped image
    """
    h, w, _ = img.shape
    h_range = h - h_target
    w_range = w - w_target
    if h_range == 0:
        h_start = 0
    else:
        h_start = np.random.randint(0, h_range)
    if w_range == 0:
        w_start = 0
    else:
        w_start = np.random.randint(0, w_range)
    img = img[h_start:h_start+h_target, w_start:w_start+w_target, :]
    return img


def random_pad_crop_image_with_label(img, size, ignore_label=255):
    """
    Random pad or crop the image to the desired shape
    :param img: the data cube, assume the label is at the last dimension
    :param size: desired size of the image
    :param ignore_label: label to be ignored
    :return: image with the desired shape
    """
    img[:, :, -1] -= ignore_label  # padded zeros will eventually become the value of ignore label
    h, w, _ = img.shape
    pad_h0, pad_h1, pad_w0, pad_w1 = 0, 0, 0, 0
    if size[0] > h:
        # need padding
        diff = size[0] - h
        if diff % 2 == 0:
            pad_h0, pad_h1 = diff // 2, diff // 2
        else:
            pad_h0 = diff // 2
            pad_h1 = diff - pad_h0
    if size[1] > w:
        # need padding
        diff = size[1] - w
        if diff % 2 == 0:
            pad_w0, pad_w1 = diff // 2, diff // 2
        else:
            pad_w0 = diff // 2
            pad_w1 = diff - pad_w0
    img = ersa_utils.pad_image(img, [pad_h0, pad_h1, pad_w0, pad_w1], mode='constant')

    img = random_crop(img, size[0], size[1])
    img[:, :, -1] += ignore_label
    return img
