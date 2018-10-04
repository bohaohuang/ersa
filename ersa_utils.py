import os
import time
import pickle
import imageio
import numpy as np
import matplotlib as plt
from functools import wraps
import ersaPath


def make_dir_if_not_exist(dir_path):
    """
    Make the directory if it does not exists
    :param dir_path: absolute path to the directory
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def timer_decorator(func):
    """
    This is a decorator to print out running time of executing func
    :param func:
    :return:
    """
    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        duration = time.time() - start_time
        print('duration: {:.3f}s'.format(duration))
    return timer_wrapper


def get_block_dir(path_field, process_dir):
    """
    Get/Make directory of given fields in process_dir, the directory will be made under path_field
    :param path_field: key defined in resaPath.py
    :param process_dir: sub directory names under path_field
    :return:
    """
    dir_path = ersaPath.PATH[path_field]
    for d in process_dir:
        dir_path = os.path.join(dir_path, d)
    make_dir_if_not_exist(dir_path)
    return dir_path


def str2list(s, sep=',', d_type=int):
    """
    Change a {sep} separated string into a list of items with d_type
    :param s: input string
    :param sep: separator for string
    :param d_type: data type of each element
    :return:
    """
    if type(s) is not list:
        s = [d_type(a) for a in s.split(sep)]
    return s


def load_file(file_name):
    """
    Read data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :return: file data, or IOError if it cannot be read by either numpy or pickle or imageio
    """
    try:
        if file_name[-3:] == 'npy':
            data = np.load(file_name)
        elif file_name[-3:] == 'pkl':
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
        elif file_name[-3:] == 'txt':
            with open(file_name, 'r') as f:
                data = f.readlines()
        else:
            data = imageio.imread(file_name)

        return data
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem loading this data')


def save_file(file_name, data):
    """
    Save data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :param data: data to save
    :return: file data, or IOError if it cannot be saved by either numpy or or pickle imageio
    """
    try:
        if file_name[-3:] == 'npy':
            np.save(file_name, data)
        elif file_name[-3:] == 'pkl':
            with open(file_name, 'wb') as f:
                data = pickle.dump(data, f)
        else:
            data = imageio.imsave(file_name, data)

        return data
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem saving this data')


def get_img_channel_num(file_name):
    """
    Get #channels of the image file
    :param file_name: absolute path to the image file
    :return: #channels or ValueError
    """
    img = load_file(file_name)
    if len(img.shape) == 2:
        channel_num = 1
    elif len(img.shape) == 3:
        channel_num = img.shape[-1]
    else:
        raise ValueError('Image can only have 2 or 3 dimensions')
    return channel_num


def rotate_list(l):
    """
    Rotate a list of lists
    :param l: list of lists to rotate
    :return:
    """
    return list(map(list, zip(*l)))


def pad_image(img, pad, mode='reflect'):
    """
    Symmetric pad pixels around images
    :param img: image to pad
    :param pad: list of #pixels pad around the image, if it is a scalar, it will be assumed to pad same number
                number of pixels around 4 directions
    :param mode: padding mode
    :return: padded image
    """
    if type(pad) is not list:
        pad = [pad for i in range(4)]
    assert len(pad) == 4
    if len(img.shape) == 2:
        return np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), mode)
    else:
        h, w, c = img.shape
        pad_img = np.zeros((h + pad[0] + pad[1], w + pad[2] + pad[3], c))
        for i in range(c):
            pad_img[:, :, i] = np.pad(img[:, :, i], ((pad[0], pad[1]), (pad[2], pad[3])), mode)
    return pad_img


def crop_image(img, y, x, h, w):
    """
    Crop the image with given top-left anchor and corresponding width & height
    :param img: image to be cropped
    :param y: height of anchor
    :param x: width of anchor
    :param h: height of the patch
    :param w: width of the patch
    :return:
    """
    if len(img.shape) == 2:
        return img[y:y+w, x:x+h]
    else:
        return img[y:y+w, x:x+h, :]


def make_center_string(char, length, center_str=''):
    """
    Make one line decoration string that has center_str at the center and surrounded by char
    The total length of the string equals to length
    :param char: character to be repeated
    :param length: total length of the string
    :param center_str: string that shown at the center
    :return:
    """
    return center_str.center(length, char)


def get_default_colors():
    """
    Get plt default colors
    :return: a list of rgb colors
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def get_color_list():
    """
    Get default color list in plt, convert hex value to rgb tuple
    :return:
    """
    colors = get_default_colors()
    return [tuple(int(a.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for a in colors]


def float2str(f):
    """
    Return a string for float number and change '.' to character 'p'
    :param f: float number
    :return: changed string
    """
    return '{}'.format(f).replace('.', 'p')
