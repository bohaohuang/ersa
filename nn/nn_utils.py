import os
import numpy as np
import tensorflow as tf
import ersa_utils


def set_gpu(gpu=None):
    """
    Set which gpu to use, this is necessary when your system have multiple gpus
    since tf will occupy all of them by default
    :param gpu: which gpu to use, could be a number or string
    :return:
    """
    if gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def get_tensor_name(t):
    """
    Get name of a tensor
    :param t: tensor variable
    :return:
    """
    return t.name.split(':')[0]


def conv_conv_pool(input_, n_filters, training, name, kernel_size=(3, 3),
                   conv_stride=(1, 1), pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                   activation=tf.nn.relu, padding='same', bn=True, dropout=False, dropout_rate=None):
    """
    Do multiple convolution and one pooling, this is the basic component in many CNNs
    :param input_: input variable
    :param n_filters: #filters in each convolutional layers, could be a list
    :param training: indicates it is in training or not
    :param name: name for this variable scope
    :param kernel_size: kernel size
    :param conv_stride: filter stride for convolutional layers
    :param pool: use pooling or not
    :param pool_size: size of the pooling
    :param pool_stride: stride of the pooling
    :param activation: activation function to use
    :param padding: which padding scheme to use in convolutional layers
    :param bn: use batch normalization or not
    :param dropout: use dropout or not
    :param dropout_rate: drop out rate
    :return:
    """
    net = input_

    with tf.variable_scope('layer{}'.format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, kernel_size, activation=None, strides=conv_stride,
                                   padding=padding, name='conv_{}'.format(i + 1))
            if bn:
                net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i + 1))
            net = activation(net, name='relu_{}'.format(name, i + 1))
            if dropout:
                net = tf.layers.dropout(net, rate=dropout_rate, training=training,
                                        name='drop_{}'.format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
        return net, pool


def concat(input_a, input_b, training, name):
    """
    Concatenate two tensors along the last dimension
    :param input_a:
    :param input_b:
    :param training: indicates it is in training or not
    :param name: name for this variable scope
    :return:
    """
    with tf.variable_scope('layer{}'.format(name)):
        input_a_norm = tf.layers.batch_normalization(input_a, training=training, name='bn')
        return tf.concat([input_a_norm, input_b], axis=-1, name='concat_{}'.format(name))


def upsampling_2d(tensor, name, size=(2, 2)):
    """
    Do 2d upsampling of the input tensor by size times
    :param tensor: input tensor, should be a 3d tensor
    :param name: name for this variable scope
    :param size: how many times the input should be upsampled
    :return:
    """
    h, w, _ = tensor.get_shape().as_list()[1:]  # first dim is batch num
    h_multi, w_multi = size
    target_h = h * h_multi
    target_w = w * w_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_h, target_w), name='upsample_{}'.format(name))


def upsample_concat(input_a, input_b, name, size=(2, 2)):
    """
    Upsample tensor a and concatenate with tensor b
    :param input_a:
    :param input_b:
    :param name: name for this variable scope
    :param size: how many times tensor a should be upsampled
    :return:
    """
    upsample = upsampling_2d(input_a, size=size, name=name)
    return tf.concat([upsample, input_b], axis=-1, name='concat_{}'.format(name))


def upsample_conv_concat(input_a, input_b, filter_n, training, name, size=(2, 2)):
    """
    Upsample tensor a, do convolution and concatenate with tensor b
    :param input_a:
    :param input_b:
    :param filter_n: #filters in convolutional layers to precess upsampled tensor a
    :param training: indicates it is in training or not
    :param name: name for this variable scope
    :param size: how many times tensor a should be upsampled
    :return:
    """
    upsample = upsampling_2d(input_a, size=size, name=name)
    upsample = conv_conv_pool(upsample, filter_n, training, 'upsample_'+name, kernel_size=(2, 2), pool=False)
    return tf.concat([input_b, upsample], axis=-1, name='concat_{}'.format(name))


def crop_upsample_concat(input_a, input_b, margin, name):
    """
    Upsample tensor a, crop tensor b and concatenate them
    :param input_a:
    :param input_b:
    :param margin: the margin tensor b need to be cropped
    :param name: name for this variable scope
    :return:
    """
    _, w, h, _ = input_b.get_shape().as_list()
    input_b_crop = tf.image.resize_image_with_crop_or_pad(input_b, w - margin, h - margin)
    return upsample_concat(input_a, input_b_crop, name)


def crop_upsample_conv_concat(input_a, input_b, margin, name, filter_n, training):
    """
    Upsample tensor a, do convolution on tensor a, crop tensor b and concatenate them
    :param input_a:
    :param input_b:
    :param margin: the margin tensor b need to be cropped
    :param name: name for this variable scope
    :param filter_n: #filters in convolutional layers to precess upsampled tensor a
    :param training: indicates it is in training or not
    :return:
    """
    _, w, h, _ = input_b.get_shape().as_list()
    input_b_crop = tf.image.resize_image_with_crop_or_pad(input_b, w - margin, h - margin)
    return upsample_conv_concat(input_a, input_b_crop, filter_n, training, name)


def fc_fc(input_, n_filters, training, name, activation=tf.nn.relu, dropout=True, dropout_rate=None):
    """
    Do consecutive fully connected layers
    :param input_: input variable
    :param n_filters: #filters in each fully connected layers, could be a list
    :param training: indicates it is in training or not
    :param name: name for this variable scope
    :param activation: filter stride for convolutional layers
    :param dropout: use dropout or not
    :param dropout_rate: drop out rate
    :return:
    """
    net = input_
    with tf.variable_scope('layer{}'.format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.dense(net, F, activation=None)
            if activation is not None:
                net = activation(net, name='relu_{}'.format(name, i + 1))
            if dropout:
                net = tf.layers.dropout(net, rate=dropout_rate, training=training,
                                        name='drop_{}'.format(name, i + 1))
    return net


def get_epoch_step(n_train, batch_size):
    """
    Get how many steps per epoch for given n_train and batch size
    :param n_train: #samples to train per epoch
    :param batch_size: batch size
    :return:
    """
    return n_train//batch_size


def decode_labels(label, label_num=2):
    """
    Decode label prediction map into rgb color map
    :param label: label prediction map
    :param label_num: #distinct classes in ground truth
    :return:
    """
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    color_list = ersa_utils.get_color_list()
    label_colors = {}
    for i in range(label_num):
        label_colors[i] = color_list[i]
    label_colors[0] = (255, 255, 255)
    for i in range(n):
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                pixels[j, k] = label_colors[np.int(label[i, j, k, 0])]
        outputs[i] = pixels
    return outputs


def get_pred_labels(pred):
    """
    Get predicted labels from the output of CNN softmax function
    :param pred: output of CNN softmax function
    :return: predicted labels
    """
    if len(pred.shape) == 4:
        n, h, w, c = pred.shape
        outputs = np.zeros((n, h, w, 1), dtype=np.uint8)
        for i in range(n):
            outputs[i] = np.expand_dims(np.argmax(pred[i, :, :, :], axis=2), axis=2)
        return outputs
    elif len(pred.shape) == 3:
        outputs = np.argmax(pred, axis=2)
        return outputs


def pad_prediction(image, prediction):
    """
    Pad prediction map if necessary, this is useful when network has smaller outputs than input images
    :param image: input rgb image
    :param prediction: network prediction map
    :return:
    """
    _, img_w, img_h, _ = image.shape
    n, pred_img_w, pred_img_h, c = prediction.shape

    if img_w > pred_img_w and img_h > pred_img_h:
        pad_w = int((img_w - pred_img_w) / 2)
        pad_h = int((img_h - pred_img_h) / 2)
        prediction_padded = np.zeros((n, img_w, img_h, c))
        pad_dim = ((pad_w, pad_w),
                   (pad_h, pad_h))

        for batch_id in range(n):
            for channel_id in range(c):
                prediction_padded[batch_id, :, :, channel_id] = \
                    np.pad(prediction[batch_id, :, :, channel_id], pad_dim, 'constant')
        prediction = prediction_padded
        return prediction
    else:
        return prediction


def iou_metric(truth, pred, truth_val=1, divide_flag=False):
    """
    calculate iou with given truth and prediction map
    :param truth: truth image
    :param pred: prediction map
    :param truth_val: truth value, default to 1
    :param divide_flag: if False, numerator and denominator will be returned separately
    :return: iou scalar value or numerator and denominator list
    """
    truth = truth / truth_val
    pred = pred / truth_val
    truth = truth.flatten()
    pred = pred.flatten()
    intersect = truth*pred
    if divide_flag:
        return np.sum(intersect == 1), np.sum(truth+pred >= 1)
    else:
        return np.sum(intersect == 1) / np.sum(truth+pred >= 1)


def read_iou_from_file(result_record):
    """
    read iou records from a file, ious will be stored based on each file and each filed (city_name)
    :param result_record: record read from a result file
    :return: tile based iou, field based iou and overall iou
    """
    tile_dict = {}
    field_list = []
    field_dict = {}
    overall = np.zeros(2)
    for cnt, line in enumerate(result_record[:-1]):
        tile_name = line.split(' ')[0]
        a, b = [float(item) for item in line.split('(')[1].strip().strip(')').split(',')]
        tile_dict[tile_name] = a / b
        field_name = ''.join([a for a in tile_name if not a.isdigit()])
        if field_name not in field_dict:
            field_list.append(field_name)
            field_dict[field_name] = np.array([a, b])
        else:
            field_dict[field_name] += np.array([a, b])
        overall += np.array([a, b])
    for field_name in field_list:
        field_dict[field_name] = field_dict[field_name][0] / field_dict[field_name][1]
    overall = overall[0] / overall[1]
    return tile_dict, field_dict, overall


def image_summary(image, truth, prediction, img_mean=np.array((0, 0, 0), dtype=np.float32), label_num=2):
    """
    Make a image summary where the format is image|truth|pred
    :param image: input rgb image
    :param truth: ground truth
    :param prediction: network prediction
    :param img_mean: image mean, need to add back here for visualization
    :param label_num: #distinct classes in ground truth
    :return:
    """
    truth_img = decode_labels(truth, label_num)

    prediction = pad_prediction(image, prediction)

    pred_labels = get_pred_labels(prediction)
    pred_img = decode_labels(pred_labels, label_num)

    _, h, w, _ = image.shape
    if w/h > 1.5:
        # concatenate image horizontally if it is too wide
        return np.concatenate([image+img_mean, truth_img, pred_img], axis=1)
    else:
        return np.concatenate([image + img_mean, truth_img, pred_img], axis=2)


def tf_warn_level(warn_level=3):
    """
    Filter out info from tensorflow output
    :param warn_level: can be 0 or 1 or 2
    :return:
    """
    if isinstance(warn_level, int):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(warn_level)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = warn_level
