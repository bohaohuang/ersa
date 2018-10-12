import os
import tensorflow as tf
from nn import basicNetwork
from nn import nn_utils


class UNet(basicNetwork.SegmentationNetwork):
    """
    Implements the U-Net from https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, class_num, input_size, dropout_rate=None, name='unet', suffix='', learn_rate=1e-4,
                 decay_step=60, decay_rate=0.1, epochs=100, batch_size=5, start_filter_num=32):
        """
        Initialize the object
        :param class_num: class number in labels, determine the # of output units
        :param input_size: input patch size
        :param dropout_rate: dropout rate in each layer, if it is None, no dropout will be used
        :param name: name of this network
        :param suffix: used to create a unique name of the network
        :param learn_rate: start learning rate
        :param decay_step: #steps before the learning rate decay
        :param decay_rate: learning rate will be decayed to lr*decay_rate
        :param epochs: #epochs to train
        :param batch_size: batch size
        :param start_filter_num: #filters at the first layer
        """
        self.sfn = start_filter_num
        super().__init__(class_num, input_size, dropout_rate, name, suffix, learn_rate, decay_step,
                         decay_rate, epochs, batch_size)

    def create_graph(self, feature, **kwargs):
        """
        Create graph for the U-Net
        :param feature: input image
        :param start_filter_num: #filters at the start layer, #filters in U-Net grows exponentially
        :return:
        """
        sfn = self.sfn

        # downsample
        conv1, pool1 = nn_utils.conv_conv_pool(feature, [sfn, sfn], self.mode, name='conv1',
                                               padding='valid', dropout=self.dropout_rate)
        conv2, pool2 = nn_utils.conv_conv_pool(pool1, [sfn * 2, sfn * 2], self.mode, name='conv2',
                                               padding='valid', dropout=self.dropout_rate)
        conv3, pool3 = nn_utils.conv_conv_pool(pool2, [sfn * 4, sfn * 4], self.mode, name='conv3',
                                               padding='valid', dropout=self.dropout_rate)
        conv4, pool4 = nn_utils.conv_conv_pool(pool3, [sfn * 8, sfn * 8], self.mode, name='conv4',
                                               padding='valid', dropout=self.dropout_rate)
        conv5 = nn_utils.conv_conv_pool(pool4, [sfn * 16, sfn * 16], self.mode, name='conv5', pool=False,
                                                padding='valid', dropout=self.dropout_rate)

        # upsample
        up6 = nn_utils.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = nn_utils.conv_conv_pool(up6, [sfn * 8, sfn * 8], self.mode, name='up6', pool=False,
                                        padding='valid', dropout=self.dropout_rate)
        up7 = nn_utils.crop_upsample_concat(conv6, conv3, 32,name='7')
        conv7 = nn_utils.conv_conv_pool(up7, [sfn * 4, sfn * 4], self.mode, name='up7', pool=False,
                                        padding='valid', dropout=self.dropout_rate)
        up8 = nn_utils.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = nn_utils.conv_conv_pool(up8, [sfn * 2, sfn * 2], self.mode, name='up8', pool=False,
                                        padding='valid', dropout=self.dropout_rate)
        up9 = nn_utils.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = nn_utils.conv_conv_pool(up9, [sfn, sfn], self.mode, name='up9', pool=False,
                                        padding='valid', dropout=self.dropout_rate)

        self.pred = tf.layers.conv2d(conv9, self.class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

    @staticmethod
    def get_overlap():
        """
        Get #pixels overlap between two output images
        This is necessary to determine how patches are extracted
        :return:
        """
        return 184

    @staticmethod
    def is_valid_patch_size(ps):
        """
        Due to the existence of cropping and pooling, U-Net cannot take arbitrary input size
        This function determines if a input size is a valid input size, other wise return closest valid size
        :param ps: input patch size, should be a tuple
        :return: True if ps is valid, otherwise the closest valid input size
        """
        if (ps[0] - 124) % 32 == 0 and (ps[1] - 124) % 32 == 0:
            return True
        else:
            ps_0 = (ps[0] - 124) // 32 + 124
            ps_1 = (ps[1] - 124) // 32 + 124
            return tuple([ps_0, ps_1])

    @ staticmethod
    def load_weights(ckpt_dir, layers2load):
        """
        This is different from network.load(). This function only loads specified layers
        :param ckpt_dir: path to the model to load
        :param layers2load: could be a list, or string where numbers separated by ,
        :return:
        """
        layers_list = []
        if isinstance(layers2load, str):
            layers2load = [int(a) for a in layers2load.split(',')]
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        try:
            latest_check_point = tf.train.latest_checkpoint(ckpt_dir)
            tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)
            print('loaded {}'.format(latest_check_point))
        except tf.errors.NotFoundError:
            with open(os.path.join(ckpt_dir, 'checkpoint'), 'r') as f:
                ckpts = f.readlines()
            ckpt_file_name = ckpts[0].split('/')[-1].strip().strip('\"')
            latest_check_point = os.path.join(ckpt_dir, ckpt_file_name)
            tf.contrib.framework.init_from_checkpoint(latest_check_point, load_dict)
            print('loaded {}'.format(latest_check_point))

    def make_loss(self, label, loss_type='xent', **kwargs):
        """
        Make loss to optimize for the network
        U-Net's output is smaller than input, thus ground truth need to be cropped
        :param label: input labels, can be generated by tf.data.Dataset
        :param loss_type:
            xent: cross entropy loss
        :return:
        """
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = label.get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(label, w - self.get_overlap(), h - self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            '''intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            a = tf.cast(tf.reduce_sum(gt), tf.float32)
            b = tf.cast(tf.reduce_sum(pred), tf.float32)
            union = a + b - intersect
            self.loss_iou = tf.convert_to_tensor([intersect, union])'''
            self.loss_iou = tf.metrics.mean_iou(labels=gt, predictions=pred, num_classes=self.class_num)

            if loss_type == 'xent':
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
            else:
                # TODO focal loss:
                # https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
                self.loss = None
