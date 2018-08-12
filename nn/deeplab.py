import tensorflow as tf
from nn import basicNetwork


class DeepLab(basicNetwork.SegmentationNetwork):
    """
    Implements the Deeplab from https://arxiv.org/pdf/1706.05587.pdf
    """
    def __init__(self, class_num, input_size, dropout_rate=None, name='deeplab', suffix='', learn_rate=1e-5,
                 decay_step=40, decay_rate=0.1, epochs=100, batch_size=5):
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
        """
        self.channel_axis = 3
        self.input_size = None
        self.encoding = None
        self.output_size = None
        super().__init__(class_num, input_size, dropout_rate, name, suffix, learn_rate, decay_step,
                         decay_rate, epochs, batch_size)

    def create_graph(self, feature, start_filter_num):
        """
        Create graph for the U-Net
        :param feature: input image
        :param start_filter_num: #filters at the start layer, #filters in U-Net grows exponentially
        :return:
        """
        self.input_size = feature.shape[1:3]

        self.encoding = self.build_encoder(feature)
        self.pred = self.build_decoder(self.encoding)
        self.output_size = self.pred.shape[1:3]

        self.output = tf.image.resize_bilinear(tf.nn.softmax(self.pred), self.input_size)

    def build_encoder(self, feature):
        print("-----------build encoder-----------")
        scope_name = 'resnet_v1_101'
        with tf.variable_scope(scope_name):
            outputs = self._start_block('conv1', feature)
            print("after start block:", outputs.shape)
            with tf.variable_scope('block1'):
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_1', identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')
                print("after block1:", outputs.shape)
            with tf.variable_scope('block2'):
                outputs = self._bottleneck_resblock(outputs, 512, 'unit_1', half_size=True, identity_connection=False)
                for i in range(2, 5):
                    outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)
                print("after block2:", outputs.shape)
            with tf.variable_scope('block3'):
                outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1', identity_connection=False)
                num_layers_block3 = 23
                for i in range(2, num_layers_block3 + 1):
                    outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)
                print("after block3:", outputs.shape)
            with tf.variable_scope('block4'):
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
                print("after block4:", outputs.shape)
                return outputs

    def build_decoder(self, encoding):
        print("-----------build decoder-----------")
        with tf.variable_scope('decoder'):
            outputs = self._aspp(encoding, self.class_num, [6, 12, 18, 24])
            print("after aspp block:", outputs.shape)
            return outputs

    def _start_block(self, name, feature):
        outputs = self._conv2d(feature, 7, 64, 2, name=name)
        outputs = self._batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        return outputs

    def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
        first_s = 2 if half_size else 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False,
                                    activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False,
                                    activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _aspp(self, x, num_o, dilations):
        o = []
        for i, d in enumerate(dilations):
            o.append(self._dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i + 1), biased=True))
        return self._add(o, name='aspp/add')

        # layers

    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
        """
        Conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name):
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            s = [1, stride, stride, 1]
            o = tf.nn.conv2d(x, w, s, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
        """
        Dilated conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name):
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    @staticmethod
    def _relu(x, name):
        return tf.nn.relu(x, name=name)

    @staticmethod
    def _add(x_l, name):
        return tf.add_n(x_l, name=name)

    @staticmethod
    def _max_pool2d(x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

    @staticmethod
    def _batch_norm(x, name, is_training, activation_fn, trainable=False):
        # For a small batch size, it is better to keep
        # the statistics of the BN layers (running means and variances) frozen,
        # and to not update the values provided by the pre-trained model by setting is_training=False.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name + '/BatchNorm') as scope:
            o = tf.contrib.layers.batch_norm(
                x,
                scale=True,
                activation_fn=activation_fn,
                is_training=is_training,
                trainable=trainable,
                scope=scope)
            return o
