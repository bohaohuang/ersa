"""
This file defines PSPNet
Heavily copy from https://github.com/hellochick/PSPNet-tensorflow
"""


import numpy as np
import tensorflow as tf
from nn import basicNetwork

DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'

BN_param_map = {'scale':    'gamma',
                'offset':   'beta',
                'variance': 'moving_variance',
                'mean':     'moving_mean'}


def layer(op):
    """
    Decorator for composable network layers.
    """

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True, is_training=False, num_classes=21):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.is_training = is_training
        self.setup(is_training, num_classes)

    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        if 'bn' in op_name:
                            param_name = BN_param_map[param_name]
                            data = np.squeeze(data)

                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def get_layer_name(self):
        return self.layer_name

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')
    @layer
    def zero_padding(self, input, paddings, name):
        pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
        return tf.pad(input, paddings=pad_mat, name=name)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding,data_format=DEFAULT_DATAFORMAT)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        output = tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)
        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        output = tf.layers.batch_normalization(
            input,
            momentum=0.95,
            epsilon=1e-5,
            training=self.is_training,
            name=name
        )

        if relu:
            output = tf.nn.relu(output)

        return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def resize_bilinear(self, input, size, name):
        return tf.image.resize_bilinear(input, size=size, align_corners=True, name=name)


class PSPNet101(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
             .relu(name='conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
             .batch_normalization(relu=True, name='conv1_2_3x3_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
             .batch_normalization(relu=True, name='conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_7_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_7_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding14')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_7_3x3')
             .batch_normalization(relu=True, name='conv4_7_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_7_1x1_increase')
             .batch_normalization(relu=False, name='conv4_7_1x1_increase_bn'))

        (self.feed('conv4_6/relu',
                   'conv4_7_1x1_increase_bn')
             .add(name='conv4_7')
             .relu(name='conv4_7/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_8_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_8_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding15')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_8_3x3')
             .batch_normalization(relu=True, name='conv4_8_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_8_1x1_increase')
             .batch_normalization(relu=False, name='conv4_8_1x1_increase_bn'))

        (self.feed('conv4_7/relu',
                   'conv4_8_1x1_increase_bn')
             .add(name='conv4_8')
             .relu(name='conv4_8/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_9_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_9_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding16')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_9_3x3')
             .batch_normalization(relu=True, name='conv4_9_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_9_1x1_increase')
             .batch_normalization(relu=False, name='conv4_9_1x1_increase_bn'))

        (self.feed('conv4_8/relu',
                   'conv4_9_1x1_increase_bn')
             .add(name='conv4_9')
             .relu(name='conv4_9/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_10_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_10_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding17')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_10_3x3')
             .batch_normalization(relu=True, name='conv4_10_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_10_1x1_increase')
             .batch_normalization(relu=False, name='conv4_10_1x1_increase_bn'))

        (self.feed('conv4_9/relu',
                   'conv4_10_1x1_increase_bn')
             .add(name='conv4_10')
             .relu(name='conv4_10/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_11_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_11_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_11_3x3')
             .batch_normalization(relu=True, name='conv4_11_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_11_1x1_increase')
             .batch_normalization(relu=False, name='conv4_11_1x1_increase_bn'))

        (self.feed('conv4_10/relu',
                   'conv4_11_1x1_increase_bn')
             .add(name='conv4_11')
             .relu(name='conv4_11/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_12_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_12_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding19')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_12_3x3')
             .batch_normalization(relu=True, name='conv4_12_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_12_1x1_increase')
             .batch_normalization(relu=False, name='conv4_12_1x1_increase_bn'))

        (self.feed('conv4_11/relu',
                   'conv4_12_1x1_increase_bn')
             .add(name='conv4_12')
             .relu(name='conv4_12/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_13_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_13_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding20')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_13_3x3')
             .batch_normalization(relu=True, name='conv4_13_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_13_1x1_increase')
             .batch_normalization(relu=False, name='conv4_13_1x1_increase_bn'))

        (self.feed('conv4_12/relu',
                   'conv4_13_1x1_increase_bn')
             .add(name='conv4_13')
             .relu(name='conv4_13/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_14_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_14_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding21')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_14_3x3')
             .batch_normalization(relu=True, name='conv4_14_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_14_1x1_increase')
             .batch_normalization(relu=False, name='conv4_14_1x1_increase_bn'))

        (self.feed('conv4_13/relu',
                   'conv4_14_1x1_increase_bn')
             .add(name='conv4_14')
             .relu(name='conv4_14/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_15_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_15_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding22')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_15_3x3')
             .batch_normalization(relu=True, name='conv4_15_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_15_1x1_increase')
             .batch_normalization(relu=False, name='conv4_15_1x1_increase_bn'))

        (self.feed('conv4_14/relu',
                   'conv4_15_1x1_increase_bn')
             .add(name='conv4_15')
             .relu(name='conv4_15/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_16_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_16_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding23')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_16_3x3')
             .batch_normalization(relu=True, name='conv4_16_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_16_1x1_increase')
             .batch_normalization(relu=False, name='conv4_16_1x1_increase_bn'))

        (self.feed('conv4_15/relu',
                   'conv4_16_1x1_increase_bn')
             .add(name='conv4_16')
             .relu(name='conv4_16/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_17_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_17_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding24')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_17_3x3')
             .batch_normalization(relu=True, name='conv4_17_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_17_1x1_increase')
             .batch_normalization(relu=False, name='conv4_17_1x1_increase_bn'))

        (self.feed('conv4_16/relu',
                   'conv4_17_1x1_increase_bn')
             .add(name='conv4_17')
             .relu(name='conv4_17/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_18_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_18_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding25')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_18_3x3')
             .batch_normalization(relu=True, name='conv4_18_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_18_1x1_increase')
             .batch_normalization(relu=False, name='conv4_18_1x1_increase_bn'))

        (self.feed('conv4_17/relu',
                   'conv4_18_1x1_increase_bn')
             .add(name='conv4_18')
             .relu(name='conv4_18/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_19_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_19_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding26')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_19_3x3')
             .batch_normalization(relu=True, name='conv4_19_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_19_1x1_increase')
             .batch_normalization(relu=False, name='conv4_19_1x1_increase_bn'))

        (self.feed('conv4_18/relu',
                   'conv4_19_1x1_increase_bn')
             .add(name='conv4_19')
             .relu(name='conv4_19/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_20_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_20_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding27')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_20_3x3')
             .batch_normalization(relu=True, name='conv4_20_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_20_1x1_increase')
             .batch_normalization(relu=False, name='conv4_20_1x1_increase_bn'))

        (self.feed('conv4_19/relu',
                   'conv4_20_1x1_increase_bn')
             .add(name='conv4_20')
             .relu(name='conv4_20/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_21_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_21_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding28')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_21_3x3')
             .batch_normalization(relu=True, name='conv4_21_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_21_1x1_increase')
             .batch_normalization(relu=False, name='conv4_21_1x1_increase_bn'))

        (self.feed('conv4_20/relu',
                   'conv4_21_1x1_increase_bn')
             .add(name='conv4_21')
             .relu(name='conv4_21/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_22_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_22_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding29')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_22_3x3')
             .batch_normalization(relu=True, name='conv4_22_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_22_1x1_increase')
             .batch_normalization(relu=False, name='conv4_22_1x1_increase_bn'))

        (self.feed('conv4_21/relu',
                   'conv4_22_1x1_increase_bn')
             .add(name='conv4_22')
             .relu(name='conv4_22/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_23_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_23_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding30')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_23_3x3')
             .batch_normalization(relu=True, name='conv4_23_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_23_1x1_increase')
             .batch_normalization(relu=False, name='conv4_23_1x1_increase_bn'))

        (self.feed('conv4_22/relu',
                   'conv4_23_1x1_increase_bn')
             .add(name='conv4_23')
             .relu(name='conv4_23/relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_23/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding31')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding32')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding33')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        conv5_3 = self.layers['conv5_3/relu']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('conv5_3/relu')
             .avg_pool(90, 90, 90, 90, name='conv5_3_pool1')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
             .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(45, 45, 45, 45, name='conv5_3_pool2')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
             .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(30, 30, 30, 30, name='conv5_3_pool3')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
             .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(15, 15, 15, 15, name='conv5_3_pool6')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
             .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .concat(axis=-1, name='conv5_3_concat')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
             .batch_normalization(relu=True, name='conv5_4_bn')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))


class PSPNet(basicNetwork.SegmentationNetwork):
    """
    Implements the Deeplab from https://arxiv.org/pdf/1706.05587.pdf
    """
    def __init__(self, class_num, input_size, dropout_rate=None, name='pspnet', suffix='', learn_rate=1e-5,
                 decay_step=40, decay_rate=0.1, epochs=100, batch_size=5, weight_decay=1e-4, momentum=0.9):
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
        self.output_size = None
        self.weight_decay = weight_decay
        self.momentum = momentum
        super().__init__(class_num, input_size, dropout_rate, name, suffix, learn_rate, decay_step,
                         decay_rate, epochs, batch_size)

    def create_graph(self, feature, **kwargs):
        """
        Create graph for the deeplab
        :param feature: input image
        :return:
        """
        self.input_size = feature.shape[1:3]

        net = PSPNet101({'data': feature}, is_training=True, num_classes=self.class_num)
        self.pred = net.layers['conv6']
        self.pred = tf.image.resize_bilinear(self.pred, self.input_size)
        self.output_size = self.pred.shape[1:3]
        self.output = tf.nn.softmax(self.pred)

    def make_loss(self, label, loss_type='xent'):
        """
        Make loss to optimize for the network
        :param label: input labels, can be generated by tf.data.Dataset
        :param loss_type:
            xent: cross entropy loss
        :return:
        """
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(label, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            self.loss_iou = tf.metrics.mean_iou(labels=gt, predictions=pred, num_classes=self.class_num)

            if loss_type == 'xent':
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
                l2_losses = [self.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables()
                             if 'weight' in v.name]
                self.loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    def make_optimizer(self, train_var_filter):
        """
        Make optimizer fot the network, Adam is used
        :param train_var_filter: if not None, only optimize variables in train_var_filter
        :return:
        """
        # According from the prototxt in Caffe implement, learning rate must multiply by 10.0 in pyramid module
        fc_list = ['conv5_3_pool1_conv', 'conv5_3_pool2_conv', 'conv5_3_pool3_conv', 'conv5_3_pool6_conv', 'conv6',
                   'conv5_4']
        all_trainable = [v for v in tf.trainable_variables() if
                         ('beta' not in v.name and 'gamma' not in v.name) or True]
        fc_trainable = [v for v in all_trainable if v.name.split('/')[0] in fc_list]
        conv_trainable = [v for v in all_trainable if v.name.split('/')[0] not in fc_list]  # lr * 1.0
        fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
        fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
        assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

        with tf.control_dependencies(self.update_ops):
            opt_conv = tf.train.MomentumOptimizer(self.lr_op, self.momentum)
            opt_fc_w = tf.train.MomentumOptimizer(self.lr_op * 10.0, self.momentum)
            opt_fc_b = tf.train.MomentumOptimizer(self.lr_op * 20.0, self.momentum)

            grads = tf.gradients(self.loss, conv_trainable + fc_w_trainable + fc_b_trainable)
            grads_conv = grads[:len(conv_trainable)]
            grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
            grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

            train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable), global_step=self.global_step)
            train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
            train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

            self.optimizer = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    def load_resnet(self, resnet_dir):
        """
        Load the resnet101 model pretrained on ImageNet
        :param resnet_dir: path to the pretrained model
        :return:
        """
        ckpt = tf.train.latest_checkpoint(resnet_dir)
        with tf.Session(config=self.config) as sess:
            # init model
            init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
            sess.run(init)
            restore_var = [v for v in tf.global_variables() if 'conv6' not in v.name and 'global_step' not in v.name
                           and 'mode' not in v.name]
            loader = tf.train.Saver(var_list=restore_var)
            # load model
            self.load(ckpt, sess, loader)
