import os
import numpy as np
import tensorflow as tf
from glob import glob
from nn import nn_utils
from nn import nn_processor


class Network(object):
    """
    Definition of a basic network, this is the parent class of all networks in this repo
    """
    def __init__(self, class_num, dropout_rate=None, name='network', suffix='', learn_rate=1e-4, decay_step=60,
                 decay_rate=0.1, epochs=100, batch_size=5):
        """
        Initialize a Network object
        :param class_num: class number in labels, determine the # of output units
        :param dropout_rate: dropout rate in each layer, if it is None, no dropout will be used
        :param name: name of this network
        :param suffix: used to create a unique name of the network
        :param learn_rate: start learning rate
        :param decay_step: #steps before the learning rate decay
        :param decay_rate: learning rate will be decayed to lr*decay_rate
        :param epochs: #epochs to train
        :param batch_size: batch size
        """
        self.class_num = class_num
        self.dropout_rate = dropout_rate
        self.model_name = self.get_unique_name(name, suffix)
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.valid_cross_entropy = tf.placeholder(tf.float32, [], name='val_xent')
        self.lr = learn_rate
        self.ds = decay_step
        self.dr = decay_rate
        self.epochs = epochs
        self.bs = batch_size
        self.loss = None
        self.optimizer = None
        self.pred = None
        self.output = None
        self.summary = None
        self.ckdir = None
        self.lr_op = None
        self.update_ops = None
        self.config = None
        self.n_train = 0
        self.n_valid = 0
        # mode is used to determine it's training or not
        self.mode = tf.get_variable(name='mode', shape=[], dtype=tf.bool)
        self.train_op = {True: self.mode.assign(True), False:self.mode.assign(False)}

    def create_graph(self, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass')

    def make_loss(self, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass')

    def train(self, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass')

    def make_ckdir(self, ckdir, patch_size, par_dir=None):
        """
        Made checkpoint directory for the neural network
        :param ckdir: base directory of the ckpt
        :param patch_size: size of the input patch, could be a single number of tuple
        :param par_dir: if not None, the ckpt will be stored in ckdir/par_dir
        :return:
        """
        if type(patch_size) is list:
            patch_size = patch_size[0]
        # make unique directory for save
        dir_name = '{}_PS{}_BS{}_EP{}_LR{}_DS{}_DR{}'.\
            format(self.model_name, patch_size, self.bs, self.epochs, self.lr, self.ds, self.dr)
        if par_dir is None:
            self.ckdir = os.path.join(ckdir, dir_name)
        else:
            self.ckdir = os.path.join(ckdir, par_dir, dir_name)

    def make_learning_rate(self, n_train):
        """
        Make exponential decay learning rate
        :param n_train:
        :return:
        """
        self.lr_op = tf.train.exponential_decay(self.lr, self.global_step,
                                                tf.cast(n_train / self.bs * self.ds, tf.int32), self.dr, staircase=True,
                                                name='learning_rate')

    def make_update_ops(self, feature, label):
        """
        Build update relationship
        :param feature: input feature, can be generated by tf.data.Dataset
        :param label: input label, can be generated by tf.data.Dataset
        :return:
        """
        tf.add_to_collection('inputs', feature)
        tf.add_to_collection('inputs', label)
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self, train_var_filter):
        """
        Make optimizer fot the network, Adam is used
        :param train_var_filter: if not None, only optimize variables in train_var_filter
        :return:
        """
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                self.optimizer = tf.train.AdamOptimizer(self.lr_op).minimize(self.loss, global_step=self.global_step)
            else:
                print('Train parameters in scope:')
                for layer in train_var_filter:
                    print(layer)
                train_vars = tf.trainable_variables()
                var_list = []
                for var in train_vars:
                    if var.name.split('/')[0] in train_var_filter:
                        var_list.append(var)
                self.optimizer = tf.train.AdamOptimizer(self.lr_op).minimize(self.loss, global_step=self.global_step,
                                                                             var_list=var_list)

    def compile(self, feature, label, n_train, n_valid, patch_size, ckdir, train_var_filter=None,
                val_mult=1, par_dir=None, **kwargs):
        self.make_loss(**kwargs)
        self.make_learning_rate(n_train)
        self.make_update_ops(feature, label)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size, par_dir)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.n_train = n_train
        self.n_valid = n_valid

    @staticmethod
    def load(model_path, sess=None, saver=None, epoch=None, best_model=False):
        """
        this can only be called after create_graph()
        loads all weights in a graph
        :param model_path: path of the model to be loaded
        :param sess: session to run the loading function
        :param saver: saver to load the model
        :param epoch: if not None, it will load the specific model at given epoch
        :param best_model: if not None, it will load the model with 'best' prefix
        :return:
        """
        close_sess_flag = False
        if sess is None:
            # create sess if not given, remember to close it
            sess = tf.Session()
            close_sess_flag = True
        if saver is None:
            # create a saver if not given
            saver = tf.train.Saver(var_list=tf.global_variables())
        if os.path.exists(model_path) and tf.train.get_checkpoint_state(model_path):
            if epoch is None:
                # load the latest or best
                best_model_path = glob(os.path.join(model_path, 'best_model.ckpt*.index'))
                if len(best_model_path) > 0 and best_model:
                    # best model exists, load best model
                    best_model_name = best_model_path[0][:-6]
                    saver.restore(sess, best_model_name)
                    print('loaded {}'.format(best_model_name))
                else:
                    try:
                        # try load latest model directly
                        latest_check_point = tf.train.latest_checkpoint(model_path)
                        saver.restore(sess, latest_check_point)
                        print('loaded {}'.format(latest_check_point))
                    except (tf.errors.NotFoundError, ValueError):
                        # path in ckpt file is not correct,
                        # load file name in model_path
                        saver = tf.train.Saver(var_list=[i for i in tf.trainable_variables() if 'save' not in i.name
                                                         and 'mode' not in i.name])
                        with open(os.path.join(model_path, 'checkpoint'), 'r') as f:
                            ckpts = f.readlines()
                        ckpt_file_name = ckpts[0].split('/')[-1].strip().strip('\"')
                        latest_check_point = os.path.join(model_path, ckpt_file_name)
                        saver.restore(sess, latest_check_point)
                        print('loaded {}'.format(latest_check_point))

            else:
                # load given epoch file
                ckpt_file_name = glob(os.path.join(model_path, 'model_{}.ckpt*.index'.format(epoch)))
                ckpt_file_name = ckpt_file_name[0][:-6]
                saver.restore(sess, ckpt_file_name)
                print('loaded {}'.format(ckpt_file_name))
        else:
            saver.restore(sess, model_path)
            print('loaded {}'.format(model_path))

        if close_sess_flag:
            # close the session if it is created here
            sess.close()

    @staticmethod
    def get_unique_name(name, suffix):
        """
        Combine the model name with the suffix
        :param name: model name
        :param suffix: a customized string given be user
        :return:
        """
        if len(suffix) > 0:
            return '{}_{}'.format(name, suffix)
        else:
            return name

    def get_epoch_step(self):
        """
        Compute how many steps in an epch with given n_train and batch size
        :return:
        """
        return nn_utils.get_epoch_step(self.n_train, self.bs)

    def test_sample(self, sess, valid_init):
        """
        Test all samples in given dataset, return a list of predictions
        :param sess: session to run operations
        :param valid_init: init operation for corresponding data reader
        :return: list of predictions
        """
        result = []
        sess.run([valid_init, self.train_op[False]])
        try:
            while True:
                pred = sess.run(self.output)
                result.append(pred)
        except tf.errors.OutOfRangeError:
            result = np.vstack(result)
            return result


class SegmentationNetwork(Network):
    """
    Segmentation Network class, this is the parent class to all segmentation CNNs
    """
    def __init__(self, class_num, input_size, dropout_rate=None, name='seg_network', suffix='', learn_rate=1e-4,
                 decay_step=60, decay_rate=0.1, epochs=100, batch_size=5):
        """
        Initialize a Segmentation Network object
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
        self.loss_xent = None
        self.loss_iou = None  # segmentation network uses IoU for a measurement
        self.valid_iou = tf.placeholder(tf.float32, [], name='val_iou')
        self.input_size = input_size
        super().__init__(class_num, dropout_rate, name, suffix, learn_rate, decay_step, decay_rate,
                         epochs, batch_size)

    def create_graph(self, feature, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass')

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
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
                self.loss_xent = tf.metrics.mean(self.loss)
            else:
                # TODO focal loss:
                # https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
                self.loss = None

    def compile(self, feature, label, n_train, n_valid, patch_size, ckdir, train_var_filter=None,
                val_mult=1, par_dir=None, **kwargs):
        self.make_loss(label, loss_type=kwargs['loss_type'])
        self.make_learning_rate(n_train)
        self.make_update_ops(feature, label)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size, par_dir)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.n_train = n_train
        self.n_valid = n_valid // val_mult // self.bs

    def train(self, train_hooks, valid_hooks=None, train_init=None, valid_init=None, continue_dir=None):
        """
        Train the network
        :param train_hooks: hooks used to monitor training progress, see hook.py for hooks
        :param valid_hooks: hooks used in validation step, see hook.py for hooks
        :param train_init: init op to switch to training stream, created by tf.data.Dataset
        :param valid_init: init op to switch to validation stream, created by tf.data.Dataset
        :param continue_dir: if not None, continue training with the previous step & epoch number
        :return:
        """
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)

            if continue_dir is not None and os.path.exists(continue_dir):
                self.load(continue_dir, sess)
                gs = sess.run(self.global_step)
                start_epoch = int(np.ceil(gs / self.n_train * self.bs))
                start_step = gs - int(start_epoch * self.n_train / self.bs)
            else:
                start_epoch = 0
                start_step = 0

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            try:
                for epoch in range(start_epoch, self.epochs):
                    # training
                    sess.run([train_init, self.train_op[True]])
                    global_step = 0
                    for step in range(start_step, self.n_train, self.bs):
                        sess.run(self.optimizer)
                        global_step = sess.run(self.global_step)
                        sess.run(self.train_op[False])
                        for hook in train_hooks:
                            hook.run(global_step, sess, summary_writer)

                    # validation
                    print('Eval @ Epoch {} '.format(epoch), end='')
                    sess.run([self.train_op[False]])
                    for hook in valid_hooks:
                        sess.run(valid_init)
                        hook.run(global_step, sess, summary_writer)
            finally:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)

    def evaluate(self, file_list, input_size, tile_size, batch_size, img_mean,
                 model_dir, gpu=None, save_result_parent_dir=None, name='nn_estimator_segment',
                 verb=True, ds_name='default', load_epoch_num=None, best_model=False,
                 truth_val=1, force_run=False, score_results=True, split_char='_', **kwargs):
        """
        Evaluate model on given validation set
        :param file_list: evaluation file list
        :param input_size: dimension of the input to the network
        :param tile_size: dimension of the single evaluation file
        :param batch_size: batch size
        :param img_mean: mean of each channel
        :param model_dir: path to the pretrained model
        :param gpu: which gpu to run the model, default to use all the gpus available
        :param save_result_parent_dir: parent directory to where the result will be stored
        :param name: name of the process
        :param verb: if True, print out message when evaluating
        :param ds_name: name of the dataset
        :param load_epoch_num: which epoch's ckpt to load
        :param best_model: if True, load the model with best performance on the validation set
        :param truth_val: value of H1 pixel in gt
        :param force_run: if True, run the evaluation even if results already exist
        :param kwargs: other parameters
        :param score_result: if False, no gt used to score results
        :param split_char: character used to split file name
        :return: tile based iou, field based iou and overall iou
        """
        estimator = nn_processor.NNEstimatorSegment(
            self, file_list, input_size, tile_size, batch_size, img_mean, model_dir, ds_name, save_result_parent_dir,
            name=name, gpu=gpu, verb=verb, load_epoch_num=load_epoch_num, best_model=best_model, truth_val=truth_val,
            score_results=score_results, split_char=split_char, **kwargs)
        if score_results:
            tile_dict, field_dict, overall = estimator.run(force_run=force_run).load_results()
            return tile_dict, field_dict, overall
        else:
            estimator.run(force_run=force_run)
            return estimator.score_save_dir

    @staticmethod
    def get_overlap():
        """
        Get #pixels overlap between two output images
        This is necessary to determine how patches are extracted
        :return:
        """
        return 0
