import numpy as np
import tensorflow as tf
import utils


class DataReaderSegmentation(object):
    def __init__(self, input_size, file_list, batch_size=5, chan_mean=None, aug_func=None, is_train=True, random=True,
                 has_gt=True, gt_dim=0, include_gt=True):
        """
        Initialize a data reader for segmentation model where the lable is a densely labeled map
        :param input_size: patch size to be read
        :param file_list: list of lists of patch files
        :param batch_size: batch size to read each time
        :param chan_mean: mean of the channel, if set to None it will be zeros for all non gt channels
        :param aug_func: augmentation function, could be a list of augmentation functions or leave to None
        :param is_train: it's used for training, then no random permutation will be used
        :param random: if use random permutation or not
        :param has_gt: the input file list has ground truth or not
        :param gt_dim: how many 3rd dimension the input data has
        :param include_gt: reads gt or not
        """
        self.input_size = input_size
        self.file_list = file_list
        self.batch_size = batch_size
        self.chan_mean = chan_mean
        if aug_func is None:
            aug_func = []
        if type(aug_func) is not list:
            aug_func = [aug_func]
        self.aug_func = aug_func
        self.is_train = is_train
        self.random = random
        self.has_gt = has_gt
        self.gt_dim = gt_dim
        self.include_gt = include_gt

        # read one set of files to get #channels
        self.channel_num = 0
        for f in self.file_list[0]:
            self.channel_num += utils.get_img_channel_num(f)
        if self.chan_mean is None:
            self.chan_mean = np.zeros(self.channel_num - self.gt_dim)

    def data_reader_helper(self, files):
        """
        Helper function of data reader, reads list of lists files
        :param files: list of lists, each element is a patch name, each row is corresponds to one file
        :return: feature and label, or only feature
        """
        data_block = []
        for f in files:
            data_block.append(utils.load_file(f))
        data_block = np.dstack(data_block)
        for aug_func in self.aug_func:
            data_block = aug_func(data_block)
        if self.has_gt:
            ftr_block = data_block[:, :, :-self.gt_dim]
            ftr_block = ftr_block - self.chan_mean
            lbl_block = data_block[:, :, -self.gt_dim:]
            if self.include_gt:
                return ftr_block, lbl_block
            else:
                return ftr_block
        else:
            data_block = data_block - self.chan_mean
            return data_block

    def data_reader(self):
        """
        Read feature and label, or feature alone
        :return: data read from file list, one line each time
        """
        if self.random:
            self.file_list = np.random.permutation(self.file_list)
        for files in self.file_list:
            yield self.data_reader_helper(files)

    def get_dataset(self):
        """
        Create a tf.Dataset from the generator defined
        :return: a tf.Dataset object
        """
        def generator(): return self.data_reader()
        if self.has_gt and self.include_gt:
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.int32),
                                                     ((self.input_size[0], self.input_size[1],
                                                       self.channel_num - self.gt_dim),
                                                      (self.input_size[0], self.input_size[1], self.gt_dim)))
        elif self.has_gt and not self.include_gt:
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32,),
                                                     ((self.input_size[0], self.input_size[1],
                                                       self.channel_num - self.gt_dim)))
        else:
            dataset = tf.data.Dataset.from_generator(generator, (tf.float32,),
                                                     ((self.input_size[0], self.input_size[1], self.channel_num),))
        return dataset

    def read_op(self):
        """
        Get tf iterator as well as init operation for the dataset
        :return: reader operation and init operation
        """
        dataset = self.get_dataset()
        if self.is_train:
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
        else:
            dataset = dataset.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        reader_op = iterator.get_next()
        init_op = iterator.make_initializer(dataset)

        return init_op, reader_op


class DataReaderSegmentationTrainValid(object):
    def __init__(self, input_size, file_list_train, file_list_valid, batch_size=5, chan_mean=None, aug_func=None,
                 random=True, has_gt=True, gt_dim=0, include_gt=True, valid_mult=1):
        """
        Initialize a data reader for segmentation model where the lable is a densely labeled map
        This data reader separates training data and validation data for you
        :param input_size: patch size to be read
        :param file_list_train: list of lists of patch files for training
        :param file_list_valid: list of lists of patch files for validation
        :param batch_size: batch size to read each time
        :param chan_mean: mean of the channel, if set to None it will be zeros for all non gt channels
        :param aug_func: augmentation function, could be a list of augmentation functions or leave to None
        :param random: if use random permutation or not
        :param has_gt: the input file list has ground truth or not
        :param gt_dim: how many 3rd dimension the input data has
        :param include_gt: reads gt or not
        :param valid_mult: validation can have a batch size of valid_mult*batch_size due to the absence of backprop
        """
        self.input_size = input_size
        self.file_list_train = file_list_train
        self.file_list_valid = file_list_valid
        self.batch_size = batch_size
        self.chan_mean = chan_mean
        if aug_func is None:
            aug_func = []
        if type(aug_func) is not list:
            aug_func = [aug_func]
        self.aug_func = aug_func
        self.random = random
        self.has_gt = has_gt
        self.gt_dim = gt_dim
        self.include_gt = include_gt

        # read one set of files to get #channels
        self.channel_num = 0
        for f in self.file_list_train[0]:
            self.channel_num += utils.get_img_channel_num(f)
        if self.chan_mean is None:
            self.chan_mean = np.zeros(self.channel_num - self.gt_dim)

        self.valid_mult = valid_mult

    def data_reader_helper(self, files, is_train):
        """
        Helper function of data reader, reads list of lists files
        :param files: list of lists, each element is a patch name, each row is corresponds to one file
        :param is_train: the dataset is used for training or not
        :return: feature and label, or only feature
        """
        data_block = []
        for f in files:
            data_block.append(utils.load_file(f))
        data_block = np.dstack(data_block)
        if is_train:
            for aug_func in self.aug_func:
                data_block = aug_func(data_block)
        if self.has_gt:
            ftr_block = data_block[:, :, :-self.gt_dim]
            ftr_block = ftr_block - self.chan_mean
            lbl_block = data_block[:, :, -self.gt_dim:]
            if self.include_gt:
                return ftr_block, lbl_block
            else:
                return ftr_block
        else:
            data_block = data_block - self.chan_mean
            return data_block

    def data_reader(self, file_list, is_train, random=False):
        """
        Read feature and label, or feature alone
        :param file_list: list of files to read data
        :param is_train: the dataset is used for training or not
        :param random: include randomness in reading of not
        :return: data read from file list, one line each time
        """
        if random:
            file_list = np.random.permutation(file_list)
        for files in file_list:
            yield self.data_reader_helper(files, is_train)

    def get_dataset(self):
        """
        Create a tf.Dataset from the generator defined
        :return: a tf.Dataset object
        """
        def generator_train(): return self.data_reader(self.file_list_train, True, self.random)

        def generator_valid(): return self.data_reader(self.file_list_valid, False)

        if self.has_gt and self.include_gt:
            dataset_train = tf.data.Dataset.from_generator(generator_train, (tf.float32, tf.int32,),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - self.gt_dim),
                                                            (self.input_size[0], self.input_size[1], self.gt_dim),))
            dataset_valid = tf.data.Dataset.from_generator(generator_valid, (tf.float32, tf.int32, ),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - self.gt_dim),
                                                            (self.input_size[0], self.input_size[1], self.gt_dim),))
        elif self.has_gt and not self.include_gt:
            dataset_train = tf.data.Dataset.from_generator(generator_train, (tf.float32, ),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - self.gt_dim), ))
            dataset_valid = tf.data.Dataset.from_generator(generator_valid, (tf.float32, ),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - self.gt_dim), ))
        else:
            dataset_train = tf.data.Dataset.from_generator(
                generator_train, (tf.float32, ), ((self.input_size[0], self.input_size[1], self.channel_num), ))
            dataset_valid = tf.data.Dataset.from_generator(
                generator_valid, (tf.float32, ), ((self.input_size[0], self.input_size[1], self.channel_num), ))
        return dataset_train, dataset_valid

    def read_op(self):
        """
        Get tf iterator as well as init operation for the dataset
        :return: reader operation and init operation
        """
        dataset_train, dataset_valid = self.get_dataset()
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.batch(self.batch_size)

        dataset_valid = dataset_valid.batch(self.batch_size * self.valid_mult)

        iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        reader_op = iterator.get_next()
        train_init_op = iterator.make_initializer(dataset_train)
        valid_init_op = iterator.make_initializer(dataset_valid)

        return train_init_op, valid_init_op, reader_op
