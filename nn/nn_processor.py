import os
import time
import numpy as np
import tensorflow as tf
import ersa_utils
import processBlock
from nn import nn_utils, nn_evals
from reader import dataReaderSegmentation
from preprocess import patchExtractor


class NNEstimatorSegment(processBlock.BasicProcess):
    """
    Evaluate a segmentation network
    """
    def __init__(self, model, file_list, input_size, tile_size, batch_size, img_mean, model_dir,
                 ds_name='default', save_result_parent_dir=None, name='nn_estimator_segment',
                 gpu=None, verb=True, load_epoch_num=None, best_model=False, truth_val=1, score_results=False,
                 split_char='_', **kwargs
                 ):
        """
        :param model: model to be evaluated
        :param file_list: evaluation file list
        :param input_size: dimension of the input to the network
        :param tile_size: dimension of the single evaluation file
        :param batch_size: batch size
        :param img_mean: mean of each channel
        :param model_dir: path to the pretrained model
        :param ds_name: name of the dataset
        :param save_result_parent_dir: parent directory to where the result will be stored
        :param name: name of the process
        :param gpu: which gpu to run the model, default to use all the gpus available
        :param verb: if True, print out message when evaluating
        :param load_epoch_num: which epoch's ckpt to load
        :param best_model: if True, load the model with best performance on the validation set
        :param truth_val: value of H1 pixel in gt
        :param score_results: if False, no gt used to score results
        :param split_char: character used to split file name
        :param kwargs: other parameters
        """
        self.model = model
        self.file_list = file_list
        self.input_size = input_size
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.img_mean = img_mean
        self.model_dir = model_dir
        self.model_name = model_dir.split('/')[-1]
        if save_result_parent_dir is None:
            self.score_save_dir = ersa_utils.get_block_dir('eval', [self.model_name, ds_name])
        else:
            self.score_save_dir = ersa_utils.get_block_dir('eval', [save_result_parent_dir, self.model_name, ds_name])
        self.gpu = gpu
        self.verb = verb
        self.load_epoch_num = load_epoch_num
        self.best_model = best_model
        self.truth_val = truth_val
        self.score_results = score_results
        self.split_char = split_char
        self.kwargs = kwargs
        self.compute_shape_flag = False  # recompute tile shape or not

        super().__init__(name, self.score_save_dir, func=self.process)

    def process(self):
        """
        Evaluate the network
        :return:
        """
        nn_utils.set_gpu(self.gpu)

        if self.score_results:
            with open(os.path.join(self.score_save_dir, 'result.txt'), 'w'):
                pass
        iou_record = []

        # prepare the reader
        if self.score_results:
            init_op, reader_op = dataReaderSegmentation.DataReaderSegmentationTesting(
                self.input_size, self.tile_size, self.file_list, overlap=self.model.get_overlap(),
                pad=self.model.get_overlap() // 2, batch_size=self.batch_size, chan_mean=self.img_mean,
                is_train=False, has_gt=True, random=False,
                gt_dim=1, include_gt=True).read_op()
            feature, label = reader_op
            self.model.create_graph(feature, **self.kwargs)
        else:
            init_op, reader_op = dataReaderSegmentation.DataReaderSegmentationTesting(
                self.input_size, self.tile_size, self.file_list, overlap=self.model.get_overlap(),
                pad=self.model.get_overlap() // 2, batch_size=self.batch_size, chan_mean=self.img_mean,
                is_train=False, has_gt=False, random=False,
                gt_dim=0, include_gt=False).read_op()
            feature = reader_op
            self.model.create_graph(feature[0], **self.kwargs)
        pad = self.model.get_overlap()

        for file_cnt, (file_name_list) in enumerate(self.file_list):
            file_name_truth = None
            if self.score_results:
                file_name, file_name_truth = file_name_list
                tile_name = os.path.basename(file_name_truth).split(self.split_char)[0]
            else:
                file_name = file_name_list[0]
                tile_name = os.path.basename(file_name).split(self.split_char)[0]
            if self.verb:
                print('Evaluating {} ... '.format(tile_name))

            # read tile size if no tile size is given
            if self.tile_size is None or self.compute_shape_flag:
                self.compute_shape_flag = True
                tile = ersa_utils.load_file(file_name)
                self.tile_size = tile.shape[:2]

            start_time = time.time()

            # run the model
            if self.model.config is None:
                self.model.config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=self.model.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.model.load(self.model_dir, sess, epoch=self.load_epoch_num, best_model=self.best_model)
                result = self.model.test_sample(sess, init_op[file_cnt])
            image_pred = patchExtractor.unpatch_block(result,
                                                      tile_dim=[self.tile_size[0] + pad, self.tile_size[1] + pad],
                                                      patch_size=self.input_size, tile_dim_output=self.tile_size,
                                                      patch_size_output=[self.input_size[0] - pad,
                                                                         self.input_size[1] - pad],
                                                      overlap=pad)
            if self.compute_shape_flag:
                self.tile_size = None

            pred = nn_utils.get_pred_labels(image_pred) * self.truth_val

            if self.score_results:
                truth_label_img = ersa_utils.load_file(file_name_truth)
                iou = nn_utils.iou_metric(truth_label_img, pred, divide_flag=True)
                iou_record.append(iou)

                duration = time.time() - start_time
                if self.verb:
                    print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0] / iou[1], duration))

                # save results
                pred_save_dir = os.path.join(self.score_save_dir, 'pred')
                ersa_utils.make_dir_if_not_exist(pred_save_dir)
                ersa_utils.save_file(os.path.join(pred_save_dir, '{}.png'.format(tile_name)), pred.astype(np.uint8))
                if self.score_results:
                    with open(os.path.join(self.score_save_dir, 'result.txt'), 'a+') as file:
                        file.write('{} {}\n'.format(tile_name, iou))

        if self.score_results:
            iou_record = np.array(iou_record)
            mean_iou = np.sum(iou_record[:, 0]) / np.sum(iou_record[:, 1])
            print('Overall mean IoU={:.3f}'.format(mean_iou))
            with open(os.path.join(self.score_save_dir, 'result.txt'), 'a+') as file:
                file.write('{}'.format(mean_iou))

    def load_results(self):
        """
        load all the results computed by this process
        :return: tile based iou, field based iou and overall iou
        """
        print('Summary of results:')
        result_name = os.path.join(self.score_save_dir, 'result.txt')
        result = ersa_utils.load_file(result_name)
        tile_dict, field_dict, overall = nn_utils.read_iou_from_file(result)
        for key, val in field_dict.items():
            field_str = ersa_utils.make_center_string('=', 50, '{}={:.2f}'.format(key, val * 100))
            print(field_str)
            for key_tile, val_tile in tile_dict.items():
                if key in key_tile:
                    print('{}={:.2f}'.format(key_tile, val_tile * 100))
        print(ersa_utils.make_center_string('=', 50, 'Overall={:.2f}'.format(overall * 100)))
        return tile_dict, field_dict, overall


class NNEstimatorSegmentScene(processBlock.BasicProcess):
    """
    Evaluate a segmentation network for scene segmentation
    """
    def __init__(self, model, file_list, model_dir, init_op, reader_op, ds_name='default', save_result_parent_dir=None,
                 name='nn_estimator_segment_scene', gpu=None, verb=True, load_epoch_num=None, best_model=False,
                 score_result=False, split_char='_', post_func=None, save_func=None, ignore_label=(), **kwargs
                 ):
        """
        :param model: model to be evaluated
        :param file_list: evaluation file list
        :param model_dir: path to the pretrained model
        :param init_op: initialize operation for the dataset reader
        :param reader_op: reader_op, return feature and label if scoring the results
        :param ds_name: name of the dataset
        :param save_result_parent_dir: parent directory to where the result will be stored
        :param name: name of the process
        :param gpu: which gpu to run the model, default to use all the gpus available
        :param verb: if True, print out message when evaluating
        :param load_epoch_num: which epoch's ckpt to load
        :param best_model: if True, load the model with best performance on the validation set
        :param score_result: if False, no gt used to score results
        :param split_char: character used to split file name
        :param post_func: post processing function to make prediction
        :param save_func: post processing function to make visible prediction map
        :param ignore_label: labels to be ignored at scoring
        :param kwargs: other parameters
        """
        self.model = model
        self.file_list = file_list
        self.model_dir = model_dir
        self.init_op = init_op
        self.reader_op = reader_op
        self.model_name = model_dir.split('/')[-1]
        if save_result_parent_dir is None:
            self.score_save_dir = ersa_utils.get_block_dir('eval', [self.model_name, ds_name])
        else:
            self.score_save_dir = ersa_utils.get_block_dir('eval', [save_result_parent_dir, self.model_name, ds_name])
        self.gpu = gpu
        self.verb = verb
        self.load_epoch_num = load_epoch_num
        self.best_model = best_model
        self.score_results = score_result
        self.split_char = split_char
        self.post_func = post_func
        self.save_func = save_func
        self.ignore_label = ignore_label
        self.kwargs = kwargs

        super().__init__(name, self.score_save_dir, func=self.process)

    def process(self):
        """
        Evaluate the network
        :return:
        """
        nn_utils.set_gpu(self.gpu)

        if self.score_results:
            with open(os.path.join(self.score_save_dir, 'result.txt'), 'w'):
                pass
        iou_record = []

        # prepare the reader
        if self.score_results:
            feature, label = self.reader_op
            self.model.create_graph(feature, **self.kwargs)
        else:
            feature = self.reader_op
            self.model.create_graph(feature[0], **self.kwargs)

        # run the model
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run([init, self.init_op])
            self.model.load(self.model_dir, sess, epoch=self.load_epoch_num, best_model=self.best_model)
            for file_cnt, (file_name_list) in enumerate(self.file_list):
                file_name_truth = None
                if self.score_results:
                    file_name, file_name_truth = file_name_list
                    tile_name = os.path.basename(file_name_truth).split(self.split_char)[0]
                else:
                    file_name = file_name_list[0]
                    tile_name = os.path.basename(file_name).split(self.split_char)[0]
                if self.verb:
                    print('Evaluating {} ... '.format(tile_name))
                start_time = time.time()

                result = sess.run(self.model.output)
                pred = np.argmax(np.squeeze(result, axis=0), axis=-1)
                if self.post_func is not None:
                    pred = self.post_func(pred)
                if self.save_func is not None:
                    save_img = self.save_func(pred)
                else:
                    save_img = pred

                if self.score_results:
                    truth_label_img = ersa_utils.load_file(file_name_truth)
                    iou = nn_evals.mean_IU(pred, truth_label_img, self.ignore_label)
                    iou_record.append(iou)

                    duration = time.time() - start_time
                    if self.verb:
                        print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou, duration))

                    # save results
                    pred_save_dir = os.path.join(self.score_save_dir, 'pred')
                    ersa_utils.make_dir_if_not_exist(pred_save_dir)
                    ersa_utils.save_file(os.path.join(pred_save_dir, '{}.png'.format(tile_name)), save_img.astype(np.uint8))
                    if self.score_results:
                        with open(os.path.join(self.score_save_dir, 'result.txt'), 'a+') as file:
                            file.write('{} {}\n'.format(tile_name, iou))

        if self.score_results:
            iou_record = np.array(iou_record)
            mean_iou = np.mean(iou_record)
            print('Overall mean IoU={:.3f}'.format(mean_iou))
            with open(os.path.join(self.score_save_dir, 'result.txt'), 'a+') as file:
                file.write('{}'.format(mean_iou))

    def load_results(self):
        """
        load all the results computed by this process
        :return: tile based iou, field based iou and overall iou
        """
        print('Summary of results:')
        result_name = os.path.join(self.score_save_dir, 'result.txt')
        result = ersa_utils.load_file(result_name)
        tile_dict, field_dict, overall = nn_utils.read_iou_from_file(result)
        for key, val in field_dict.items():
            field_str = ersa_utils.make_center_string('=', 50, '{}={:.2f}'.format(key, val * 100))
            print(field_str)
            for key_tile, val_tile in tile_dict.items():
                if key in key_tile:
                    print('{}={:.2f}'.format(key_tile, val_tile * 100))
        print(ersa_utils.make_center_string('=', 50, 'Overall={:.2f}'.format(overall * 100)))
        return tile_dict, field_dict, overall
