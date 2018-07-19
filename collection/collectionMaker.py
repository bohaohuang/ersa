import os
import re
import numpy as np
from glob import glob
from tqdm import tqdm
import utils
import processBlock


def get_channel_mean(parent_dir, img_files):
    """
    Get mean of all channels, means are usually used in neural network training
    If img_files has full path, parent_dir actually doesn't matter here, I add this to make this function usable by
    other cases even without collection
    :param parent_dir: parent directory of all image files
    :param img_files: list of image file names
    :return: a numpy array of means of each channel
    """
    def get_mean(d, channel_num):
        if len(data.shape) == 2:
            return np.mean(d)
        else:
            channel_mean = np.zeros(channel_num)
            for i in range(channel_num):
                channel_mean[i] = np.mean(d[:, :, i])
            return channel_mean

    # get channel number
    temp_file = img_files[0]
    chan_num = 0
    chan_record = [0]
    for f in temp_file:
        c_num = utils.get_img_channel_num(os.path.join(parent_dir, f))
        chan_num += c_num
        chan_record.append(chan_num)
    # compute channel mean
    chan_mean = np.zeros(chan_num)
    pbar = tqdm(img_files)
    for file_item in pbar:
        pbar.set_description('Calculating means')
        for cnt, f in enumerate(file_item):
            data = utils.load_file(os.path.join(parent_dir, f))
            chan_mean[chan_record[cnt]:chan_record[cnt+1]] += get_mean(data, chan_record[cnt+1]-chan_record[cnt])
    chan_mean = chan_mean / len(img_files)
    return chan_mean


def read_collection(clc_name=None, clc_dir=None):
    """
    Read and initialize a collection from a directory
    :param clc_name: name of the collection
    :param clc_dir: directory to the collection
    :return: the collection object, assertion error if no process hasn't completed
    """
    if clc_dir is None:
        assert clc_name is not None
        clc_dir = utils.get_block_dir('data', ['collection', clc_name])
    # check if finish
    if processBlock.BasicProcess('collection_maker', clc_dir).check_finish():
        # read metadata
        meta_data = utils.load_file(os.path.join(clc_dir, 'meta.pkl'))
        # create collection
        cm = CollectionMaker(meta_data['raw_data_path'], meta_data['field_name'], meta_data['field_id'],
                             meta_data['rgb_ext'], meta_data['gt_ext'], meta_data['file_ext'],
                             meta_data['files'], meta_data['clc_name'], force_run=False)
        return cm
    else:
        raise AssertionError('You need to make collection first')


class CollectionMaker(object):
    """
    Collection is the basic class for processing and organizing data files
    Collection just records status of the file selection. It does not make a copy of the files.
    But you can use collection to modify/add files at the raw data path. The modified data will be stored locally
    """
    def __init__(self, raw_data_path, field_name, field_id, rgb_ext, gt_ext, file_ext, files=None,
                 clc_name=None, force_run=False):
        """
        Create a collection
        :param raw_data_path: path to where the data are stored
        :param field_name: could be name of the cities, or other prefix of the images
        :param field_id: could be id of the tiles, or other suffix of the images
        :param rgb_ext: name extensions that indicates the images are not ground truth, use ',' to separate if you have
                        multiple extensions
        :param gt_ext: name extensions that indicates the images are ground truth, you can only have at most one ground
                       truth extension
        :param file_ext: extension of the files, use ',' to separate if you have multiple extensions, if all the files
                         have the same extension, you only need to specify one
        :param files: files in the raw_data_path, can be specified by user to exclude some of the raw files, if it is
                      None, all files will be found automatically
        :param clc_name: name of the collection, if set to None, it will be the name of the raw_data_path folder
        :param force_run: force run the collection maker even if it already exists
        """
        self.raw_data_path = raw_data_path
        self.field_name = utils.str2list(field_name, d_type=str)    # name of the 'cities' to include in the collection
        self.field_id = utils.str2list(field_id, d_type=str)        # id of the 'cities' to include in the collection
        self.rgb_ext = utils.str2list(rgb_ext, d_type=str)
        self.gt_ext = gt_ext
        if len(gt_ext) == 0:
            has_gt_ext = 0
        else:
            has_gt_ext = 1
        self.file_ext = utils.str2list(file_ext, d_type=str)
        assert len(self.file_ext) == 1 or len(self.file_ext) == len(self.rgb_ext) + has_gt_ext
        if len(self.file_ext) == 1:
            self.file_ext = [self.file_ext[0] for _ in range(len(self.rgb_ext) + has_gt_ext)]
        if clc_name is None:
            clc_name = os.path.basename(raw_data_path)
        self.clc_name = clc_name                                    # name of the collection
        self.clc_dir = self.get_dir()                               # directory to store the collection
        self.force_run = force_run

        # make collection
        if files is None:
            self.files = sorted(glob(os.path.join(self.raw_data_path, '*.*')))
        else:
            self.files = files
        self.clc_pb = processBlock.BasicProcess('collection_maker', self.clc_dir, self.make_collection)
        self.clc_pb.run(self.force_run)
        self.meta_data = self.read_meta_data()

    @staticmethod
    def make_regexp(field_name, field_id, field_txt, file_ext):
        """
        make regex pattern for filter out the selected files
        :param field_name: name of the 'cities' to include in the collection
        :param field_id: id of the 'cities' to include in the collection
        :param field_txt: extension of the field, e.g. 'GT' or 'RGB
        :param file_ext: extension of the images
        :return: regex pattern string
        """
        regexp = r'.*({}).*({}).*{}.*.{}'.format('|'.join(field_name), '|'.join(field_id), field_txt, file_ext)
        return regexp

    def get_dir(self):
        """
        Get or create directory of this collection
        :return: directory of the collection
        """
        return utils.get_block_dir('data', ['collection', self.clc_name])

    def get_files(self, regexp, full_path=False):
        """
        Get all files fit specific regular expression
        :param regexp: regular expression
        :param full_path: if it is true, return full path of the file
        :return: list of selected files
        """
        if full_path:
            files = [f for f in self.files if re.match(regexp, f)]
        else:
            files = [os.path.basename(f) for f in self.files if re.match(regexp, f)]
        return files

    def get_file_selection(self, field_name, field_id, field_ext, file_ext):
        """
        Get list of lists of files selected by given field names, field ids, field extensions and file extensions
        :param field_name: name of the fields (e.g., city names)
        :param field_id: id of the fields (e.g., tile numbers)
        :param field_ext: extension of the fields (e.g., RGB)
        :param file_ext: file extension (e.g., tif)
        :return: list of lists, where each row is file names of same place with different files
        """
        field_name = utils.str2list(field_name, d_type=str)
        field_id = utils.str2list(field_id, d_type=str)
        field_ext = utils.str2list(field_ext, d_type=str)
        file_ext = utils.str2list(file_ext, d_type=str)
        if len(file_ext) == 1:
            file_ext = [file_ext[0] for _ in range(len(field_ext))]
        file_selection = []
        for field, file in zip(field_ext, file_ext):
            regexp = self.make_regexp(field_name, field_id, field, file)
            file_selection.append(self.get_files(regexp, full_path=True))
        file_selection = utils.rotate_list(file_selection)
        return file_selection

    def make_collection(self):
        """
        Make meta data of the collection, including tile dimension, ground truth and rgb files list
        means of all channels in rgb files
        :return:
        """
        # collect files selection
        gt_regexp = self.make_regexp(self.field_name, self.field_id, self.gt_ext, self.file_ext[-1])
        gt_files = self.get_files(gt_regexp, full_path=True)
        # rgb data can have multiple channels and be stored in multiple files
        rgb_files = []
        for cnt, ext in enumerate(self.rgb_ext):
            rgb_regexp = self.make_regexp(self.field_name, self.field_id, ext, self.file_ext[cnt])
            rgb_files.append(self.get_files(rgb_regexp, full_path=True))
        rgb_files = utils.rotate_list(rgb_files)

        # make meta_data
        tile_dim = utils.load_file(rgb_files[0][0]).shape[:2]
        channel_mean = get_channel_mean(self.raw_data_path, rgb_files)

        meta_data = {'raw_data_path': self.raw_data_path,
                     'field_name': self.field_name,
                     'field_id': self.field_id,
                     'rgb_ext': self.rgb_ext,
                     'gt_ext': self.gt_ext,
                     'file_ext': self.file_ext,
                     'clc_name': self.clc_name,
                     'tile_dim': tile_dim,
                     'gt_files': gt_files,
                     'rgb_files': rgb_files,
                     'chan_mean': channel_mean,
                     'files': self.files}
        utils.save_file(os.path.join(self.clc_dir, 'meta.pkl'), meta_data)

    def read_meta_data(self):
        """
        Read meta data of the collection
        :return:
        """
        meta_data = utils.load_file(os.path.join(self.clc_dir, 'meta.pkl'))
        return meta_data

    def print_meta_data(self):
        """
        Print the meta data in a human readable format
        :return:
        """
        print(utils.make_center_string('=', 88, self.clc_name))
        skip_keys = ['gt_files', 'rgb_files', 'rgb_ext', 'gt_ext', 'file_ext', 'files']
        for key, val in self.meta_data.items():
            if key in skip_keys:
                continue
            print('{}: {}'.format(key, val))
        print('Source file: {}'.format(' '.join(['*{}*.{}'.format(ext1, ext2)
                                                 for ext1, ext2 in zip(self.rgb_ext, self.file_ext)])))
        if len(self.gt_ext) > 0:
            print('GT file: {}'.format('*{}*.{}'.format(self.gt_ext, self.file_ext[-1])))
        print(utils.make_center_string('=', 88))

    def replace_channel(self, files, is_gt, field_ext_pair, new_file_ext=None):
        """
        Replace a channel in the collection, then remake the meta data
        :param files: files correspond to the replaced channel
        :param is_gt: is replacing ground truth or not
        :param field_ext_pair: old filed extension and new field extension, should be a list
        :param new_file_ext: new file extension, if it has been changed
        :return:
        """
        new_file_ext = files[0].split('.')[-1]
        if is_gt:
            self.gt_ext = field_ext_pair[1]
            self.file_ext[-1] = new_file_ext
        else:
            self.rgb_ext = [field_ext_pair[1] if ext == field_ext_pair[0] else ext for ext in self.rgb_ext]
            self.file_ext = [new_file_ext if self.rgb_ext[i] == field_ext_pair[0] else self.file_ext[i]
                             for i in range(len(self.rgb_ext))]
        self.files += files
        self.clc_pb.run(True)
        self.meta_data = self.read_meta_data()

    def add_channel(self, files, new_field_ext):
        """
        Add a channel to the collection, then remake the meta data
        :param files: files correspond to the added channel
        :param new_field_ext: new field extension
        :return:
        """
        new_file_ext = files[0].split('.')[-1]
        self.rgb_ext.append(new_field_ext)
        if len(self.gt_ext) == 0:
            self.file_ext.append(new_file_ext)
        else:
            self.file_ext.append(self.file_ext[-1])
            self.file_ext[-2] = new_file_ext
        # update files, then re-make meta data
        # need to re-compute mean here, can be improved in the future
        self.files += files
        self.clc_pb.run(True)
        self.meta_data = self.read_meta_data()

    def load_files(self, field_name=None, field_id=None, field_ext=None):
        """
        Load all files meet the given filters, each one above can be left blank
        :param field_name: name of the field
        :param field_id: name of the id
        :param field_ext: name of the field extension
        :return:
        """
        if field_name is None:
            field_name = self.field_name
        if field_id is None:
            field_id = self.field_id
        field_ext = utils.str2list(field_ext, d_type=str)
        files = []
        for fe in field_ext:
            if fe in self.rgb_ext:
                select_file_ext = self.file_ext[self.rgb_ext.index(fe)]
            else:
                select_file_ext = self.file_ext[-1]
            if type(field_id) is not str:
                field_id = str(field_id)
            file = utils.rotate_list(self.get_file_selection(field_name, field_id, fe, select_file_ext))[0]
            files.append(file)

        files = utils.rotate_list(files)
        if len(files) == 1:
            if len(files[0]) == 1:
                # only one file been requested
                return files[0][0]
        return files
