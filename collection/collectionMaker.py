import os
import re
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
import utils
import processBlock


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


def get_channel_mean(parent_dir, img_files):
    """
    Get mean of all channels, means are usually used in neural network training
    :param parent_dir: parent directory of all image files
    :param img_files: list of image file names
    :return: a numpy array of means of each channel
    """
    def get_mean(d, chan_num):
        if len(data.shape) == 2:
            return np.mean(d)
        else:
            chan_mean = np.zeros(chan_num)
            for i in range(chan_num):
                chan_mean[i] = np.mean(d[:, :, i])
            return chan_mean

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


class CollectionMaker(object):
    def __init__(self, raw_data_path, field_name, field_id, rgb_ext, gt_ext, file_ext, clc_name=None, force_run=False):
        self.raw_data_path = raw_data_path
        self.field_name = utils.str2list(field_name, d_type=str)    # name of the 'cities' to include in the collection
        self.field_id = utils.str2list(field_id, d_type=str)        # id of the 'cities' to include in the collection
        self.rgb_ext = utils.str2list(rgb_ext, d_type=str)
        self.gt_ext = gt_ext
        self.file_ext = utils.str2list(file_ext, d_type=str)
        assert len(self.file_ext) == 1 or len(self.file_ext) == len(self.rgb_ext) + 1
        if len(self.file_ext) == 1:
            self.file_ext = [self.file_ext[0] for i in range(len(self.rgb_ext) + 1)]
        if clc_name is None:
            clc_name = os.path.basename(raw_data_path)
        self.clc_name = clc_name                                    # name of the collection
        self.clc_dir = self.get_dir()                               # directory to store the collection
        self.force_run = force_run

        # make collection
        self.clc_pb = processBlock.BasicProcess('collection_maker', self.clc_dir, self.make_collection)
        self.clc_pb.run(self.force_run)
        self.meta_data = self.read_meta_data()

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
        files = sorted(glob(os.path.join(self.raw_data_path, '*.*')))
        if full_path:
            files = [f for f in files if re.match(regexp, f)]
        else:
            files = [os.path.basename(f) for f in files if re.match(regexp, f)]
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
            file_ext = [file_ext[0] for i in range(len(field_ext))]
        file_selection = []
        for field, file in zip(field_ext, file_ext):
            regexp = make_regexp(field_name, field_id, field, file)
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
        gt_regexp = make_regexp(self.field_name, self.field_id, self.gt_ext, self.file_ext[-1])
        gt_files = self.get_files(gt_regexp)
        # rgb data can have multiple channels and be stored in multiple files
        rgb_files = []
        for cnt, ext in enumerate(self.rgb_ext):
            rgb_regexp = make_regexp(self.field_name, self.field_id, ext, self.file_ext[cnt])
            rgb_files.append(self.get_files(rgb_regexp))
        rgb_files = utils.rotate_list(rgb_files)

        # make meta_data
        tile_dim = imageio.imread(os.path.join(self.raw_data_path, rgb_files[0][0])).shape[:2]
        channel_mean = get_channel_mean(self.raw_data_path, rgb_files)

        meta_data = {'tile_dim': tile_dim,
                     'gt_files': gt_files,
                     'rgb_files': rgb_files,
                     'chan_mean': channel_mean}
        utils.save_file(os.path.join(self.clc_dir, 'meta.pkl'), meta_data)

    def read_meta_data(self):
        """
        Read meta data of the collection
        :return:
        """
        meta_data = utils.load_file(os.path.join(self.clc_dir, 'meta.pkl'))
        return meta_data

    def replace_channel(self, is_gt, field_ext_pair):
        """
        Replace a channel in the collection, then remake the meta data
        :param is_gt:
        :param field_ext_pair:
        :return:
        """
        if is_gt:
            self.gt_ext = field_ext_pair[1]
        else:
            self.rgb_ext = [field_ext_pair[1] if ext == field_ext_pair[0] else ext for ext in self.rgb_ext]
        self.clc_pb.run(True)
        self.meta_data = self.read_meta_data()

    def add_channel(self):
        # TODO
        pass


if __name__ == '__main__':
    from collection import collectionEditor
    ds_name = 'inria'
    f_run =False
    if ds_name == 'inria':
        cm = CollectionMaker(r'/media/ei-edl01/user/bh163/ersa_data_temp/inria_toy',
                             field_name='austin|chicago|tyrol-w', field_id='1,2', rgb_ext='RGB', gt_ext='GT',
                             file_ext='tif', force_run=f_run)
        file_list = cm.get_file_selection('austin|chicago|tyrol-w', '1,2', 'GT', 'tif,tif')
        ce = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['GT', 'gt_d255'])
        ce.run(file_list=file_list, d_type=np.uint8, force_run=False)
        cm.replace_channel(True, ['GT', 'gt_d255'])
    else:
        cm = CollectionMaker(r'/media/ei-edl01/user/bh163/ersa_data_temp/um_toy',
                             field_name='RIC', field_id='0,2,13', rgb_ext='RGB,DTM,DSM', gt_ext='gt',
                             file_ext='tif', force_run=f_run)
        print(cm.get_file_selection('RIC', '0,2', 'RGB,DTM', 'tif'))
