import os
import numpy as np
from tqdm import tqdm
from fractions import Fraction
import ersa_utils
import processBlock
from collection import collectionMaker


class SingleChanMult(processBlock.BasicProcess):
    """
    Edit collection channels by multiply a single channel with a constant
    """
    def __init__(self, path, mult_factor, field_ext_pair):
        """
        :param path: directory of the collection
        :param mult_factor: constant number
        :param field_ext_pair: a list where the first element is the field extension to be operated on and the second
                               field extension is the name of the new field
        """
        if mult_factor >= 1:
            name = 'chan_mult_{:.5f}'.format(mult_factor).replace('.', 'p')
        else:
            name = 'chan_mult_{}'.format(str(Fraction(mult_factor).limit_denominator()).replace('/', 'd'))
        func = self.process
        self.mult_factor = mult_factor
        self.field_ext_pair = field_ext_pair
        self.clc = collectionMaker.read_collection(clc_dir=path)
        path = ersa_utils.get_block_dir('data', ['preprocess', os.path.basename(path), name])
        self.files = []
        super().__init__(name, path, func)

    def process(self, **kwargs):
        """
        process to make the new field
        :param kwargs:
            file_list: the list of the files, if not given, use all the files with selected field extension
            file_ext: the new file extension, if not given, use the same as the old one
            d_type: the new data type, if not given, use the same as the old one
        :return:
        """
        if 'file_list' not in kwargs:
            file_list = self.clc.load_files(','.join(self.clc.field_name), ','.join(self.clc.field_id),
                                            self.field_ext_pair[0])
        else:
            file_list = kwargs['file_list']
        assert len(file_list[0]) == 1
        pbar = tqdm(file_list)
        for img_file in pbar:
            save_name = img_file[0].replace(''.join([a for a in self.field_ext_pair[0] if a != '.' and a != '*']),
                                            self.field_ext_pair[1])
            if 'file_ext' in kwargs:
                # user specified a new file extension
                save_name = save_name.replace(save_name.split('.')[-1], kwargs['file_ext'])
            save_name = os.path.join(self.path, os.path.basename(save_name))
            pbar.set_description('Making {}'.format(os.path.basename(save_name)))
            img = ersa_utils.load_file(img_file[0])
            img = (img * self.mult_factor)
            if 'd_type' in kwargs:
                img = img.astype(kwargs['d_type'])
            ersa_utils.save_file(save_name, img)
            self.files.append(save_name)


class SingleChanSwitch(processBlock.BasicProcess):
    """
    Edit collection channels by multiply a single channel with a constant
    """
    def __init__(self, path, switch_dict, field_ext_pair, name):
        """
        :param path: directory of the collection
        :param mult_factor: constant number
        :param field_ext_pair: a list where the first element is the field extension to be operated on and the second
                               field extension is the name of the new field
        """
        func = self.process
        self.switch_dict = switch_dict
        self.field_ext_pair = field_ext_pair
        self.clc = collectionMaker.read_collection(clc_dir=path)
        path = ersa_utils.get_block_dir('data', ['preprocess', os.path.basename(path), name])
        self.files = []
        super().__init__(name, path, func)

    def process(self, **kwargs):
        """
        process to make the new field
        :param kwargs:
            file_list: the list of the files, if not given, use all the files with selected field extension
            file_ext: the new file extension, if not given, use the same as the old one
            d_type: the new data type, if not given, use the same as the old one
        :return:
        """
        if 'file_list' not in kwargs:
            file_list = self.clc.load_files(','.join(self.clc.field_name), ','.join(self.clc.field_id),
                                            self.field_ext_pair[0])
        else:
            file_list = kwargs['file_list']
        assert len(file_list[0]) == 1
        pbar = tqdm(file_list)
        for img_file in pbar:
            save_name = img_file[0].replace(''.join([a for a in self.field_ext_pair[0] if a != '.' and a != '*']),
                                            self.field_ext_pair[1])
            if 'file_ext' in kwargs:
                # user specified a new file extension
                save_name = save_name.replace(save_name.split('.')[-1], kwargs['file_ext'])
            save_name = os.path.join(self.path, os.path.basename(save_name))
            pbar.set_description('Making {}'.format(os.path.basename(save_name)))
            img = ersa_utils.load_file(img_file[0])
            for old_val, new_val in self.switch_dict.items():
                img[np.where(img == old_val)] = new_val
            if 'd_type' in kwargs:
                img = img.astype(kwargs['d_type'])
            ersa_utils.save_file(save_name, img)
            self.files.append(save_name)
