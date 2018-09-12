import os
import cv2
import numpy as np
from tqdm import tqdm
import ersa_utils
import processBlock


class GammaAdjust(processBlock.BasicProcess):
    """
    Adjust images' brightness with O = I^(1/G)
    """
    def __init__(self, gamma, path=None, ds_name=None, name='gamma_adjust'):
        """
        :param path: path to save the adjusted images
        :param ds_name: where to save data if path is None
        :param gamma: gamma to adjust images' brightness
        """
        self.gamma = gamma
        pe_name = '{}_gamma{}'.format(name, self.gamma)
        if path is None:
            assert ds_name is not None
            path = ersa_utils.get_block_dir('data', ['preprocess', ds_name, name])
        super().__init__(pe_name, path, func=self.process)

    def process(self, **kwargs):
        """
        Extract the patches
        :param kwargs:
            file_list: list of files
        :return:
        """
        pbar = tqdm(kwargs['file_list'])
        for files in pbar:
            pbar.set_description('Adjusting {} with gamma={}'.format(os.path.basename(files), self.gamma))
            img = ersa_utils.load_file(files)
            img = self.adjust_gamma(img, self.gamma)
            old_name = os.path.basename(files)
            tile_name = '{}_gamma{}.{}'.format(old_name.split('.')[0], ersa_utils.float2str(self.gamma),
                                               old_name.split('.')[1])
            tile_name = os.path.join(self.path, tile_name)
            ersa_utils.save_file(tile_name, img.astype(np.uint8))

    @staticmethod
    def adjust_gamma(img, gamma):
        """
        Adjust the gamma for given image
        :param img:
        :param gamma:
        :return:
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        img_adjust = cv2.LUT(img, table)
        return img_adjust
