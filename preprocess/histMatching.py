import os
from glob import glob
from tqdm import tqdm
import rio_hist.match
import ersa_utils
import processBlock


class HistMatching(processBlock.BasicProcess):
    """
    Adjust images' brightness with O = I^(1/G)
    """
    def __init__(self, ref_path, path=None, color_space='RGB', ds_name=None, name='hist_matching'):
        """
        :param path: path to save the adjusted images
        :param ds_name: where to save data if path is None
        :param gamma: gamma to adjust images' brightness
        """
        self.ref_path = ref_path
        self.color_space = color_space
        pe_name = '{}_hist_matching_{}'.format(name, self.color_space)
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
            pbar.set_description('Adjusting {} in channels={}'.format(os.path.basename(files), self.color_space))
            old_name = os.path.basename(files)
            tile_name = '{}_hist{}.{}'.format(old_name.split('.')[0], ersa_utils.float2str(self.color_space),
                                               old_name.split('.')[1])
            tile_name = os.path.join(self.path, tile_name)
            rio_hist.match.hist_match_worker(files, self.ref_path, dst_path=tile_name, match_proportion=1,
                                             creation_options='',
                                             bands='1,2,3', color_space=self.color_space, plot=False)

    def get_files(self):
        """
        return extracted files
        :return:
        """
        files = sorted(glob(os.path.join(self.path, '*.*')))
        files = [f for f in files if 'txt' not in f]
        return files
