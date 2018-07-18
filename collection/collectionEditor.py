import os
from tqdm import tqdm
import utils
import processBlock


class SingleChanMult(processBlock.BasicProcess):
    def __init__(self, path, mult_factor, field_ext_pair):
        name = 'chan_mult_{:.5f}'.format(mult_factor).replace('.', 'p')
        func = self.process
        self.mult_factor = mult_factor
        self.field_ext_pair = field_ext_pair
        super().__init__(name, path, func)

    def process(self, **kwargs):
        assert len(kwargs['file_list'][0]) == 1
        pbar = tqdm(kwargs['file_list'])
        for img_file in pbar:
            save_name = img_file[0].replace(self.field_ext_pair[0], self.field_ext_pair[1])
            pbar.set_description('Making {}'.format(os.path.basename(save_name)))
            img = utils.load_file(img_file[0])
            img = (img * self.mult_factor).astype(kwargs['d_type'])
            utils.save_file(save_name, img)
