import os
import numpy as np
from tqdm import tqdm
import utils
import processBlock


def make_grid(tile_size, patch_size, overlap):
    # this function should be changed in the subclass if desired.
    # Default behavior is to extract chips at fixed locations.
    # Output coordinates for Y,X as a list (not two lists)

    # make the grid of indexes at which to extract patches
    # get the boundary of the tile given the patch size
    max_im0 = tile_size[0] - patch_size[0] - 1
    max_im1 = tile_size[1] - patch_size[1] - 1
    # overlap by number of pixels specified
    # add the last possible patch to ensure that you are covering all the pixels in the image
    patch_grid_y = list(range(0, max_im0, patch_size[0] - overlap))
    patch_grid_y = patch_grid_y + [max_im0]
    patch_grid_x = list(range(0, max_im1, patch_size[1] - overlap))
    patch_grid_x = patch_grid_x + [max_im1]

    y, x = np.meshgrid(patch_grid_y, patch_grid_x)
    return list(zip(y.flatten(), x.flatten()))


class PatchExtractor(processBlock.BasicProcess):
    def __init__(self, patch_size, tile_size, ds_name, overlap=0, pad=0, name='patch_extractor'):
        self.patch_size = np.array(patch_size, dtype=np.int32)
        self.tile_size = np.array(tile_size, dtype=np.int32)
        self.overlap = overlap
        self.pad = pad
        pe_name = '{}_w{}h{}_overlap{}_pad{}'.format(name, self.patch_size[0], self.patch_size[1], self.overlap, self.pad)
        path = utils.get_block_dir('data', [name, ds_name, pe_name])
        super().__init__(pe_name, path, func=self.process)

    def process(self, **kwargs):
        assert len(kwargs['file_exts']) == len(kwargs['file_list'][0])
        grid_list = make_grid(self.tile_size + 2*self.pad, self.patch_size, self.overlap)
        pbar = tqdm(kwargs['file_list'])
        record_file = open(os.path.join(self.path, 'file_list.txt'), 'w')
        for files in pbar:
            pbar.set_description('Extracting {}'.format(os.path.basename(files[0])))
            patch_list = []
            for f, ext in zip(files, kwargs['file_exts']):
                patch_list_ext = []
                img = utils.load_file(f)
                # pad image first if it is necessary
                if self.pad > 0:
                    img = utils.pad_image(img, self.pad)
                # extract images
                for y, x in grid_list:
                    patch_name = '{}_y{}x{}.{}'.format(os.path.basename(f).split('.')[0], int(y), int(x), ext)
                    patch_name = os.path.join(self.path, patch_name)
                    patch = utils.crop_image(img, y, x, self.patch_size[0], self.patch_size[1])
                    utils.save_file(patch_name, patch.astype(np.uint8))
                    patch_list_ext.append(patch_name)
                patch_list.append(patch_list_ext)
            patch_list = utils.rotate_list(patch_list)
            for items in patch_list:
                record_file.write('{}\n'.format(' '.join(items)))
        record_file.close()


if __name__ == '__main__':
    from collection import collectionMaker
    cm = collectionMaker.CollectionMaker(
        r'/media/ei-edl01/user/bh163/ersa_data_temp/inria_toy',
        field_name='austin|chicago|tyrol-w', field_id='1,2', rgb_ext='RGB', gt_ext='gt_d255',
        file_ext='tif')
    file_list = cm.get_file_selection('austin|chicago|tyrol-w', '1,2', 'RGB,gt_d255', 'tif,tif')

    pe = PatchExtractor((550, 550), (5000, 5000), 'inria_toy', pad=0)
    pe.run(file_list=file_list, file_exts=['jpg', 'png'], force_run=True)
