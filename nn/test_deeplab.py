import time
import numpy as np
import ersaPath
from nn import deeplab, hook, nn_utils
from preprocess import patchExtractor
from reader import dataReaderSegmentation, reader_utils
from collection import collectionMaker, collectionEditor

# define parameters
class_num = 2
patch_size = (321, 321)
tile_size = (5000, 5000)
lr = 1e-5
ds = 40
dr = 0.1
epochs = 6
bs = 5
ds_name = 'Inria'
suffix = 'test'
sfn = 32
n_train = 8000
valid_mult = 5
n_valid = 1000//bs//valid_mult
gpu = 0
verb_step = 200
save_epoch = 5
nn_utils.set_gpu(gpu)

# define network
unet = deeplab.DeepLab(class_num, patch_size, suffix=suffix, learn_rate=lr, decay_step=ds, decay_rate=dr,
                       epochs=epochs, batch_size=bs)
overlap = unet.get_overlap()

cm = collectionMaker.read_collection(raw_data_path=r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles',
                                     field_name='austin,chicago,kitsap,tyrol-w,vienna',
                                     field_id=','.join(str(i) for i in range(37)),
                                     rgb_ext='RGB',
                                     gt_ext='GT',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name=ds_name)
gt_d255 = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['GT', 'gt_d255']).\
    run(force_run=False, file_ext='png', d_type=np.uint8,)
cm.replace_channel(gt_d255.files, True, ['GT', 'gt_d255'])
cm.print_meta_data()
file_list_train = cm.load_files(field_id=','.join(str(i) for i in range(6, 37)), field_ext='RGB,gt_d255')
file_list_valid = cm.load_files(field_id=','.join(str(i) for i in range(6)), field_ext='RGB,gt_d255')
chan_mean = cm.meta_data['chan_mean'][:3]

patch_list_train = patchExtractor.PatchExtractor(patch_size, tile_size, ds_name+'_train', overlap, overlap//2).\
    run(file_list=file_list_train, file_exts=['jpg', 'png'], force_run=False).get_filelist()
patch_list_valid = patchExtractor.PatchExtractor(patch_size, tile_size, ds_name+'_valid', overlap, overlap//2).\
    run(file_list=file_list_valid, file_exts=['jpg', 'png'], force_run=False).get_filelist()

train_init_op, valid_init_op, reader_op = \
    dataReaderSegmentation.DataReaderSegmentationTrainValid(
        patch_size, patch_list_train, patch_list_valid, batch_size=bs, chan_mean=chan_mean,
        aug_func=[reader_utils.image_flipping, reader_utils.image_rotating],
        random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=valid_mult).read_op()
feature, label = reader_op

unet.create_graph(feature, sfn)
unet.compile(feature, label, n_train, n_valid, patch_size, ersaPath.PATH['model'], par_dir='test', loss_type='xent')
train_hook = hook.ValueSummaryHook(verb_step, [unet.loss, unet.lr_op], value_names=['train_loss', 'learning_rate'],
                                   print_val=[0])
model_save_hook = hook.ModelSaveHook(unet.get_epoch_step()*save_epoch, unet.ckdir)
valid_loss_hook = hook.ValueSummaryHook(unet.get_epoch_step(), [unet.loss],
                                        value_names=['valid_loss'], log_time=True, run_time=unet.n_valid)
valid_iou_hook = hook.IoUSummaryHook(unet.get_epoch_step(), unet.loss_iou, log_time=True, run_time=unet.n_valid,
                                     cust_str='\t')
image_hook = hook.ImageValidSummaryHook(unet.get_epoch_step(), unet.valid_images, feature, label, unet.pred,
                                        nn_utils.image_summary, img_mean=chan_mean)
start_time = time.time()
unet.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, valid_iou_hook, image_hook],
           train_init=train_init_op, valid_init=valid_init_op)
print('Duration: {:.3f}'.format((time.time() - start_time)/3600))
