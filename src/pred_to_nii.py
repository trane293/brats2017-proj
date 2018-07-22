import h5py, os
from modules.configfile import config
import numpy as np
import random as random
random.seed(config['seed'])
np.random.seed(config['seed'])
import shutil
import SimpleITK as sitk
import logging

logging.basicConfig(level=logging.DEBUG)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# Open the prediction HDF5 file
logger.info('Opening prediction HDF5 dataset that holds prediction data')
model_name = 'BRATS_E160--0.78.h5'
pred_filename = os.path.join(config['model_prediction_location'], 'model_predictions_' + model_name)
pred_nii_folder = '.'.join(pred_filename.split('.')[0:-1])
if os.path.isdir(pred_nii_folder):
    x = raw_input('Prediction directory already exists, do you want to overwrite? (y,n)')
    if x.lower() == 'y':
        shutil.rmtree(pred_nii_folder)
        os.makedirs(pred_nii_folder)
    else:
        os.exit(-1)
else:
    os.makedirs(pred_nii_folder)

h5_file = h5py.File(pred_filename, mode='r')
data = h5_file['validation_data']
names = h5_file['validation_data_pat_name']

logger.info('Opening sample file from Training dataset')
sample_img = sitk.ReadImage('/home/anmol/mounts/cedar-rm/scratch/asa224/Datasets/BRATS2018/Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz')

for i in range(data.shape[0]):
    logger.info('Patient {}'.format(i))
    pred = data[i]
    pat_name = names[i]

    pred = np.expand_dims(pred, axis=0)

    # create SITK object
    # Swap axes of the prediction
    pred_sw = np.swapaxes(pred, 4, 3)
    pred_sw = np.swapaxes(pred_sw, 3, 2)

    pred_sw = pred_sw > 0.2
    pred_sw = pred_sw.astype(np.uint16)

    mask_shape = np.shape(pred_sw)
    main_mask = np.zeros((mask_shape[2], mask_shape[3], mask_shape[4]))

    # now we need to build the mask as it came from the data
    # edema = while tumor - tumor core (2)
    # enhancing = enhancing (4)
    # necrosis + non enhancing = tumor core - enhancing (1)
    # for each index
    # 0 = whole tumor (1 + 2 + 4)
    # 1 = enhancing (4)
    # 2 = tumor core (1 + 4)

    edema_mask = np.clip(pred_sw[0, 0] - pred_sw[0, 2], 0, 1)  # WT - TC, 0 - 2
    enhancing_mask = np.clip(pred_sw[0, 1], 0, 1)
    nec_nh_mask = np.clip(pred_sw[0, 2] - pred_sw[0, 1], 0, 1)  # TC - EN

    main_mask[np.where(edema_mask == 1)] = 2
    main_mask[np.where(enhancing_mask == 1)] = 4
    main_mask[np.where(nec_nh_mask == 1)] = 1

    # assert np.max(np.unique(main_mask)) <= 4, 'Segmentation labels may be wrong!'

    sitk_pred_img = sitk.GetImageFromArray(main_mask)
    sitk_pred_img.CopyInformation(sample_img)

    logger.info('Saving prediction as nii.gz file')
    sitk.WriteImage(sitk_pred_img, os.path.join(pred_nii_folder, '{}.nii.gz'.format(pat_name)))