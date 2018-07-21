import h5py
from modules.configfile import config, mount_path_prefix
from modules.training_helpers import *
import cPickle as pickle
import numpy as np
import random as random
random.seed(config['seed'])
np.random.seed(config['seed'])
import importlib
import optparse
import os
import logging
from modules.training_helpers import standardize
import SimpleITK as sitk
import shutil

logging.basicConfig(level=logging.DEBUG)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

parser = optparse.OptionParser()
parser.add_option('--dm', '--defmodelfile',
                  dest="defmodelfile",
                  default='isensee',
                  type='str'
                  )

# we need grade to load the appropriate mean file
parser.add_option('--g', '--grade',
                  dest="grade",
                  default='HGG',
                  type='str'
                  )

# decouple this from global variable.
parser.add_option('--vo', '--validate-on',
                  dest="validate_on",
                  default='2017',
                  type='str'
                  )


parser.add_option('--mn', '--model-name',
                  dest="model_name",
                  default='/home/anmol/mounts/cedar-rm/scratch/asa224/model-checkpoints/3d_unet_old_checkpoints_uncompleted_job/3dunet_patches.h525--0.21.h5',
                  type='str'
                  )


options, remainder = parser.parse_args()

# ---------------------------------------------------------------------
# SET PARAMETERS HERE
# ---------------------------------------------------------------------
# set the model name to load here
options.defmodelfile = "isensee"
options.grade = "Combined"
options.model_name = "/home/anmol/mounts/cedar-rm/scratch/asa224/model-staging/BRATS_E160--0.78.h5"
options.validate_on = "2018"
# ---------------------------------------------------------------------

if options.defmodelfile is None:
    logger.info('No defmodel file name defined, using default model (cnn)')
    options.defmodelfile = 'cnn'

# --------------------------------------------------------------------------------------
# open required files
# --------------------------------------------------------------------------------------
# open mean and variance dictionary

if options.grade == 'HGG' or options.grade == 'LGG':
    mean_var = pickle.load(open(config['saveMeanVarFilepath' + options.grade.upper()], 'rb'))
else:
    mean_var = pickle.load(open(config['saveMeanVarCombinedData'], 'rb'))

# open new database with cropped images
# if you want to run on cropped images, then load the cropping coordinates as well
# then create a new empty array, set the voxels of that array according to the cropping coordinates
# to recreate the original image that the system accpts.
if options.validate_on == '2018':
    logger.info('Validating on 2018 Validation Set!')
    hdf5_file = h5py.File(config['hdf5_filepath_prefix'], mode='r')
    hdf5_file_g = hdf5_file['original_data']
    val_shape = config['val_shape_after_prediction']
else:
    logger.info('Validating on 2017 Validation Set!')
    hdf5_file = h5py.File(config['hdf5_filepath_prefix_2017'], mode='r')
    hdf5_file_g = hdf5_file['original_data']
    val_shape = config['val_shape_after_prediction']


# get the validation data
validation_data = hdf5_file_g['validation_data']
validation_data_pat_name = hdf5_file_g['validation_data_pat_name']

# ------------------------------------------------------------------------------------
# create new HDF5 file to hold prediction data
# ------------------------------------------------------------------------------------
logger.info('Creating new HDF5 dataset to hold prediction data')
pred_filename = os.path.join(config['model_prediction_location'], 'model_predictions_' + options.model_name.split('/')[-1])
pred_nii_folder = '.'.join(pred_filename.split('.')[0:-1])
if os.path.isdir(pred_nii_folder):
    x = raw_input('Prediction directory already exists, do you want to overwrite? (y,n)')
    if x.lower() == 'y':
        shutil.rmtree(pred_nii_folder)
        os.makedirs(pred_nii_folder)

new_hdf5 = h5py.File(pred_filename, mode='w')

new_hdf5.create_dataset("validation_data", val_shape, np.float32)
new_hdf5.create_dataset("validation_data_pat_name", (val_shape[0],), dtype="S100")

# get the patient names as they are from the original file
new_hdf5['validation_data_pat_name'][:] = hdf5_file_g['validation_data_pat_name'][:]
# ====================================================================================

# -------------------------------------------------------------------------------------
# get model file and load model
# -------------------------------------------------------------------------------------
# open the model file module

modeldefmodule = importlib.import_module('defmodel.' + options.defmodelfile, package=None)
custom_objs = modeldefmodule.custom_loss()
model = modeldefmodule.open_model_with_hyper_and_history(name=options.model_name, custom_obj=custom_objs, load_model_only=True)
# =====================================================================================

# sample segmentation mask that comes with BRATS to  copy meta information from
sample_img = sitk.ReadImage('/home/anmol/mounts/cedar-rm/scratch/asa224/Datasets/BRATS2018/Training/HGG/Brats18_2013_2_1/Brats18_2013_2_1_seg.nii.gz')
# -------------------------------------------------------------------------------------
# Open the data, standardize and prepare for prediction
# -------------------------------------------------------------------------------------
logger.info(validation_data.shape)
logger.info('Looping over validation data for prediction')
for i in range(0, validation_data.shape[0]):
    logger.info('Indexing HDF5 datastore...')
    logger.info('Patient {}'.format(i))
    pat_volume = validation_data[i]
    pat_name = validation_data_pat_name[i]

    logger.info('Standardizing..')
    pat_volume = standardize(pat_volume, applyToTest=mean_var)

    curr_shape = list(pat_volume.shape)
    curr_shape.insert(0, 1) # insert 1 at index 0 to make reshaping easy

    pat_volume = pat_volume.reshape(curr_shape)

    # SUPER HACK WAY TO CHANGE VOLUME COMPATIBILITY WITH ISENSEE MODEL. MAKE 155 = 160
    new_pat_volume = np.zeros((1, 4, 240, 240, 160))
    new_pat_volume[:, :, :, :, 0:155] = pat_volume

    logger.info('Starting prediction..')
    # predict using the whole volume
    pred = model.predict(new_pat_volume)

    # sample for testing
    # pred = np.ones((1, 3, 240, 240, 160))
    # get back the main volume and strip the padding
    pred = pred[:,:,:,:,0:155]

    assert pred.shape == (1,3,240,240,155)

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

    edema_mask = np.clip(pred_sw[0,0] - pred_sw[0,2], 0, 1) # WT - TC, 0 - 2
    enhancing_mask = np.clip(pred_sw[0,1], 0, 1)
    nec_nh_mask = np.clip(pred_sw[0,2] - pred_sw[0,1], 0, 1) # TC - EN

    main_mask[np.where(edema_mask == 1)] = 2
    main_mask[np.where(enhancing_mask == 1)] = 4
    main_mask[np.where(nec_nh_mask == 1)] = 1

    # assert np.max(np.unique(main_mask)) <= 4, 'Segmentation labels may be wrong!'

    sitk_pred_img = sitk.GetImageFromArray(main_mask)
    sitk_pred_img.CopyInformation(sample_img)

    logger.info('Adding predicted volume to HDF5 store..')
    # we use the batch size = 1 for prediction, so the first one.
    new_hdf5['validation_data'][i] = pred[0]

    logger.info('Saving prediction as nii.gz file')
    sitk.WriteImage(sitk_pred_img, os.path.join(pred_nii_folder, '{}.nii.gz'.format(pat_name)))

# =====================================================================================

new_hdf5.close()

