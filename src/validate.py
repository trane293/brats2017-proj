import h5py
from modules.configfile import config, mount_path_prefix
from modules.training_helpers import *
import cPickle as pickle
import numpy as np
import random as random
random.seed(1337)
np.random.seed(1337)
import importlib
import optparse
import os
import logging
from modules.dataloader import standardize

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

parser.add_option('--o', '--out-name',
                  dest="output_name",
                  default='isensee_main',
                  type='str'
                  )

# decouple this from global variable.
parser.add_option('--vo', '--validate-on',
                  dest="validate_on",
                  default='original',
                  type='str'
                  )


parser.add_option('--mn', '--model-name',
                  dest="model_name",
                  default='/home/anmol/mounts/cedar-rm/scratch/asa224/model-checkpoints/3d_unet_old_checkpoints_uncompleted_job/3dunet_patches.h525--0.21.h5',
                  type='str'
                  )


options, remainder = parser.parse_args()

# CHANGE THE MODEL PATH HERE
options.model_name = mount_path_prefix + 'cedar-rm/scratch/asa224/model-staging/isensee_main.h5'

if options.output_name is None:
    logger.info('No output name defined, using default values')
    pred_filename = os.path.join(config['model_prediction_location'],
                                 'BRATS_Validation_Prediction_' + 'model_default.h5')
    options.output_name = os.path.join(config['model_snapshot_location'], 'model_default.h5')
    logger.info('Name of output file: {}'.format(options.output_name))
else:
    pred_filename = os.path.join(config['model_prediction_location'],
                                 'BRATS_Validation_Prediction_' + options.output_name)
    options.output_name = os.path.join(config['model_snapshot_location'], options.output_name)
    logger.info('Name of input model file: {}'.format(options.output_name))

if options.defmodelfile is None:
    logger.info('No defmodel file name defined, using default model (cnn)')
    options.defmodelfile = 'cnn'

# --------------------------------------------------------------------------------------
# open required files
# --------------------------------------------------------------------------------------
# open mean and variance dictionary

mean_var = pickle.load(open(config['saveMeanVarFilepath' + options.grade.upper()], 'rb'))

# open new database with cropped images
# if you want to run on cropped images, then load the cropping coordinates as well
# then create a new empty array, set the voxels of that array according to the cropping coordinates
# to recreate the original image that the system accpts.
if options.validate_on == 'cropped':
    hdf5_file = h5py.File(config['hdf5_filepath_cropped'], mode='r')
    hdf5_file_g = hdf5_file['preprocessed']
    val_shape = config['val_shape_crop']
else:
    hdf5_file = h5py.File(config['hdf5_filepath_prefix'], mode='r')
    hdf5_file_g = hdf5_file['original_data']
    val_shape = config['val_shape_after_prediction']

# get the validation data
validation_data = hdf5_file_g['validation_data']


# ------------------------------------------------------------------------------------
# create new HDF5 file to hold prediction data
# ------------------------------------------------------------------------------------
logger.info('Creating new HDF5 dataset to hold prediction data')

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

# -------------------------------------------------------------------------------------
# Open the data, standardize and prepare for prediction
# -------------------------------------------------------------------------------------
for i in range(0, validation_data.shape[0]):
    pat_volume = validation_data[i]
    pat_volume = standardize(pat_volume, applyToTest=mean_var)

    curr_shape = list(pat_volume.shape)
    curr_shape.insert(0, 1) # insert 1 at index 0 to make reshaping easy

    pat_volume = pat_volume.reshape(curr_shape)

    # SUPER HACK WAY TO CHANGE VOLUME COMPATIBILITY WITH ISENSEE MODEL. MAKE 155 = 160
    new_pat_volume = np.zeros((1, 4, 240, 240, 160))
    new_pat_volume[:, :, :, :, 0:155] = pat_volume

    # predict using the whole volume
    pred = model.predict(new_pat_volume)

    # get back the main volume and strip the padding
    pred = pred[:,:,:,:,0:155]

    assert pred.shape == (1,3,240,240,155)

    # we use the batch size = 1 for prediction, so the first one.
    new_hdf5['validation_data'][i] = pred[0]
# =====================================================================================

new_hdf5.close()

