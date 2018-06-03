"""
==========================================================
    Combine HGG and LGG BRATS Data for easy Training
==========================================================
AUTHOR: Anmol Sharma
AFFILIATION: Simon Fraser University
             Burnaby, BC, Canada
PROJECT: Analysis of Brain MRI Scans for Management of
         Malignant Tumors
COLLABORATORS: Anmol Sharma (SFU)
               Prof. Ghassan Hamarneh (SFU)
               Dr. Brian Toyota (VGH)
               Dr. Mostafa Fatehi (VGH)
DESCRIPTION: This file uses the generated HDF5 data store to
             create a new HDF5 file which contains both HGG
             and LGG data in one single dataset.

             The created HDF5 file also ensures the patient
             names are saved.

LICENCE: Proprietary for now.
"""

import logging
from modules.training_helpers import standardize
from modules.mischelpers import *
import os

logging.basicConfig(level=logging.DEBUG)

try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------
# open existing datafile
# ------------------------------------------------------------------------------------
logger.info('opening previously generated HDF5 file.')

# open the existing datafile
hdf5_file_main = h5py.File(config['hdf5_filepath_prefix'], 'r')

logger.info('opened HDF5 file at {}'.format(config['hdf5_filepath_prefix']))

# get the group identifier for original dataset
hdf5_file = hdf5_file_main['original_data']

# ====================================================================================

# ------------------------------------------------------------------------------------
# create new HDF5 file to hold cropped data.
# ------------------------------------------------------------------------------------
logger.info('creating new HDF5 dataset to hold combined HGG and LGG data')
filename = os.path.join(os.sep.join(config['hdf5_filepath_prefix'].split(os.sep)[0:-1]), 'BRATS_Combined.h5')
new_hdf5 = h5py.File(filename, mode='w')
logger.info('created new database at {}'.format(filename))

# create a folder group to  hold the datasets. The schema is similar to original one except for the name of the folder
# group
g_combined = new_hdf5.create_group('combined')

# get total number of patients
total_pats = config['train_hgg_patients'] + config['train_lgg_patients']
tmp = list(config['train_shape_hgg'])
tmp[0] = total_pats
total_shape_img = tmp

tmp = list(config['train_segmasks_shape_hgg'])
tmp[0] = total_pats
total_shape_seg = tmp

# create similar datasets in this file.
g_combined.create_dataset("training_data", total_shape_img, np.float32)
g_combined.create_dataset("training_data_segmasks", total_shape_seg, np.int16)
g_combined.create_dataset("training_data_pat_name", (total_pats,), dtype="S100")
logger.info('Created datasets!')
# ====================================================================================

# just copy the patient  names directly
# from 0 - HGG_pats we have hgg pat names and from HGG_pats:end we have LGG pat names
logger.info('Copying patient names..')
g_combined['training_data_pat_name'][0:config['train_hgg_patients']] = hdf5_file['training_data_hgg_pat_name'][:]
g_combined['training_data_pat_name'][config['train_hgg_patients']:] = hdf5_file['training_data_lgg_pat_name'][:]
logger.info('Copy patient names successful!')
# ------------------------------------------------------------------------------------
# Copy HGG and LGG data from original datastore
# ------------------------------------------------------------------------------------
logger.info('Copying HGG patient data')
g_combined['training_data'][0:config['train_hgg_patients']] = hdf5_file['training_data_hgg'][:]
g_combined['training_data_segmasks'][0:config['train_hgg_patients']] = hdf5_file['training_data_segmasks_hgg'][:]

logger.info('Copying LGG patient data')
g_combined['training_data'][config['train_hgg_patients']:] = hdf5_file['training_data_lgg'][:]
g_combined['training_data_segmasks'][config['train_hgg_patients']:] = hdf5_file['training_data_segmasks_lgg'][:]
logger.info('Data copy done!')

data = g_combined['training_data']
logger.info('Calculating mean/var values from this data')
_tmp, vals = standardize(data, findMeanVarOnly=True, saveDump=config['saveMeanVarCombinedData'])
logger.info('Calculating mean/var values from this data')

hdf5_file_main.close()
new_hdf5.close()
logger.info('HDF5 Files closed!')