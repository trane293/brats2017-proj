"""
==========================================================
            Generate Database with Bounding Boxes
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
DESCRIPTION: This file uses the previously generated data
             (using create_hdf5_file.py) and generates
             bounding box based labels for each patient.

             The default annotations that are provided are
             segmentation masks. This script uses those masks
             to find the minimum volume bounding box and stores
             the coordinates in a new hdf5 file.
LICENCE: Proprietary for now.
"""

import h5py
from modules.configfile import config
import numpy as np
import logging
from modules.mischelpers import *

logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

logger.info('Opening previously generated HDF5 file.')

# open the file in append mode, so that we can append the new database
hdf5_file = h5py.File(config['hdf5_filepath_prefix'], 'a')

if u"bounding_box_hgg" in hdf5_file.keys():
    logger.warn('HGG Bounding box database already exists!')
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["bounding_box_hgg"]

if u"bounding_box_lgg" in hdf5_file.keys():
    logger.warn('LGG Bounding box database already exists!')
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["bounding_box_lgg"]

hdf5_file.create_dataset("bounding_box_hgg", (config['train_hgg_patients'], 6), np.int16)
hdf5_file.create_dataset("bounding_box_lgg", (config['train_lgg_patients'], 6), np.int16)

logger.info('Running on HGG Data.')
for pat in range(0, config['train_hgg_patients']): # 210 patients in HGG
    rect_obj = bbox_3D(hdf5_file['training_data_segmasks_hgg'][pat, ...], tol=0.1)
    hdf5_file['bounding_box_hgg'][pat] = rect_obj.list_view

logger.info('Running on LGG Data.')
for pat in range(0, config['train_lgg_patients']): # 75 patients in HGG
    rect_obj = bbox_3D(hdf5_file['training_data_segmasks_lgg'][pat, ...], tol=0.1)
    hdf5_file['bounding_box_lgg'][pat] = rect_obj.list_view

logger.info('Closing HDF5 File.')
hdf5_file.close()