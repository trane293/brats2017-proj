"""
==========================================================
        Generate Database with 2D Bounding Boxes
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
             2D bounding box based labels for each patient.

             The default annotations that are provided are
             segmentation masks. This script uses those masks
             to find the minimum volume bounding box (in 2D)
             and stores the coordinates in a new hdf5 file.
LICENCE: Proprietary for now.
"""

import h5py
from modules.configfile import config
import numpy as np
import logging
from modules.mischelpers import *

logging.basicConfig(level=logging.DEBUG)

try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

logger.info('Opening previously generated HDF5 file.')

# open the file in append mode, so that we can append the new database
hdf5_file = h5py.File(config['hdf5_filepath_prefix'], 'a')

if u"training_hgg_patients_all_slices" in hdf5_file.keys():
    logger.warn('{} Database already exists!'.format('training_hgg_patients_all_slices'))
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["training_hgg_patients_all_slices"]

if u"bounding_box_hgg_all_slices" in hdf5_file.keys():
    logger.warn('{} database already exists!'.format('bounding_box_hgg_all_slices'))
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["bounding_box_hgg_all_slices"]

if u"training_lgg_patients_all_slices" in hdf5_file.keys():
    logger.warn('{} database already exists!'.format('training_lgg_patients_all_slices'))
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["training_lgg_patients_all_slices"]

if u"bounding_box_lgg_all_slices" in hdf5_file.keys():
    logger.warn('{} database already exists!'.format('bounding_box_lgg_all_slices'))
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["bounding_box_lgg_all_slices"]

if u"pat_names_hgg_all_slices" in hdf5_file.keys():
    logger.warn('{} database already exists!'.format('pat_names_hgg_all_slices'))
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["pat_names_hgg_all_slices"]

if u"pat_names_lgg_all_slices" in hdf5_file.keys():
    logger.warn('{} database already exists!'.format('pat_names_lgg_all_slices'))
    logger.warn('We will proceed by deleting the old database and creating the new one')
    del hdf5_file["pat_names_lgg_all_slices"]

'''
Create the dataset

training_hgg_patients_all_slices
bounding_box_hgg_all_slices

training_lgg_patients_all_slices
bounding_box_lgg_all_slices

pat_names_hgg_all_slices
pat_names_lgg_all_slices

'''

num_slices_hgg = config['train_hgg_patients'] * config['num_slices']
num_slices_lgg = config['train_lgg_patients'] * config['num_slices']
shape_bounding_box_hgg_all_slices = (num_slices_hgg, 4)
shape_pat_names_hgg_all_slices = (num_slices_hgg,)
shape_bounding_box_lgg_all_slices = (num_slices_lgg, 4)
shape_pat_names_lgg_all_slices = (num_slices_lgg,)

if config['data_order'] == 'th':
    shape_training_hgg_patients_all_slices = (config['train_hgg_patients']*config['num_slices'],
                                              4,
                                              config['spatial_size_for_training'][0],
                                              config['spatial_size_for_training'][1])



    shape_training_lgg_patients_all_slices = (config['train_lgg_patients'] * config['num_slices'],
                                              4,
                                              config['spatial_size_for_training'][0],
                                              config['spatial_size_for_training'][1])

else:
    shape_training_hgg_patients_all_slices = (num_slices_hgg,
                                              config['spatial_size_for_training'][0],
                                              config['spatial_size_for_training'][1],
                                              4)

    shape_training_lgg_patients_all_slices = (num_slices_lgg,
                                              config['spatial_size_for_training'][0],
                                              config['spatial_size_for_training'][1],
                                              4)

hdf5_file.create_dataset("training_hgg_patients_all_slices", shape_training_hgg_patients_all_slices, np.int16)
hdf5_file.create_dataset("training_lgg_patients_all_slices", shape_training_lgg_patients_all_slices, np.int16)

hdf5_file.create_dataset("bounding_box_hgg_all_slices", shape_bounding_box_hgg_all_slices, np.int16)
hdf5_file.create_dataset("bounding_box_lgg_all_slices", shape_bounding_box_lgg_all_slices, np.int16)

hdf5_file.create_dataset("pat_names_hgg_all_slices", shape_pat_names_hgg_all_slices, dtype="S100")
hdf5_file.create_dataset("pat_names_lgg_all_slices", shape_pat_names_lgg_all_slices, dtype="S100")

logger.info('Running on LGG Data.')
main_count_lgg = 0
for pat in range(0, config['train_lgg_patients']): # 210 patients in HGG
    # we have the first patient
    patient_data = hdf5_file['training_data_lgg'][pat, ...]
    segmask = hdf5_file['training_data_segmasks_lgg'][pat,...]
    patname = hdf5_file['training_data_lgg_pat_name'][pat]
    logger.debug('Patient {}'.format(pat))
    for slice in range(0, config['num_slices']):
        # for this patient, save each slice in the training data.
        hdf5_file['training_lgg_patients_all_slices'][main_count_lgg,...] = patient_data[:,:,:,slice]

        # for this slice, get the 2D bounding box, and save it in the dataset
        rect_obj = bbox_2D(segmask[:,:, slice], tol=0.1)
        hdf5_file['bounding_box_lgg_all_slices'][main_count_lgg] = rect_obj.list_view

        # for this slice, save the name of patient so that we can backtrack
        hdf5_file['pat_names_lgg_all_slices'][main_count_lgg] = patname
        main_count_lgg += 1

logger.info('Running on HGG Data.')
main_count_hgg = 0
for pat in range(0, config['train_hgg_patients']): # 210 patients in HGG
    # we have the first patient
    patient_data = hdf5_file['training_data_hgg'][pat, ...]
    segmask = hdf5_file['training_data_segmasks_hgg'][pat,...]
    patname = hdf5_file['training_data_hgg_pat_name'][pat]
    logger.debug('Patient {}'.format(pat))
    for slice in range(0, config['num_slices']):
        # for this patient, save each slice in the training data.
        hdf5_file['training_hgg_patients_all_slices'][main_count_hgg,...] = patient_data[:,:,:,slice]

        # for this slice, get the 2D bounding box, and save it in the dataset
        rect_obj = bbox_2D(segmask[:,:, slice], tol=0.1)
        hdf5_file['bounding_box_hgg_all_slices'][main_count_hgg] = rect_obj.list_view

        # for this slice, save the name of patient so that we can backtrack
        hdf5_file['pat_names_hgg_all_slices'][main_count_hgg] = patname
        main_count_hgg += 1

logger.info('Closing HDF5 File.')
hdf5_file.close()