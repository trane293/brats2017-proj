"""
==========================================================
                        Train Models
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
DESCRIPTION: This file serves the purpose of loading one of
             pre-defined networks from the defmodel folder
             and training it. The script provides modularity
             in the sense that it can be used to load any
             of the defined models in the defmodel folder.

             Check the defmodel  folder on how to define one
             such model file, complete wtth someof the mandatory
             functions that need to  be defined in it as well.
LICENCE: Proprietary for now.
"""
import platform
import logging

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# to make the code portable even on cedar,you need to add conditions here
node_name = platform.node()
if node_name == 'XPS15' or 'cs-mial-31' in node_name:
    # this is my laptop, so the cedar-rm directory is at a different place
    # disable GPU
    logger.info('Disabling GPU!')
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # -1 !!!!

import h5py
import sys
from modules.configfile import config
from modules.training_helpers import *
import cPickle as pickle
import numpy as np
import random as random
random.seed(1337)
np.random.seed(1337)
import importlib
import optparse
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

parser = optparse.OptionParser()
parser.add_option('--dm', '--defmodelfile',
                  dest="defmodelfile",
                  default='isensee',
                  type='str',
                  help='Define the name of module from defmodel directory to load the model definition from'
                  )

parser.add_option('--o', '--out-name',
                  dest="output_name",
                  default='isensee_main.h5',
                  type='str',
                  help='Define the name of the file that was generated after training was finished. Check model-snapshots'
                  )

parser.add_option('--e', '--epochs',
                  dest="epochs",
                  default=10,
                  type='int',
                  help='Number of epochs to train with'
                  )

parser.add_option('--b', '--batch-size',
                  dest="batch_size",
                  default=2,
                  type='int',
                  help='Batch size to train with'
                  )

parser.add_option('--h', '--hdf5',
                  dest="hdf5_filepath",
                  default=None,
                  type='str',
                  help='HDF5 filepath in case loading from a non-standard location'
                  )

parser.add_option('--rs', '--remove-seq',
                  dest="remove_seq",
                  action="store_true",
                  default=False,
                  help='Enable mean imputation based data augmentation'
                  )

parser.add_option('--n', '--add-noise',
                  dest="add_noise",
                  action="store_true",
                  default=False,
                  help='Enable noise addition based data augmentation'
                  )


options, remainder = parser.parse_args()

# --------------------------------------------------------------------------------------
# open required files
# --------------------------------------------------------------------------------------
# open mean and variance dictionary

mean_var = pickle.load(open(config['saveMeanVarCombinedData'], 'rb'))

# open new database with cropped images

hdf5_file = h5py.File(config['hdf5_combined'], mode='r')
hdf5_file_g = hdf5_file['combined']

# get all the HGG/LGG data
training_data = hdf5_file_g['training_data']
training_data_segmasks = hdf5_file_g['training_data_segmasks']

# ======================================================================================

# --------------------------------------------------------------------------------------
# split data into training and testing
# --------------------------------------------------------------------------------------
indices = list(range(0, training_data.shape[0]))
random.shuffle(indices) # in-place shuffling

train_end = int((len(indices) * config['data_split']['train']) / 100.0)
train_indices = indices[0:train_end]
test_indices = indices[train_end:]

# ======================================================================================

if options.output_name is None:
    logger.info('No output name defined, using default values')
    options.output_name = os.path.join(config['model_snapshot_location'], 'model_default.h5')
    logger.info('Name of output file: {}'.format(options.output_name))
else:
    options.output_name = os.path.join(config['model_snapshot_location'], options.output_name)
    logger.info('Name of output file: {}'.format(options.output_name))

if options.defmodelfile is None:
    logger.info('No defmodel file name defined, using default model (cnn)')
    options.defmodelfile = 'cnn'

# -------------------------------------------------------------------------------------
# get model file
# -------------------------------------------------------------------------------------
# open the model file module
modeldefmodule = importlib.import_module('defmodel.'+options.defmodelfile, package=None)

# get the model
# inp_shape = tuple(config['patch_input_shape'])
inp_shape = (4, None, None, None)
model = modeldefmodule.get_model(inp_shape=inp_shape) # (4, x, y, z)

# # compile
# model = modeldefmodule.compile_model(model)
# ======================================================================================

# --------------------------------------------------------------------------------------
# intialize some callbacks
# --------------------------------------------------------------------------------------
mc = ModelCheckpoint(os.path.join(config['model_checkpoint_location'],
                                  options.output_name.split('/')[-1]+'{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', verbose=1,
                                  save_best_only=False, save_weights_only=False, mode='auto', period=1)

reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.001)

es = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=6, verbose=1, mode='auto')
exp_name = options.output_name.split('/')[-1]
os.system('mkdir ./graph/{}'.format(exp_name))
tb = TensorBoard(log_dir='./graph/{}'.format(exp_name), histogram_freq=0,
          write_graph=True, write_images=True)
# ======================================================================================

c = 0
epochs = options.epochs
batch_size = options.batch_size
total_per_epoch_training = (len(train_indices) * config['num_patches_per_patient'] / batch_size)
total_per_epoch_testing = (len(test_indices) * config['num_patches_per_patient'] / batch_size)

augment = ['permute']
if options.remove_seq == True:
    logger.info('Running training with remove_sequence=True')
    augment.append('remove_seq')
if options.add_noise == True:
    logger.info('Running training with add_noise=True')
    augment.append('add_noise')

train_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                   t_i=train_indices, mean_var=mean_var, batch_size=batch_size, gen_name='Training',
                                   applyNorm=True, augment=augment)

test_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                  t_i=test_indices, mean_var=mean_var, batch_size=batch_size, gen_name='Testing',
                                  applyNorm=True, augment=augment)

history = model.fit_generator(train_gen, steps_per_epoch=total_per_epoch_training,
                    epochs=epochs, verbose=1, callbacks=[mc, reduceLR, tb],
                    validation_data=test_gen, validation_steps=total_per_epoch_testing,
                    class_weight=None, max_queue_size=100, workers=1,
                    use_multiprocessing=True, shuffle=False, initial_epoch=0)

modeldefmodule.save_model_with_hyper_and_history(model, history=history, name=options.output_name)



