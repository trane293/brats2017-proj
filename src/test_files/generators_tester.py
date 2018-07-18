import sys
sys.path.append('..')
from modules.training_helpers import *
from modules.configfile import config
import logging
import random as random
import optparse
import cPickle as pickle
import h5py
import numpy as np
random.seed(config['seed'])
np.random.seed(config['seed'])

logging.basicConfig(level=logging.DEBUG)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)


parser = optparse.OptionParser()
parser.add_option('--dm', '--defmodelfile',
                  dest="defmodelfile",
                  default='cnn_patches',
                  type='str'
                  )

parser.add_option('--g', '--grade',
                  dest="grade",
                  default='HGG',
                  type='str'
                  )

parser.add_option('--o', '--out-name',
                  dest="output_name",
                  default='cnn_patches_v1.h5',
                  type='str'
                  )

options, remainder = parser.parse_args()

# --------------------------------------------------------------------------------------
# open required files
# --------------------------------------------------------------------------------------
# open mean and variance dictionary

mean_var = pickle.load(open(config['saveMeanVarFilepath'+options.grade.upper()], 'rb'))

# open new database with cropped images

if config['gen_patches_from'] == 'cropped':
    hdf5_file = h5py.File(config['hdf5_filepath_cropped'], mode='r')
    hdf5_file_g = hdf5_file['preprocessed']
else:
    hdf5_file = h5py.File(config['hdf5_filepath_prefix'], mode='r')
    hdf5_file_g = hdf5_file['original_data']

# get all the HGG/LGG data
training_data = hdf5_file_g['training_data_'+options.grade.lower()]
training_data_segmasks = hdf5_file_g['training_data_segmasks_'+options.grade.lower()]

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
batch_size = 10
augment = ['permute', 'remove_seq']
train_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                   t_i=train_indices, mean_var=mean_var, batch_size=batch_size, debug_mode=True,
                                   applyNorm=False, augment=augment)

test_gen = generate_patch_batches(X=training_data, Y=training_data_segmasks,
                                  t_i=test_indices, mean_var=mean_var, batch_size=batch_size, debug_mode=True,
                                  applyNorm=False, augment=augment)

count = 0
for x_patches, y_patches in test_gen:
    print(x_patches.shape, y_patches.shape)
    if count > 20:
        break

count = 0
for x_patches, y_patches in train_gen:
    print(x_patches.shape, y_patches.shape)
    if count > 20:
        break
