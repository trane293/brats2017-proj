from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
import sys

sys.path.append('..')
from modules.configfile import config
import logging

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

hyparams = {}

if config['data_order'] == 'th':
    logger.info('Setting keras backend data format to "channels_first"')
    keras.backend.set_image_data_format('channels_first')
    hyparams['input_shape'] = (4, config['spatial_size_for_training'][0], config['spatial_size_for_training'][0], config['num_slices'])
else:
    logger.info('Setting keras backend data format to "channels_last"')
    keras.backend.set_image_data_format('channels_last')
    hyparams['input_shape'] = (config['spatial_size_for_training'][0], config['spatial_size_for_training'][0], config['num_slices'], 4)

hyparams['output_units'] = 6

hyparams['training_batch_size'] = 1

# ============================================================================
# BLOCK: 1
# ============================================================================
bname = 'B1'
hyparams[bname] = {}
hyparams[bname]['c1_num_filters'] = 32
hyparams[bname]['c1_filter_size'] = (3, 3, 3)
hyparams[bname]['c1_stride'] = (1, 1, 1)
hyparams[bname]['c1_padding'] = 'valid'
hyparams[bname]['c1_activation'] = 'relu'

hyparams[bname]['c2_num_filters'] = 32
hyparams[bname]['c2_filter_size'] = (3, 3, 3)
hyparams[bname]['c2_stride'] = (1, 1, 1)
hyparams[bname]['c2_padding'] = 'valid'
hyparams[bname]['c2_activation'] = 'relu'

hyparams[bname]['p1_pool_size'] = (2, 2, 2)
hyparams[bname]['p1_stride'] = None
# ----------------------------------------------------------------------------


# ============================================================================
# BLOCK: 2
# ============================================================================
bname = 'B2'
hyparams[bname] = {}
hyparams[bname]['c1_num_filters'] = 64
hyparams[bname]['c1_filter_size'] = (3, 3, 3)
hyparams[bname]['c1_stride'] = (1, 1, 1)
hyparams[bname]['c1_padding'] = 'valid'
hyparams[bname]['c1_activation'] = 'relu'

hyparams[bname]['c2_num_filters'] = 64
hyparams[bname]['c2_filter_size'] = (3, 3, 3)
hyparams[bname]['c2_stride'] = (1, 1, 1)
hyparams[bname]['c2_padding'] = 'valid'
hyparams[bname]['c2_activation'] = 'relu'

hyparams[bname]['p1_pool_size'] = (2, 2, 2)
hyparams[bname]['p1_stride'] = None
# ----------------------------------------------------------------------------


# ============================================================================
# BLOCK: 3
# ============================================================================
bname = 'B3'
hyparams[bname] = {}
hyparams[bname]['c1_num_filters'] = 128
hyparams[bname]['c1_filter_size'] = (3, 3, 3)
hyparams[bname]['c1_stride'] = (1, 1, 1)
hyparams[bname]['c1_padding'] = 'valid'
hyparams[bname]['c1_activation'] = 'relu'

hyparams[bname]['c2_num_filters'] = 128
hyparams[bname]['c2_filter_size'] = (3, 3, 3)
hyparams[bname]['c2_stride'] = (1, 1, 1)
hyparams[bname]['c2_padding'] = 'valid'
hyparams[bname]['c2_activation'] = 'relu'

hyparams[bname]['p1_pool_size'] = (2, 2, 2)
hyparams[bname]['p1_stride'] = None
# ----------------------------------------------------------------------------


# ============================================================================
# BLOCK: 4
# ============================================================================
bname = 'B4'
hyparams[bname] = {}
hyparams[bname]['c1_num_filters'] = 128
hyparams[bname]['c1_filter_size'] = (5, 5, 5)
hyparams[bname]['c1_stride'] = (1, 1, 1)
hyparams[bname]['c1_padding'] = 'valid'
hyparams[bname]['c1_activation'] = 'relu'

hyparams[bname]['c2_num_filters'] = 128
hyparams[bname]['c2_filter_size'] = (5, 5, 5)
hyparams[bname]['c2_stride'] = (1, 1, 1)
hyparams[bname]['c2_padding'] = 'valid'
hyparams[bname]['c2_activation'] = 'relu'

hyparams[bname]['p1_pool_size'] = (2, 2, 2)
hyparams[bname]['p1_stride'] = None
# ----------------------------------------------------------------------------

# ============================================================================
# BLOCK: 5
# ============================================================================
bname = 'B5'
hyparams[bname] = {}
hyparams[bname]['c1_num_filters'] = 256
hyparams[bname]['c1_filter_size'] = (5, 5, 5)
hyparams[bname]['c1_stride'] = (1, 1, 1)
hyparams[bname]['c1_padding'] = 'same'
hyparams[bname]['c1_activation'] = 'relu'

hyparams[bname]['c2_num_filters'] = 256
hyparams[bname]['c2_filter_size'] = (5, 5, 5)
hyparams[bname]['c2_stride'] = (1, 1, 1)
hyparams[bname]['c2_padding'] = 'same'
hyparams[bname]['c2_activation'] = 'relu'

hyparams[bname]['p1_pool_size'] = (2, 2, 2)
hyparams[bname]['p1_stride'] = None
# ----------------------------------------------------------------------------

# ============================================================================
# BLOCK: 5
# ============================================================================
bname = 'B6'
hyparams[bname] = {}
hyparams[bname]['d1_num_units'] = 600
hyparams[bname]['d1_activation'] = 'relu'

hyparams[bname]['d2_num_units'] = 600
hyparams[bname]['d2_activation'] = 'relu'

hyparams[bname]['out_num_units'] = 6
hyparams[bname]['out_activation'] = None


# ----------------------------------------------------------------------------

def get_model(hyparams=hyparams):
    # INPUT LAYER
    main_input = Input(shape=hyparams['input_shape'], name='input')

    # ============================================================================
    # BLOCK: 1
    # ============================================================================
    bname = 'B1'

    x = Conv3D(hyparams[bname]['c1_num_filters'], hyparams[bname]['c1_filter_size'],
               strides=hyparams[bname]['c1_stride'],
               padding=hyparams[bname]['c1_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c1_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(main_input)

    x = Conv3D(hyparams[bname]['c2_num_filters'], hyparams[bname]['c2_filter_size'],
               strides=hyparams[bname]['c2_stride'],
               padding=hyparams[bname]['c2_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c2_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = MaxPooling3D(pool_size=hyparams[bname]['p1_pool_size'], strides=hyparams[bname]['p1_stride'],
                     padding='valid',
                     data_format=None)(x)
    # ----------------------------------------------------------------------------

    # ============================================================================
    # BLOCK: 2
    # ============================================================================
    bname = 'B2'

    x = Conv3D(hyparams[bname]['c1_num_filters'], hyparams[bname]['c1_filter_size'],
               strides=hyparams[bname]['c1_stride'],
               padding=hyparams[bname]['c1_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c1_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = Conv3D(hyparams[bname]['c2_num_filters'], hyparams[bname]['c2_filter_size'],
               strides=hyparams[bname]['c2_stride'],
               padding=hyparams[bname]['c2_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c2_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = MaxPooling3D(pool_size=hyparams[bname]['p1_pool_size'], strides=hyparams[bname]['p1_stride'],
                     padding='valid',
                     data_format=None)(x)

    # ----------------------------------------------------------------------------

    # ============================================================================
    # BLOCK: 3
    # ============================================================================
    bname = 'B3'

    x = Conv3D(hyparams[bname]['c1_num_filters'], hyparams[bname]['c1_filter_size'],
               strides=hyparams[bname]['c1_stride'],
               padding=hyparams[bname]['c1_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c1_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = Conv3D(hyparams[bname]['c2_num_filters'], hyparams[bname]['c2_filter_size'],
               strides=hyparams[bname]['c2_stride'],
               padding=hyparams[bname]['c2_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c2_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = MaxPooling3D(pool_size=hyparams[bname]['p1_pool_size'], strides=hyparams[bname]['p1_stride'],
                     padding='valid',
                     data_format=None)(x)

    # ----------------------------------------------------------------------------

    # ============================================================================
    # BLOCK: 4
    # ===========================================================================
    bname = 'B4'

    x = Conv3D(hyparams[bname]['c1_num_filters'], hyparams[bname]['c1_filter_size'],
               strides=hyparams[bname]['c1_stride'],
               padding=hyparams[bname]['c1_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c1_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = Conv3D(hyparams[bname]['c2_num_filters'], hyparams[bname]['c2_filter_size'],
               strides=hyparams[bname]['c2_stride'],
               padding=hyparams[bname]['c2_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c2_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = MaxPooling3D(pool_size=hyparams[bname]['p1_pool_size'], strides=hyparams[bname]['p1_stride'],
                     padding='valid',
                     data_format=None)(x)

    # ----------------------------------------------------------------------------

    # ============================================================================
    # BLOCK: 4
    # ===========================================================================
    bname = 'B5'

    x = Conv3D(hyparams[bname]['c1_num_filters'], hyparams[bname]['c1_filter_size'],
               strides=hyparams[bname]['c1_stride'],
               padding=hyparams[bname]['c1_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c1_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = Conv3D(hyparams[bname]['c2_num_filters'], hyparams[bname]['c2_filter_size'],
               strides=hyparams[bname]['c2_stride'],
               padding=hyparams[bname]['c2_padding'], data_format=None,
               dilation_rate=(1, 1, 1), activation=hyparams[bname]['c2_activation'],
               use_bias=True, kernel_initializer='glorot_uniform',
               bias_initializer='zeros', kernel_regularizer=None,
               bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None)(x)

    x = MaxPooling3D(pool_size=hyparams[bname]['p1_pool_size'], strides=hyparams[bname]['p1_stride'],
                     padding='valid',
                     data_format=None)(x)

    # ----------------------------------------------------------------------------

    # ============================================================================
    # BLOCK: 5
    # ============================================================================
    bname = 'B6'
    x = Flatten()(x)
    x = Dense(hyparams[bname]['d1_num_units'], activation=hyparams[bname]['d1_activation'], use_bias=True,
              kernel_initializer='glorot_uniform', bias_initializer='zeros',
              kernel_regularizer=None, bias_regularizer=None,
              activity_regularizer=None, kernel_constraint=None,
              bias_constraint=None)(x)

    x = Dense(hyparams[bname]['d2_num_units'], activation=hyparams[bname]['d2_activation'], use_bias=True,
              kernel_initializer='glorot_uniform', bias_initializer='zeros',
              kernel_regularizer=None, bias_regularizer=None,
              activity_regularizer=None, kernel_constraint=None,
              bias_constraint=None)(x)

    output = Dense(hyparams[bname]['out_num_units'], activation=hyparams[bname]['out_activation'], use_bias=True,
                   kernel_initializer='glorot_uniform', bias_initializer='zeros',
                   kernel_regularizer=None, bias_regularizer=None,
                   activity_regularizer=None, kernel_constraint=None,
                   bias_constraint=None)(x)

    # ----------------------------------------------------------------------------

    model = Model(inputs=[main_input], outputs=[output])

    model.summary()

    return model


def compile_model(model):
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    return model


def start_training(x_train, y_train, x_val, y_val, model, callbacks=None):
    history = model.fit(x=x_train, y=y_train,
              batch_size=hyparams['training_batch_size'], epochs=50,
              verbose=1, callbacks=callbacks,
              validation_split=0.0, validation_data=[x_val, y_val],
              shuffle=True, class_weight=None,
              sample_weight=None, initial_epoch=0,
              steps_per_epoch=None, validation_steps=None)
    return model, history


def save_model_with_hyper_and_history(model, history, name=None, hyparams=hyparams):
    import cPickle as pickle
    filename = name if name != None else "model"

    if '.h5' not in filename:
        filename_dict = filename + '_hyper_dict.p'
        filename = filename + '.h5'
        filename_history = filename + '_history.p'
    else:
        filename_dict = filename.split('.')[-2] + '_hyper_dict.p'
        filename_history = filename.split('.')[-2] + '_history.p'

    logger.info('Saving trained model with name {}'.format(filename))
    model.save(filename)
    logger.info('Model save successful!')

    logger.info('Saving hyperparameter dictionary with name {}'.format(filename_dict))
    with open(filename_dict, "wb") as f:
        pickle.dump(hyparams, f)
    logger.info('Saved hyperparameter dictionary!')

    logger.info('Saving history object with name {}'.format(filename_dict))
    with open(filename_history, "wb") as f:
        pickle.dump(history.history, f)
    logger.info('Saved history object!')


def open_model_with_hyper_and_history(name=None):
    import cPickle as pickle
    from keras.models import load_model
    filename = name if name != None else "model"

    if '.h5' not in filename:
        filename_dict = filename + '_hyper_dict.p'
        filename = filename + '.h5'
        filename_history = filename + '_history.p'
    else:
        filename_dict = filename.split('.')[-2] + '_hyper_dict.p'
        filename_history = filename.split('.')[-2] + '_history.p'

    logger.info('Opening trained model with name {}'.format(filename))
    model = load_model(filename)
    logger.info('Model open successful!')

    logger.info('Opening hyperparameter dictionary with name {}'.format(filename_dict))
    hyperparams = pickle.load(open(filename_dict, "rb"))
    logger.info('Opened hyperparameter dictionary!')

    logger.info('Opening history object with name {}'.format(filename_dict))
    history = pickle.load(open(filename_history, "rb"))
    logger.info('Saved history object!')

    return model, hyperparams, history

