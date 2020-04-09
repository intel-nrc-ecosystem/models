#
# Copyright © 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

# Extension (vs. orig version) by
#     get_dataset_full(dataset, params)   # train, test, and(!) val data

from __future__ import absolute_import
from __future__ import print_function

import os
import time

import keras
from keras import backend, layers, models, regularizers
from keras.applications import MobileNet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import Constraint
from keras.layers import ReLU
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MaxConstraint(Constraint):
    def __init__(self,
                 max_value,
                 min_value=None):
        if max_value is None:
            self.max_value = self.min_value = None
        else:
            assert (max_value > 0)
            self.max_value = max_value
            if min_value is None:
                self.min_value = -max_value
            else:
                assert min_value <= 0
                self.min_value = min_value

    def __call__(self, w):
        if self.max_value is None:
            return w

        return K.clip(w,
                      self.min_value,
                      self.max_value)

class Parameters():
    """A parameter class providing suitable defaults for training a DNN."""

    def __init__(self, input_shape, batch_size=64, num_epochs=400,
                 num_classes=10, alpha=1.0, num_blocks=2,
                 max_num_training_samples=None, weight_regularizer=1e-4,
                 dropout=0, use_bias=False, pretrained_model_path=None,
                 max_value=None, activity_regularizer=None,
                 signed_input=False, working_dir=None,
                 train_val_ratio=0.2, train_val_rand=1234, device=None):
        """
        Initializes Parameter class.

        :param input_shape: Shape of input layer.
        :param batch_size: Batch size used during training.
        :param num_epochs: Number of epochs to train for.
        :param num_classes: Number of classes in data set.
        :param alpha: Inspired by MobileNet architecture. Controls number of
        feature channels per layer.
        :param num_blocks: Controls number of depthwise-separable and 1x1 \
        convolution blocks.
        :param max_num_training_samples: Number of training samples.
        :param weight_regularizer: Regularizer for weights.
        :param dropout: Dropout level.
        :param use_bias: Turn biases on or off.
        :param pretrained_model_path: Path to continue model file to continue \
        training from.
        :param max_value: Saturation level of ReLU activation function.
        :param activity_regularizer: Regularizer for activity.
        :param signed_input: Set to True if inputs are signed. If unsigned, \
        set to False.
        :param working_dir: Base working directory in which model checkpoints \
        are stored.
        """

        self.input_shape = input_shape
        self.alpha = alpha
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.max_num_training_samples = max_num_training_samples
        self.weight_regularizer = weight_regularizer
        self.activity_regularizer = activity_regularizer
        self.dropout = dropout
        self.use_bias = use_bias
        self.pretrained_model_path = pretrained_model_path
        self.max_value = max_value
        self.signed_input = signed_input
        self.working_dir = working_dir
        self.train_val_ratio = train_val_ratio
        self.train_val_rand = train_val_rand
        self.device = device

    def to_string(self):
        return "inputShape{}_alpha{}_numBlocks{}_batchSize{}_numEpochs{}_" \
               "maxNumTrainingSamples{}_weightReg{}_dropout{}_useBias{}" \
               "_maxValue{}_activityReg{}_signedInput{}".format(
            "-".join([str(i) for i in self.input_shape]),
            self.alpha, self.num_blocks, self.batch_size,
            self.num_epochs, self.max_num_training_samples, self.weight_regularizer,
            self.dropout, self.use_bias, self.max_value,
            self.activity_regularizer, self.signed_input,
            self.working_dir,
            self.train_val_ratio, self.train_val_rand
        )

def gen_model_checkpoint_dir(working_dir, run_ctr, run_name=None):
    """Creates model_checkpoints directory within working directory.
    Raises error if attempting to overwrite existing checkpoint directory to
    protect previously trained models."""

    if run_name is not None:
        run_name = "_"+run_name

    model_checkpoints_dir = os.path.join(
        working_dir,
        "model_checkpoints",
        "%03d"%(run_ctr)+run_name
    )

    if os.path.isdir(model_checkpoints_dir):
        raise ValueError("Run directory %d already exists."%(run_ctr))

    os.system("mkdir -p {}".format(model_checkpoints_dir))

    return model_checkpoints_dir

def build_model(params):
    """Instantiates the model architecture inspired by MobileNet architecture..
    The model consists of pairs of depth-wise-separable and 1x1 point-wise \
    layers considered to be a block. The number of feature channels increases \
    for deeper blocks.

    :param input_shape: Shape of input layer.
    :param params: Parameters class.
    :returns: Untrained ANN Keras model.
    """

    p = params

    img_input = layers.Input(shape=p.input_shape)

    x = _conv_block(img_input, 32, params, strides=(2, 2), dropout=p.dropout)
    x = _depthwise_conv_block(x, 64, params, dropout=p.dropout*2, block_id=1)

    x = _depthwise_conv_block(x, 128, params, strides=(2, 2), dropout=p.dropout*4, block_id=2)
    x = _depthwise_conv_block(x, 128, params, dropout=p.dropout*8, block_id=3)

    if p.num_blocks > 1:
        x = _depthwise_conv_block(x, 256, params, strides=(2, 2), dropout=p.dropout*16, block_id=4)
        x = _depthwise_conv_block(x, 256, params, dropout=p.dropout*32, block_id=5)

    if p.num_blocks > 2:
        x = _depthwise_conv_block(x, 512, params, strides=(2, 2), dropout=p.dropout*32, block_id=6)
        x = _depthwise_conv_block(x, 512, params, dropout=p.dropout*32, block_id=7)

    if p.num_blocks > 3:
        x = _depthwise_conv_block(x, 512, params, dropout=p.dropout*32,block_id=8)
        x = _depthwise_conv_block(x, 512, params,dropout=p.dropout*32, block_id=9)

    if p.num_blocks > 4:
        x = _depthwise_conv_block(x, 512, params, dropout=p.dropout*32, block_id=10)
        x = _depthwise_conv_block(x, 512, params, dropout=p.dropout*32,block_id=11)

    if p.num_blocks > 5:
        x = _depthwise_conv_block(x, 1024, params, strides=(2, 2), block_id=12)
        x = _depthwise_conv_block(x, 1024, params, block_id=13)

    x = layers.AveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(p.num_classes, activation='softmax',use_bias=p.use_bias)(x)

    return models.Model(img_input, x, name='mobilenet')

def _conv_block(inputs, filters, params,
                kernel=(3, 3), strides=(1, 1), dropout=2**-6):
    """Adds an convolution layer."""

    p = params
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * p.alpha)

    if p.activity_regularizer is not None:
        activity_regularizer = regularizers.l2(p.activity_regularizer)
    else:
        activity_regularizer = None

    if p.max_value is not None:
        max_constraint = MaxConstraint(p.max_value)
    else:
        max_constraint = None

    x = layers.Conv2D(filters, kernel,
                      padding='same',
                      use_bias=p.use_bias,
                      strides=strides,
                      activity_regularizer=activity_regularizer,
                      kernel_regularizer=regularizers.l2(p.weight_regularizer),
                      kernel_constraint=max_constraint,
                      bias_constraint=max_constraint,
                      name='conv1')(inputs)
    x = layers.BatchNormalization(axis=channel_axis,
                                  beta_constraint=max_constraint,
                                  gamma_constraint=max_constraint,
                                  name='conv1_bn')(x)
    x = layers.ReLU(max_value=None, name='conv1_relu')(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x

def _depthwise_conv_block(inputs, pointwise_conv_filters, params,
                          strides=(1, 1), dropout=2**-6, block_id=1):
    """Adds a depth-wise convolution block."""

    p = params
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * p.alpha)

    if p.activity_regularizer is not None:
        activity_regularizer = regularizers.l2(p.activity_regularizer)
    else:
        activity_regularizer = None

    if p.max_value is not None:
        max_constraint = MaxConstraint(p.max_value)
    else:
        max_constraint = None

    x = layers.DepthwiseConv2D((3, 3),
                               padding='same',
                               strides=strides,
                               use_bias=p.use_bias,
                               activity_regularizer=activity_regularizer,
                               kernel_regularizer=regularizers.l2(p.weight_regularizer),
                               kernel_constraint=max_constraint,
                               bias_constraint=max_constraint,
                               name='conv_dw_%d' % block_id)(inputs)
    x = layers.BatchNormalization(axis=channel_axis,
                                  beta_constraint=max_constraint,
                                  gamma_constraint=max_constraint,
                                  name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(max_value=params.max_value, name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=p.use_bias,
                      strides=(1, 1),
                      activity_regularizer=activity_regularizer,
                      kernel_regularizer=regularizers.l2(p.weight_regularizer),
                      kernel_constraint=max_constraint,
                      bias_constraint=max_constraint,
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  beta_constraint=max_constraint,
                                  gamma_constraint=max_constraint,
                                  name='conv_pw_%d_bn' % block_id)(x)
    x = layers.ReLU(max_value=params.max_value, name='conv_pw_%d_relu' % block_id)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x

class Logger(keras.callbacks.Callback):
    """Logger to monitor training progress."""

    def __init__(self, log_dir=None, num_epochs=None):
        super(Logger, self).__init__()
        self.log_dir = log_dir
        self.num_epochs = num_epochs

    def on_train_begin(self, logs={}):
        self.i = 0
        self.t0 = time.time()

        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.dts = []

        self.fig = plt.figure(figsize=(20, 7))

    def on_epoch_end(self, epoch, logs={}):

        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_accuracies.append(logs.get('val_acc'))
        self.learning_rates.append(keras.backend.get_value(self.model.optimizer.lr))
        self.dts.append((time.time() - self.t0)/60)
        self.i += 1

        self.plot()

    def plot(self, showFig=False):
        plt.clf()

        ax1 = plt.subplot(1, 3, 1)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if self.num_epochs is not None:
            plt.xlim(0, self.num_epochs)

        ax1 = plt.subplot(1, 3, 2)
        plt.plot(self.x, self.accuracies, label="acc")
        plt.plot(self.x, self.val_accuracies, label="val_acc")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if self.num_epochs is not None:
            plt.xlim(0, self.num_epochs)
            plt.ylim(0, 1)

        ax1 = plt.subplot(1, 3, 3)
        plt.plot(self.x, self.learning_rates)
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')

        if self.num_epochs is not None:
            plt.xlim(0, self.num_epochs)

        plt.tight_layout()

        if showFig:
            plt.show()
        else:
            plt.savefig(os.path.join(self.log_dir, 'loss_acc_' + '.png'), dpi=300)

def get_dataset(dataset, params):
    """Loads training and test set from Keras."""

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    y_train = to_categorical(y_train, params.num_classes)
    y_test = to_categorical(y_test, params.num_classes)

    x_train = x_train / 255.
    x_test = x_test / 255.

    if params.signed_input:
        x_train = 2 * x_train - 1
        x_test = 2 * x_test - 1

    # Select a subset of the training set
    if params.max_num_training_samples is not None:
        x_train = x_train[:params.max_num_training_samples]
        y_train = y_train[:params.max_num_training_samples]

    return x_train, y_train, x_test, y_test

def get_dataset_full(dataset, params):
    """Loads training and test set from Keras
       and generate validation dataset, too."""

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    y_train = to_categorical(y_train, params.num_classes)
    y_test = to_categorical(y_test, params.num_classes)

    x_train = x_train / 255.
    x_test = x_test / 255.

    if params.signed_input:
        x_train = 2 * x_train - 1
        x_test = 2 * x_test - 1

    # Select a subset of the training set
    if params.max_num_training_samples is not None:
        x_train = x_train[:params.max_num_training_samples]
        y_train = y_train[:params.max_num_training_samples]
        
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      test_size=params.train_val_ratio, random_state=params.train_val_rand)

    return x_train, y_train, x_val, y_val, x_test, y_test

def train_model(params, x_train, y_train, x_test, y_test,
                model_checkpoints_dir):
    """Compiles and trains model given training and test data."""

    model = build_model(params)
    model.summary()

    # Load and continue from pre-trained model
    if params.pretrained_model_path is not None:
        model.load_weights(
            params.pretrained_model_path, by_name=True, skip_mismatch=False)

    # Setup model
    model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])

    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 rotation_range=10,
                                 horizontal_flip=True)

    it_train = datagen.flow(x_train, y_train, params.batch_size)

    cb_create_checkpoint = ModelCheckpoint(
        os.path.join(model_checkpoints_dir,
                     'mobilenet.{epoch:02d}-{val_acc:.2f}.h5'),
        verbose=1, save_best_only=True, monitor='val_acc')

    # Callback to callbacks used during training
    cb_reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=0.5,
        patience=15,
        verbose=0,
        mode='auto',
        min_delta=0.001,
        cooldown=0,
        min_lr=5 * 10 ** -7)

    cb_log_losses = Logger(log_dir=model_checkpoints_dir,
                           num_epochs=params.num_epochs)

    cb_stop_early = keras.callbacks.EarlyStopping(
        monitor='val_acc',
        min_delta=0.001,
        patience=30,
        verbose=1,
        mode='auto',
        baseline=0.8,
        restore_best_weights=False)

    # Train
    steps = int(x_train.shape[0] / params.batch_size)
    model.fit_generator(
        it_train, steps, params.num_epochs,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[cb_create_checkpoint, cb_reduce_learning_rate,
                   cb_log_losses, cb_stop_early]
    )

def train_models_alpha_num_blocks_sweep(params, alpha, num_blocks,
                                        run_ctr_start=1, name_prefix=None):
    """Trains a model for a range of alpha and num_blocks parameters.

    :param params: Parameters class.
    :param alpha: List of alpha values.
    :param num_blocks: List of num_blocks values.
    :param run_ctr_start: Numeric prefix for each model checkpoint directory \
    in order to enforce an alphabetic order of directories.
    :param name_prefix: Arbitrary prefix used for all model checkpoint \
    directories.
    """

    p = params
    x_train, y_train, x_test, y_test = get_dataset(cifar10, p)

    run_ctr = run_ctr_start
    for a in alpha:
        p.alpha = a
        for nb in num_blocks:
            p.num_blocks = nb
            run_name = p.to_string()

            if name_prefix is not None:
                run_name = name_prefix + "_" + run_name

            mcd = gen_model_checkpoint_dir(
                run_ctr, run_name, params.working_dir)
            train_model(params, x_train, y_train, x_test, y_test, mcd)

            run_ctr += 1


if __name__ == "__main__":
    p = Parameters(input_shape=(32, 32, 3),
                   batch_size=64,
                   num_epochs=400,
                   num_classes=10,
                   alpha=0.5,
                   num_blocks=2,
                   max_num_training_samples=5000,
                   weight_regularizer=1e-4,
                   dropout=2 ** -6,
                   use_bias=False,
                   pretrained_model_path=None,
                   max_value=None,
                   signed_input=False,
                   working_dir=os.path.join(os.path.expanduser("~"),
                               "projects",
                               "image_classification_pseudo_mobile_net"))

    x_train, y_train, x_test, y_test = get_dataset(cifar10, p)


    mcd = gen_model_checkpoint_dir(p.working_dir, run_ctr=0, run_name="test")
    train_model(p, x_train, y_train, x_test, y_test,
                model_checkpoints_dir=mcd)

    PRETRAINED_ANN_MODEL_PATH = os.path.join(os.path.abspath(''), '..',
                                             'models',
                                             'pseudo_mobile_net_0.87.h5')

    """
    working_dir = os.path.dirname(__file__)

    p = Parameters(input_shape=(32, 32, 3),
                   batch_size=64,
                   num_epochs=400,
                   num_classes=10,
                   alpha=1,
                   num_blocks=2,
                   max_num_training_samples=5000,
                   weight_regularizer=1e-4,
                   dropout=2**-6,
                   use_bias=False,
                   pretrained_model_path=None,
                   max_value=None,
                   signed_input=False)

    alpha = [0.5]
    num_blocks = [2]

    train_models_alpha_num_blocks_sweep(p, alpha, num_blocks,
                                        run_ctr_start=5, name_prefix=None)
    """


