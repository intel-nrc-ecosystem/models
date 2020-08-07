#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

"""
This test runs MNIST DNN module using batch mode encoding and checks if
accuracy is > 0.8.
"""

# Importing all the modules
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import keras
import unittest
from keras.layers import Dropout, Flatten, Conv2D, Input
from nxsdk_modules_ncl.dnn.src.utils import extract
from nxsdk_modules_ncl.dnn.src.dnn_layers import NxInputLayer, NxConv2D, NxModel, ProbableStates
from keras.datasets import mnist
from keras.utils import np_utils
from nxsdk_modules_ncl.dnn.src.utils import to_integer
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN as DNN
from nxsdk_modules_ncl.input_generator.input_generator import InputGenerator
from nxsdk.graph.monitor.probes import IntervalProbeCondition, SpikeProbeCondition
from nxsdk.logutils.nxlogging import set_verbosity, LoggingLevel
set_verbosity(LoggingLevel.ERROR)

# Setting up model parameters
batch_size = 32
num_training_epochs = 2
input_shape = (28, 28, 1)

num_steps_per_img = 512
num_train_samples = 60000
num_test_samples = 128

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input so we can train ANN with it.
# Will be converted back to integers for SNN layer.
x_train = x_train[:num_train_samples, :, :] / 255
x_test = x_test / 255

# Add a channel dimension.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode target vectors.
y_train = np_utils.to_categorical(y_train[:num_train_samples], 10)
y_test = np_utils.to_categorical(y_test, 10)


def train_ann_model():
    """
    Creating an ann in Keras and train int
    :return: ANN Model
    """
    from keras.models import Model
    train_model = True
    # Path for pre-trained model
    pretrained_model_path = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        '../../dnn',
        'models',
        'd_minist_model.h5')

    # Generate model
    if train_model or not os.path.isfile(pretrained_model_path):
        # Define model
        input_layer = Input(input_shape)

        layer = Conv2D(filters=16,
                       kernel_size=(5, 5),
                       strides=(2, 2),
                       input_shape=input_shape,
                       activation='relu')(input_layer)
        layer = Dropout(0.1)(layer)
        layer = Conv2D(filters=32,
                       kernel_size=(3, 3),
                       activation='relu')(layer)
        layer = Dropout(0.1)(layer)
        layer = Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       activation='relu')(layer)
        layer = Dropout(0.1)(layer)
        layer = Conv2D(filters=10,
                       kernel_size=(4, 4),
                       activation='softmax')(layer)
        layer = Flatten()(layer)

        model = Model(input_layer, layer)

        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

        # Training
        model.fit(x_train, y_train, batch_size, num_training_epochs, verbose=2,
                  validation_data=(x_test, y_test))

        # Save model
        model.save(pretrained_model_path)
    else:
        # Load pre-trained model
        model = keras.models.load_model(pretrained_model_path)

    model.summary()
    return model


def create_snn_model():
    """
    Create a Spiking Model of the MNIST dnn
    :return: Spiking Model
    """
    vth_mant = 2**9
    bias_exp = 6
    weight_exponent = 0
    synapse_encoding = 'sparse'

    inputLayer = NxInputLayer(input_shape,
                              vThMant=vth_mant,
                              biasExp=bias_exp)

    layer = NxConv2D(filters=16,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     input_shape=input_shape,
                     vThMant=vth_mant,
                     weightExponent=weight_exponent,
                     synapseEncoding=synapse_encoding)(inputLayer.input)
    layer = NxConv2D(filters=32,
                     kernel_size=(3, 3),
                     vThMant=vth_mant,
                     weightExponent=weight_exponent,
                     synapseEncoding=synapse_encoding)(layer)
    layer = NxConv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     vThMant=vth_mant,
                     weightExponent=weight_exponent,
                     synapseEncoding=synapse_encoding)(layer)
    layer = NxConv2D(filters=10,
                     kernel_size=(4, 4),
                     activation='softmax',
                     vThMant=vth_mant,
                     weightExponent=weight_exponent,
                     synapseEncoding=synapse_encoding)(layer)

    spiking_model = NxModel(inputLayer.input, layer,
                            numCandidatesToCompute=1)

    spiking_model.summary()
    return spiking_model


def map_ann_to_snn(model, spiking_model):
    """
    Map the ann model paramters to snn model
    :param model: ANN Model
    :param spiking_model: Spiking Model
    :return: Spiking Model
    """
    # Extract weights and biases from parameter list.
    parameters = model.get_weights()
    weights = parameters[0::2]
    biases = parameters[1::2]

    # Quantize weights and biases
    parameters_int = []
    for w, b in zip(weights, biases):
        w_int, b_int = to_integer(w, b, 8)
        parameters_int += [w_int, b_int]

    # Set quantized weigths and biases for spiking model
    spiking_model.set_weights(parameters_int)
    return spiking_model


def create_composable_model(spiking_model):
    """
    Create a composable DNN Model with Input Generator
    :param spiking_model: Spiking Model
    :returns: Composable model and Input Generator
    """
    import nxsdk.composable.model
    # NxModel is not yet implemented as a Composable -> Wrap it with DNN
    # composable class
    dnn = DNN(model=spiking_model, num_steps_per_img=num_steps_per_img)
    input_generator = InputGenerator(
        shape=input_shape, interval=num_steps_per_img)
    input_generator.setBiasExp(biasExpValue=6)
    model = nxsdk.composable.model.Model("dnn_pipeline")
    model.add(dnn)
    model.add(input_generator)
    input_generator.connect(dnn)
    input_generator.processes.inputEncoder.executeAfter(dnn.processes.reset)
    model.compile()
    return model, input_generator


def createProbes(spiking_model):
    """
    Add probes to the Spiking Model
    :param spiking_model: Spiking Model
    :return: List of spike probes
    """
    probeDt = 1
    probeStart = 1
    output_layer = spiking_model.layers[-1]
    spike_probes = []
    for i in range(int(np.prod(output_layer.output_shape[1:]))):
        spike_probes.append(output_layer[i].probe(
            state=ProbableStates.SPIKE,
            probeCondition=SpikeProbeCondition(dt=probeDt, tStart=probeStart)))
    return spike_probes


def runModel(model, input_generator, x_test, y_test, spike_probes):
    """
    Runs the model and gets the accuracy
    :param model: MNIST composable model
    :param input_generator: Input Generator
    :param x_test: The test vector
    :param y_test: The test data labels
    :param spike_probes: List of Spike Probes created
    :return: Accuracy
    """
    # Define batch size and number of batches
    batch_size = 16
    num_batches = 8

    # Initialize arrays for results
    num_samples = num_batches * batch_size
    classifications = np.zeros(num_samples, int)
    labels = np.zeros(num_samples, int)
    dts = np.zeros(num_samples)

    # Classify images
    x_test_int = (x_test * 255).astype(int)
    model.board.sync = False
    t_start_eff = time.time()
    for b in range(num_batches):
        print("Batch: {}".format(b))
        batch_x_test = x_test_int[b * batch_size:(b + 1) * batch_size]
        batch_y_test = y_test[b * batch_size:(b + 1) * batch_size]
        input_generator.batchEncode(batch_x_test)

        # Run model
        model.run(num_steps_per_img * batch_size)
        spike_trains = extract(spike_probes)[-num_steps_per_img * batch_size:]
        for i, (input_image, target) in enumerate(
                zip(batch_x_test, batch_y_test)):
            it = b * batch_size + i
            # Extract output
            spike_train = spike_trains[i * \
                num_steps_per_img:(i + 1) * num_steps_per_img]
            spike_count = np.sum(spike_train, 0)
            classifications[it] = np.argmax(spike_count)
            labels[it] = np.argmax(target)

    t_end_eff = time.time()
    model.disconnect()
    errors = classifications != labels
    num_errors = np.sum(errors)
    return (num_samples - num_errors) / num_samples


class TestInputGenWithBatchMode(unittest.TestCase):
    """
    This test runs MNIST DNN module using batch mode encoding and checks if
    accuracy is > 0.8.
    """
    @unittest.skip
    def test_input_generator_with_batch_mode(self):
        """
        This test runs MNIST DNN module using batch mode encoding and checks if
        accuracy is > 0.8.
        """
        ann_model = train_ann_model()
        spiking_model = create_snn_model()
        mapped_model = map_ann_to_snn(ann_model, spiking_model)
        model, input_generator = create_composable_model(mapped_model)
        spike_probes = createProbes(mapped_model)
        model.start(model.board)
        accuracy = runModel(
            model,
            input_generator,
            x_test,
            y_test,
            spike_probes)
        print("Accuracy with batched mode is : ", accuracy)
        self.assertTrue(
            accuracy > 0.8,
            msg="Accuracy not upto the mark : {}".format(accuracy))
