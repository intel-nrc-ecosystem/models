import unittest
from test import support

import os
import time

import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.datasets import cifar10

from nxsdk_modules_ncl.dnn.tests.test_dnn_compiler import extract, kernel_initializer, \
    _data_to_img, _plot_stimulus_response, normalize_image_dims
from nxsdk_modules_ncl.dnn.src.dnn_layers import NxInputLayer, NxConv2D, \
    NxModel, ProbableStates, NxFlatten, NxDense, NxLayer
from functools import partial


class Test_SoftReset(unittest.TestCase):
    """ Test the soft reset mode in rate-coded DNN. """

    def test_NxInputLayer(self):
        """
        Test input layer in soft-reset mode.
        """

        plot = False
        verbose = False

        resetMode = 'soft'
        neuronSize = 2 if resetMode == 'soft' else 1

        input_shape = (32, 32, 3)
        inputSize = np.prod(input_shape)
        inputLayer = NxInputLayer(input_shape, probeSpikes=True,
                                  vThMant=255, resetMode=resetMode)
        out = inputLayer.input

        model = NxModel(out, out)

        model.compile('adam', 'categorical_crossentropy', ['accuracy'])

        model.summary()

        layers = [inputLayer]

        input_image = np.linspace(0, 256,
                                  endpoint=False,
                                  num=inputSize).reshape(input_shape).astype(int)
        x_test = [input_image]
        y_test = [0]

        mapper = model.compileModel()
        if verbose:
            printLayerMappings(layers, mapper, synapses=True, inputAxons=True)
            printLayers(layers)

        layerProbes = []
        numProbes = 32
        for i, layer in enumerate(layers):
            shape = layer.output_shape[1:]

            # Define probes to read out currents.
            vProbes = []
            sProbes = []
            uProbes = []

            toProbe = numProbes if i == (len(layers) - 1) else numProbes * neuronSize
            toProbe = min(toProbe, int(np.asscalar(np.prod(shape))) * neuronSize)

            for j in range(toProbe):
                vProbes.append(layer[j].probe(ProbableStates.VOLTAGE))
                sProbes.append(layer[j].probe(ProbableStates.ACTIVITY))
                uProbes.append(layer[j].probe(ProbableStates.CURRENT))

            layerProbes.append([uProbes, vProbes, sProbes])

        # How many time steps to run each sample.
        num_steps = 1000
        # How many samples to test.
        num_samples_to_test = 1

        # Iterate over samples in testset.
        for i, (input_image, target) in enumerate(zip(x_test, y_test)):
            if i == num_samples_to_test:
                break

            if plot:
                plt.hist(input_image.ravel())
                plt.show()

            # Set input bias currents.
            for j, b in enumerate(np.ravel(input_image, 'F')):
                inputLayer[j * neuronSize].biasMant = b
                inputLayer[j * neuronSize].biasExp = 6
                inputLayer[j * neuronSize].phase = 2

            # Run model.
            model.run(num_steps)


        # Clean up.
        data = [[extract(probe) for probe in lp] for lp in layerProbes]
        spikesRates = []
        for i, (layer, d) in enumerate(zip(layers, data)):
            sData = d[2][:, (neuronSize - 1)::neuronSize]
            spikecount = (sData // 127).sum(0)
            spikesRates.append(spikecount / num_steps)

        model.disconnect()

        layer_activations = [x_test[0]]

        if plot:
            for activations, spikerate in zip(layer_activations, spikesRates):
                plt.figure()
                scale = np.max(activations)
                spikesFlat = spikerate #spikerate.flatten()
                plt.plot(activations.flatten('F')[:len(spikesFlat)] / scale,
                         spikesFlat, '.')

            plotLayerProbes(layers, data, neuronSize)

        cor = np.corrcoef(np.ravel(spikesRates[-1]),
                          np.ravel(layer_activations[0], 'F')[:numProbes // 2])[0, 1]
        self.assertGreater(cor, 0.99)

    def test_random_cor(self):
        """
        Tests the soft reset mode by comparing activations from an ANN
        to spike-rate from the converted SNN. The input and weights
        are randomly initialized.
        """

        seed = 123
        np.random.seed(seed)

        plot = False
        verbose = False

        visualizePartitions = False
        logger = None
        resetMode = 'soft'
        neuronSize = 2 if resetMode == 'soft' else 1

        inputShape = (4, 4, 1)
        inputScale = 255

        inputImage = np.random.randint(int(inputScale * 0.25),
                                       int(inputScale), inputShape)

        maxNumSpikes = 100
        numSteps = int(np.max(inputImage / 255) * maxNumSpikes)

        inputLayer = NxInputLayer(batch_input_shape=(1,) + inputShape,
                                  vThMant=255,
                                  visualizePartitions=visualizePartitions,
                                  resetMode=resetMode,
                                  probeSpikes=True)
        out = inputLayer.input

        layers = [inputLayer]

        # Conv2D
        kernelShape = (3, 3, 1)
        kernelScale = 4
        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = 2**9 - 1

        kernel_init = partial(kernel_initializer, kernelScale=kernelScale)

        numLayers = 1
        for i in range(numLayers):

            layer = NxConv2D(
                filters=kernelShape[-1], kernel_size=kernelShape[:-1],
                vThMant=vThMant, kernel_initializer=kernel_init,
                bias_initializer='ones', validatePartitions=False,
                probeSpikes=True, activation='relu', resetMode=resetMode)

            layers.append(layer)
            out = layer(out)

        model = NxModel(inputLayer.input, out, logger=logger)

        for layer in layers[1:]:
            weights, biases = layer.get_weights()
            weights, biases = to_integer(weights, biases, 8, np.max(weights) // 2)
            layer.set_weights([weights, biases])

        mapper = model.compileModel()

        if verbose:
            printLayerMappings(layers, mapper, synapses=True, inputAxons=True)
            printLayers(layers)
        print(model.summary())

        layerProbes = []

        for layer in layers:
            shape = layer.output_shape[1:]

            # Define probes to read out currents.
            vProbes = []
            sProbes = []
            uProbes = []

            for i in range(int(np.asscalar(np.prod(shape))) * neuronSize):
                vProbes.append(layer[i].probe(ProbableStates.VOLTAGE))
                sProbes.append(layer[i].probe(ProbableStates.ACTIVITY))
                uProbes.append(layer[i].probe(ProbableStates.CURRENT))

            layerProbes.append([uProbes, vProbes, sProbes])

        # Set bias currents
        for i, b in enumerate(np.ravel(inputImage, 'F')):
            inputLayer[i * neuronSize].biasMant = b
            inputLayer[i * neuronSize].phase = 2

        if verbose:
            for layer in layers:
                print(getCompartmentStates(layer, neuronSize))

        model.run(numSteps)

        if verbose:
            for layer in layers:
                print(getCompartmentStates(layer, neuronSize))

        model.disconnect()

        data = [[extract(probe) for probe in lp] for lp in layerProbes]

        if plot:
            plotLayerProbes(layers, data, neuronSize)

        spikesRates = []
        for i, (layer, d) in enumerate(zip(layers, data)):
            sData = d[2][:, (neuronSize - 1)::neuronSize]
            shape = layer.output_shape[1:]
            spikecount = _data_to_img(sData // 127, shape)
            spikesRates.append(spikecount / numSteps)


        batchInputImage = np.expand_dims(inputImage, 0)
        activations = model.predict(batchInputImage)[0]

        if plot:
            plt.figure()
            plt.plot(inputImage.flatten(), spikesRates[0].flatten(), '.')
            plt.show()

            plt.figure()
            plt.plot(activations.flatten(), spikesRates[-1].flatten(), '.')
            plt.show()

            plt.figure()
            plt.imshow(normalize_image_dims(activations))
            plt.show()

            plt.figure()
            plt.imshow(normalize_image_dims(spikesRates[-1]))
            plt.show()

        cor = np.corrcoef(np.ravel(spikesRates[-1]), np.ravel(activations))[0, 1]
        self.assertGreater(cor, 0.99)
        if verbose:
            print(cor)


def printLayerMappings(layers, mapper, compartments=True,
                       synapses=True, inputAxons=True, outputAxons=True):
    """
    Helper function to pring layer mappings.

    :param list<NxLayer> layers: List of layers to print mapping
    :param DnnMapper mapper: mapper associated with layers
    :param compartments: Print compartment mapping
    :param synapses: Print synapse mapping
    :param inputAxons: Print input axon mapping
    :param outputAxons: Print output axon mapping
    """
    assert len(layers) > 0
    for layer in layers:
        cores = np.unique(layer._cxResourceMap[:, 1])
        for coreId in cores:
            n2Core = layer._board.n2Chips[0].coreMap[coreId]
            print(layer.name, n2Core.id)
            mapper.printCore(n2Core, compartments=compartments, synapses=synapses,
                             inputAxons=inputAxons, outputAxons=outputAxons)


def printLayers(layers):
    """ Helper function to print cores associated with a list of layers. """
    for layer in layers:
        printLayer(layer)


def printLayer(layer):
    """ Helper function to print core attributes from cores associated with layer. """
    print(layer.name)
    coreIds = np.unique(layer._cxResourceMap[:, 1])
    for coreId in coreIds:
        printCore(layer._board.n2Chips[0].coreMap[coreId])


def printCore(core):
    """" Helper function to print core attributes. """
    attrs = ['id', 'cxCfg', 'cxProfileCfg', 'vthProfileCfg',
             'cxMetaState', 'synapseMap', 'synapseFmt',
             'synapseMem', 'synapses', 'axons', 'axonCfg', 'axonMap']
    for attr in attrs:
        coreAttr = getattr(core, attr)
        if hasattr(coreAttr, 'modified'):
            coreAttr = len(getattr(coreAttr, 'modified'))
        if isinstance(coreAttr, list):
            coreAttr = len(coreAttr)
        print(attr + ': ', coreAttr)


def plotLayerProbes(layers, layerProbes, neuronSize, save=False):
    """
    Plot probes for all layers.

    :param list layers:
    :param list layerProbes:
    """
    for layer, lp in zip(layers, layerProbes):
        numProbes = len(lp)
        fig, axes = plt.subplots(neuronSize, numProbes) #, dpi=200)
        axes = axes.reshape((neuronSize, numProbes))
        for j, data in enumerate(lp):
            for i in range(neuronSize):
                d = data[:, i::neuronSize]
                plt.sca(axes[i, j])
                plt.plot(d)
                plt.ylabel(ProbableStates(j))
                plt.title(layer.name)
        plt.show()
    if save:
        plt.savefig()


def getCompartmentStates(layer, neuronSize):
    """
    Get the state variables for the compartments in a layer.

    :param NxLayer layer: NxLayer to get compartment states
    :return np.ndarray:  compartment states
    """
    states = []
    for i in range(np.prod(layer.output_shape[1:]) * neuronSize):
        states.append([layer[i].current,
                       layer[i].voltage,
                       layer[i].biasMant,
                       layer[i].biasExp,
                       layer[i].phase])
    return np.array(states)


def to_integer(weights, biases, bitwidth, norm):
    a_min = - 2 ** bitwidth
    a_max = - a_min - 1
    weights = np.clip(weights / norm * a_max, a_min, a_max).astype(int)
    biases = np.clip(biases / norm * a_max, a_min, a_max).astype(int)
    return weights, biases


def main():
    """ Run unit test. """
    support.run_unittest(Test_SoftReset)


if __name__ == '__main__':
    main()