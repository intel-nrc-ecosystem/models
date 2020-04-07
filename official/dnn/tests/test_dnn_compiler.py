# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2018 Intel Corporation.
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

"""Test DNN compilation."""

import logging
import unittest
from collections import namedtuple
from functools import partial
from io import StringIO
from test import support

import keras.backend as k
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model
from keras.layers import Flatten, Dense

from nxsdk_modules.dnn.src.dnn_layers import NxInputLayer, NxConv2D, \
    NxModel, ProbableStates, NxAveragePooling2D, NxDepthwiseConv2D, NxConv1D, \
    NxDense, NxFlatten
from nxsdk_modules.dnn.src.utils import extract


class TestDnnCompiler(unittest.TestCase):
    """Test CNN compilation."""

    def setUp(self):
        """Setup logger and verbosity."""

        self.verbose = False
        self.stream = StringIO()
        self.logger = logging.getLogger()
        self.handler = logging.StreamHandler(stream=self.stream)
        for hdlr in self.logger.handlers:
            self.logger.removeHandler(hdlr)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        """Remove the stream handler for rest of the tests."""

        self.handler.close()
        self.logger.removeHandler(self.handler)

    @staticmethod
    def _setup_2layer_stimulus_net(inputImage, vTh, padding='same',
                                   verbose=False):
        """Helper method to set up a 2-layer CNN.

        The output layer has a single kernel with all weights equal to 1.
        """

        assert isinstance(inputImage, np.ndarray)

        inputShape = inputImage.shape

        inputLayer = NxInputLayer(inputShape, vThMant=vTh, biasExp=0,
                                  visualizePartitions=False)
        outputLayer = NxConv2D(
            filters=1, kernel_size=3, padding=padding,
            vThMant=1000, weightExponent=0, synapseEncoding='sparse',
            kernel_initializer='ones', bias_initializer='zeros',
            visualizePartitions=False)

        model = NxModel(inputLayer.input, outputLayer(inputLayer.input),
                        numCandidatesToCompute=1)

        mapper = model.compileModel()

        if verbose:
            hiddenCore = model.board.n2Chips[0].n2CoresAsList[1]
            outputCore = model.board.n2Chips[0].n2CoresAsList[0]

            mapper.printCore(hiddenCore, compartments=True)
            mapper.printCore(outputCore, compartments=True)

        outputShape = model.output_shape[1:]

        # Define probes to read out current and voltages
        vProbes1 = []
        uProbes2 = []
        for i in range(int(np.asscalar(np.prod(inputShape)))):
            vProbes1.append(
                inputLayer[i].probe(state=ProbableStates.VOLTAGE))
        for i in range(int(np.asscalar(np.prod(outputShape)))):
            uProbes2.append(
                outputLayer[i].probe(state=ProbableStates.CURRENT))

        # Set bias currents
        for i, b in enumerate(np.ravel(inputImage, 'F')):
            inputLayer[i].biasMant = b
            inputLayer[i].phase = 2

        return model, vProbes1, uProbes2

    def test_stimulus_1_padding_same(self):
        """Check correct propagation of spikes one layer to the next.

        Sets up a two layer network with one channel each.

        Here we stimulate a single node in input layer and validate
        excitation of current in feature layer.
        """

        doPlot = False

        # Configures input layer to produce a single spike that's detected in
        #  output layer
        padding = 'same'
        bias = 10
        vTh = 1
        numSteps = (vTh << 6) // bias + 1 + 1 + 1

        # Define size and an arbitrary single pixel stimulus of input layer
        inputImage = np.zeros((3, 3, 1), int)
        inputImage[0, 1, 0] = bias

        # Set up network and run
        model, vProbes1, uProbes2 = self._setup_2layer_stimulus_net(
            inputImage, vTh, padding, self.verbose)
        model.run(numSteps)
        model.disconnect()

        # Extract probe data
        outputShape = model.output_shape[1:]
        data = extract(uProbes2)
        imgData = _data_to_img(data // 2 ** 6, outputShape)

        target = np.asarray([[1, 1, 1], [1, 1, 1], [0, 0, 0]], int)
        target = np.expand_dims(target, -1)

        self.assertTrue(np.array_equal(imgData, target))

        if doPlot:
            plt.figure(1)
            plt.imshow(imgData)

            plt.figure(2)
            _plot_stimulus_response(vProbes1, uProbes2)

            plt.show()

            print("Hidden layer resourceMap:")
            print(model.layers[0]._cxResourceMap)

            print("Output layer resourceMap:")
            print(model.layers[1]._cxResourceMap)

    def test_stimulus_2_padding_valid(self):
        """Same as test_stimulus_1 but using padding='valid'."""

        doPlot = False

        # Configures input layer to produce a single spike that's detected in
        #  output layer
        padding = 'valid'
        bias = 10
        vTh = 1
        numSteps = (vTh << 6) // bias + 1 + 1 + 1

        # Define size and an arbitrary single pixel stimulus of input layer
        inputImage = np.zeros((8, 8, 1), int)
        inputImage[1, 3, 0] = bias

        # Set up network and run
        model, vProbes1, uProbes2 = self._setup_2layer_stimulus_net(
            inputImage, vTh, padding, self.verbose)
        model.run(numSteps)
        model.disconnect()

        # Extract probe data
        outputShape = model.output_shape[1:]
        data = extract(uProbes2)
        imgData = _data_to_img(data // 2 ** 6, outputShape)

        target = np.asarray([[0, 1, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]], int)
        target = np.expand_dims(target, -1)

        self.assertTrue(np.array_equal(imgData, target))

        if doPlot:
            plt.figure(1)
            plt.imshow(imgData)

            plt.figure(2)
            _plot_stimulus_response(vProbes1, uProbes2)

            plt.show()

            print("Hidden layer resourceMap:")
            print(model.layers[0]._cxResourceMap)

            print("Output layer resourceMap:")
            print(model.layers[1]._cxResourceMap)

    def test_stimulus_3(self):
        """Check correct propagation of spikes one layer to the next.

        Sets up a two layer network with one channel each.

        Here we stimulate a single node in input layer and validate
        excitation of current in feature layer, by sweeping across a certain
        patch of the input.
        """

        doPlot = False

        padding = 'same'
        bias = 10
        vTh = 1
        numSteps = (vTh << 6) // bias + 1 + 1 + 1

        nY = 45
        nX = 5

        i = 0
        for x in range(nX):
            for y in range(36, 38):
                inputImage = np.zeros((nY, nX, 1), int)
                inputImage[y, x, 0] = bias

                model, vProbes1, uProbes2 = self._setup_2layer_stimulus_net(
                    inputImage, vTh, padding, self.verbose)
                model.run(numSteps)
                model.disconnect()

                outputShape = model.output_shape[1:]
                data = extract(uProbes2)
                imgData = _data_to_img(data // 2 ** 6, outputShape)

                assert np.max(imgData) == 1, \
                    "x = {}, y = {}, max = {}".format(x, y, np.max(imgData))
                if np.max(imgData) > 1:
                    print("--------------------------------------------------")
                    print("Error for x = {}, y = {}, max = {}".format(
                        x, y, np.max(imgData)))
                    print("--------------------------------------------------")

                if doPlot:
                    plt.title("x = {}, y = {}".format(x, y))
                    plt.imshow(imgData[:, :, 0])
                    plt.show()

                i += 1

    def test_stimulus_4(self):
        """Check correct propagation of spikes one layer to the next.

        Sets up a two layer network with one channel each, and sweeps across
        various input sizes.
        """

        doPlot = False

        padding = 'same'
        bias = 10
        vTh = 1
        numSteps = (vTh << 6) // bias + 1 + 1 + 1
        stepSize = 20
        minSize = 60
        maxSize = 101
        numSizes = (maxSize - minSize) // stepSize
        sizeY = int(np.sqrt(numSizes))
        sizeX = int(np.ceil(numSizes/sizeY))

        sizes = range(minSize, maxSize, stepSize)
        for i, s in enumerate(sizes):
            inputImage = np.ones((s, s, 2), int) * bias
            model, vProbes1, uProbes2 = self._setup_2layer_stimulus_net(
                inputImage, vTh, padding, self.verbose)
            model.run(numSteps)
            model.disconnect()

            outputShape = model.output_shape[1:]
            data = extract(uProbes2)
            imgData = _data_to_img(data // 2 ** 6, outputShape)

            # In center region, 2*9 kernels overlap each pixel
            self.assertTrue(np.all(imgData[1:-1, 1:-1] == 18))
            # Along edges, 2*6 kernels overlap each pixel
            self.assertTrue(np.all(imgData[1:-1, 0] == 12))
            self.assertTrue(np.all(imgData[1:-1, -1] == 12))
            self.assertTrue(np.all(imgData[0, 1:-1] == 12))
            self.assertTrue(np.all(imgData[-1, 1:-1] == 12))
            # In corners, 2*4 kernels overlap each pixel
            self.assertTrue(np.all(imgData[0, 0] == 8))
            self.assertTrue(np.all(imgData[0, -1] == 8))
            self.assertTrue(np.all(imgData[-1, 0] == 8))
            self.assertTrue(np.all(imgData[-1, -1] == 8))

            if doPlot:
                plt.subplot(sizeY, sizeX, i + 1)
                plt.title("Size={}".format(s))
                plt.imshow(imgData[:, :, 0])

        if doPlot:
            plt.show()

    def test_compile(self):
        """Check compilation of NxModel and retrieval of cxResourceMap."""

        # Create an arbitrary model
        inputShape = (31, 35, 2)
        inputLayer = NxInputLayer(inputShape)
        hiddenLayer = NxConv2D(4, 3)
        outputLayer = NxConv2D(7, 3)
        model = NxModel(inputLayer.input,
                        outputLayer(hiddenLayer(inputLayer.input)),
                        numCandidatesToCompute=1)

        model.compileModel()

        if self.verbose:
            hiddenLayer.exclusionCriteria.print()
            outputLayer.exclusionCriteria.print()

            for layer in model.layers:
                if isinstance(layer, NxConv2D):
                    print(layer._cxResourceMap)

    def test_synMem(self):
        """Test issue where nxcompiler runs out of synMem.

        The problem is that whenever the number of syn bits is an exact
        multiple of 64, nxcompiler requires an extra mem word. E.g. 128 bits
        would count as 3 words.

        Normally, this test would fail due to an assertion in the nxcompiler
        that checks for number of syn mem words to be smaller than 2**14.

        We have implemented a check in the NxTF compiler that allocates an
        extra mem word when it detects that the numBits are an exact multiple
        of 64. So this test should pass.
        """

        np.random.seed(123)
        inputLayer = NxInputLayer((1, 1, 256))

        x = NxConv2D(256, (1, 1), name='conv1')(inputLayer.input)

        model = NxModel(inputLayer.input, x)

        model.compileModel()
        model.run(1)
        model.disconnect()

    def test_synMem2(self):
        """Test issue where nxcompiler runs out of synMem.

        The problem is that whenever the number of syn bits is an exact
        multiple of 64, nxcompiler requires an extra mem word. E.g. 128 bits
        would count as 3 words.

        We have implemented a check in the NxTF compiler that allocates an
        extra mem word when it detects that the numBits are an exact multiple
        of 64. So this test should pass.
        """

        inputLayer = NxInputLayer((1, 1, 1))
        x = NxConv2D(6, (1, 1), name='conv1', kernel_initializer='ones',
                     synapseEncoding='dense1',
                     useSharedSign=False)(inputLayer.input)
        model = NxModel(inputLayer.input, x, saveOutput=False)
        model.compileModel()
        model.run(1)
        model.disconnect()

        layers = model.partitionOptimizer.getLayers()
        assert layers[1].numSynMemWords == 2

    def test_Conv2DBiases(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of an input layer connected to a convolution
        layer. The input to the network is a square gray-scale image of random
        integers. The weights are also random integers. Biases are non-zero.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        def bias_initializer(shape, dtype=None, biasScale=1):
            """Increasing integer bias initializer for Keras layer.

            :param list | tuple | np.ndarray shape: Shape of biases.
            :param str | type | None dtype: Data type of biases.
            :param int biasScale: Scale factor applied to the biases.

            :return: Bias tensor.
            """

            return k.constant(np.arange(biasScale), dtype, shape)

        numFilters = 16
        kernelShape = (3, 3)
        inputShape = (28, 28, 3)
        vThMant = numFilters * 2 ** 4
        thrGain = 2 ** 6
        visualizePartitions = False
        plotUV = False
        numSteps = 500

        bias_init = partial(bias_initializer, biasScale=numFilters)

        layer = NxConv2D(
            filters=numFilters, kernel_size=kernelShape, vThMant=vThMant,
            kernel_initializer='zeros', bias_initializer=bias_init,
            validatePartitions=True, probeSpikes=True, activation='relu',
            strides=(2, 2))

        inputLayer = NxInputLayer(batch_input_shape=(1,) + inputShape,
                                  vThMant=vThMant,
                                  visualizePartitions=visualizePartitions)

        model = NxModel(inputLayer.input, layer(inputLayer.input))

        model.compileModel()

        outputShape = layer.output_shape[1:]

        # Define probes to read out currents.
        vProbes = []
        sProbes = []
        for i in range(int(np.asscalar(np.prod(outputShape)))):
            vProbes.append(layer[i].probe(ProbableStates.VOLTAGE))
            sProbes.append(layer[i].probe(ProbableStates.ACTIVITY))

        model.run(numSteps)
        model.disconnect()

        data = extract(sProbes)
        spikecount = _data_to_img(data // 127, outputShape)
        spikerates = spikecount / numSteps

        batchInputImage = np.expand_dims(np.zeros(inputShape), 0)
        activations = model.predict(batchInputImage)[0] / (vThMant * thrGain)

        if plotUV:
            plt.figure(1)
            _plot_stimulus_response(vProbes, sProbes)
            plt.show()

            plt.figure(2)
            plt.plot(activations.flatten(), spikerates.flatten(), '.')
            plt.show()

            plt.figure(3)
            plt.imshow(normalize_image_dims(activations))
            plt.show()

            plt.figure(4)
            plt.imshow(normalize_image_dims(spikerates))
            plt.show()

        corr = np.corrcoef(np.ravel(spikerates), np.ravel(activations))[0, 1]

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_Conv2D(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of an input layer connected to a convolution
        layer. The input to the network is a square gray-scale image of random
        integers. The weights are also random integers. Biases are zero.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        kernelShape = (3, 3, 1)
        kernelScale = 10
        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = int(np.asscalar(np.prod(kernelShape))) * kernelScale

        kernel_init = partial(kernel_initializer, kernelScale=kernelScale)

        layer = NxConv2D(
            filters=kernelShape[-1], kernel_size=kernelShape[:-1],
            vThMant=vThMant, kernel_initializer=kernel_init,
            bias_initializer='zeros', validatePartitions=True,
            probeSpikes=True, activation='relu')

        corr = runCorrelationRandom(layer, vThMant)

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_DepthwiseConv2D(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of an input layer connected to a
        depthwise-separable convolution layer. The input to the network is a
        square gray-scale image of random integers. The weights are also random
        integers. Biases are zero.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        kernelShape = (3, 3)
        kernelScale = 10

        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = int(np.asscalar(np.prod(kernelShape))) * kernelScale

        kernel_init = partial(kernel_initializer, kernelScale=kernelScale)
        layer = NxDepthwiseConv2D(
            kernel_size=kernelShape, vThMant=vThMant, validatePartitions=True,
            depthwise_initializer=kernel_init, bias_initializer='zeros',
            probeSpikes=True, activation='relu')

        corr = runCorrelationRandom(layer, vThMant)

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_Dense(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of a 1D input layer connected to a fully-connected
        layer. The input to the network is a vector of random integers.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = 10

        kernelScale = 3
        kernel_init = partial(kernel_initializer, kernelScale=kernelScale)

        layer = NxDense(10, vThMant=vThMant, validatePartitions=True,
                        kernel_initializer=kernel_init, probeSpikes=True,
                        bias_initializer='zeros')

        corr = runCorrelationRandom(layer, vThMant, inputShape=(49,),
                                    insertFlatten=False)

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_Dense2(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of a 3D input layer connected to a flatten and a
        fully-connected layer. The input to the network is a square gray-scale
        image of random integers.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = 10

        kernelScale = 3
        kernel_init = partial(kernel_initializer, kernelScale=kernelScale)

        layer = NxDense(10, vThMant=vThMant, validatePartitions=True,
                        kernel_initializer=kernel_init, probeSpikes=True,
                        bias_initializer='zeros', activation='relu')

        corr = runCorrelationRandom(layer, vThMant, insertFlatten=True)

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_Conv1D(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of an input layer connected to a Conv1D layer.
        The input to the network is a square gray-scale image of random
        integers.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        inputShape = (20, 3)
        kernelShape = (3,)
        kernelScale = 10

        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = int(np.asscalar(np.prod(kernelShape))) * kernelScale

        kernel_init = partial(kernel_initializer, kernelScale=kernelScale)

        layer = NxConv1D(2, kernel_size=kernelShape, vThMant=vThMant,
                         kernel_initializer=kernel_init, probeSpikes=True,
                         bias_initializer='zeros', validatePartitions=True,
                         activation='relu')

        corr = runCorrelationRandom(layer, vThMant, inputShape=inputShape)

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_DilatedConv1D(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of an input layer connected to a Conv1D layer.
        The input to the network is a square gray-scale image of random
        integers.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        inputShape = (20, 3)
        kernelShape = (3,)
        kernelScale = 10

        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = int(np.asscalar(np.prod(kernelShape))) * kernelScale

        kernel_init = partial(kernel_initializer, kernelScale=kernelScale)

        layer = NxConv1D(2, kernel_size=kernelShape, vThMant=vThMant,
                         kernel_initializer=kernel_init, probeSpikes=True,
                         bias_initializer='zeros', validatePartitions=True,
                         activation='relu', dilation_rate=2, padding='same')

        corr = runCorrelationRandom(layer, vThMant, inputShape=inputShape)

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_AveragePooling2D(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of an input layer connected to a pooling layer.
        The input to the network is a square gray-scale image of random
        integers.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        poolShape = (2, 2)
        scale = 1

        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = int(np.asscalar(np.prod(poolShape))) * scale

        layer = NxAveragePooling2D(pool_size=poolShape, vThMant=vThMant,
                                   validatePartitions=True, probeSpikes=True)

        corr = runCorrelationRandom(layer, vThMant)

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    def test_Flatten(self):
        """Test correlation between ANN activations and SNN spikerates.

        The network consists of a 3D input layer, followed by a flatten layer,
        which has 1-to-1 connections to the output layer. The input pixel
        values are set to ascending integers in Fortran style.

        This test asserts that the spikerates in the output layer are close to
        the ANN activations, by computing the Pearson correlation coefficient.
        A perfect correlation cannot be expected due to quantization errors
        when approximating ANN activations with discrete spikes. However,
        correlations should be higher than 0.99.
        """

        visualizePartitions = False
        doPlot = False

        # Height, width, depth
        inputShape = (3, 4, 5)

        numInputNeurons = int(np.asscalar(np.prod(inputShape)))
        numOutputNeurons = numInputNeurons - 1
        inputScale = 255

        thrToInputRatio = 2 ** 7
        thrGain = 2 ** 6

        # No need to divide by thrGain because spike input receives equal gain.
        vThMant = 1

        vThMantInput = thrToInputRatio * inputScale // thrGain

        maxNumSpikes = 100
        numSteps = thrToInputRatio * maxNumSpikes

        weights = np.eye(numInputNeurons, numOutputNeurons, dtype=int)
        biases = np.zeros(numOutputNeurons, int)

        nxInput = NxInputLayer(inputShape, vThMant=vThMantInput,
                               visualizePartitions=visualizePartitions)
        nxLayer = NxDense(numOutputNeurons, weights=[weights, biases],
                          vThMant=vThMant, validatePartitions=True,
                          probeSpikes=True)
        nxModel = NxModel(nxInput.input, nxLayer(NxFlatten()(nxInput.input)))
        nxModel.compileModel()

        kerasInput = Input(inputShape)
        kerasLayer = Dense(numOutputNeurons,
                           weights=[weights, biases])(Flatten()(kerasInput))
        kerasModel = Model(kerasInput, kerasLayer)

        # Define probes to read out currents.
        sProbes = []
        for i in range(numOutputNeurons):
            sProbes.append(nxLayer[i].probe(ProbableStates.ACTIVITY))

        # Set bias currents
        inputImage = np.reshape(np.arange(numInputNeurons), inputShape, 'F')
        inputImage = inputImage % 255
        for i, b in enumerate(np.ravel(inputImage, 'F')):
            nxInput[i].biasMant = b
            nxInput[i].phase = 2

        nxModel.run(numSteps)
        nxModel.disconnect()

        data = extract(sProbes)
        spikecount = _data_to_img(data // 127, nxLayer.output_shape[1:])
        spikerates = spikecount / numSteps * thrToInputRatio

        batchInputImage = np.expand_dims(inputImage, 0)
        activations = \
            kerasModel.predict(batchInputImage)[0] / (vThMant * thrGain)

        if doPlot:

            plt.figure(3)
            plt.imshow(normalize_image_dims(inputImage))
            plt.show()

            plt.figure(6)
            plt.plot(activations.flatten(), spikerates.flatten(), '.')
            plt.show()

            plt.figure(7)
            plt.imshow(normalize_image_dims(activations))
            plt.show()

            plt.figure(8)
            plt.imshow(normalize_image_dims(spikerates))
            plt.show()

        corr = np.corrcoef(np.ravel(spikerates), np.ravel(activations))[0, 1]

        self.assertAlmostEqual(corr, 1, 2,
                               msg="Correlation between ANN activations "
                                   "and SNN spikerates is too low.")

    @unittest.skip("Expected to fail for certain corner cases.")
    def test_twoConv2D(self):
        """Test correlation between ANN activations and SNN spikecounts.

        The network consists of an input layer and two convolution layers. The
        input to the network is uniformely white. The weights are also uniform
        arrays of ones. Biases are zero.

        The thresholds and the number of timesteps are tuned such that every
        layer spikes exactly once. We measure the number of spikes that each
        post-synaptic neuron receives, and compare this number against the ANN
        activations. In contrast to ``test_correlationRandom``, we can expect
        perfect correlation, because inputs and weights are uniform, so there
        is no irregularity in the firing dynamics and no need to integrate over
        several discrete spikes to estimate a rate.

        The test is parameterized to cover a large number of combinations of
        network design choices, input shapes, synapse encodings etc.

        If a particular configuration fails the correlation check, the error
        message contains a code snippet that can be run directly to reproduce
        only the failed subtest.
        """

        np.random.seed(123)

        # Define parameter space to sweep.
        # ################################

        # Todo: debug and enable 'runlength'.
        encodings = ['sparse', 'dense1', 'dense2']

        paddings = ['valid', 'same']

        stridesY = [1, 2]
        stridesX = [1, 2]

        # Todo: image size should range to 256, but currently this takes too
        #       long to run.
        imgHeights = np.arange(10, 20)
        imgWidths = np.arange(10, 20)
        imgDepths = [1, 3]

        kernelHeights = [1, 2, 3, 5, 7]
        kernelWidths = [1, 2, 3, 5, 7]
        # Todo: kernelDepths should range to 2 ** 10, but currently this takes
        #       too long to run.
        kernelDepths = np.insert(np.logspace(0, 5, 5, True, 2, int), 2, 3)

        # Reducing this number forces the partitioner to create more
        # partitions, which allows us to test various splittings.
        maxNumCxPerCore = [512, 1024]

        # Choose a subset of points in parameter space for testing.
        # #########################################################

        def selectN(arr, num):
            """Randomly select ``num`` items from ``arr``.

            :param list | np.ndarray arr: Input array to choose from.
            :param int num:
            :return: List of ``num`` items from ``arr``.
            :rtype: list
            """

            return [arr[j] for j in np.random.choice(len(arr), num, False)]

        # How many points to choose along each dimension.
        N = 2

        # Parameters of input layer.
        paramsInput = []
        for maxNumCx in maxNumCxPerCore:
            for imgDepth in imgDepths:
                for imgHeight in selectN(imgHeights, N):
                    for imgWidth in selectN(imgWidths, N):
                        paramsInput.append(InputParams(imgHeight,
                                                       imgWidth,
                                                       imgDepth,
                                                       maxNumCx))

        # Parameters of hidden layer.
        paramsHidden = []
        for maxNumCx in maxNumCxPerCore:
            for encoding in encodings:
                for padding in paddings:
                    for strideX in stridesX:
                        for strideY in stridesY:
                            for kernelHeight in selectN(kernelHeights, N):
                                for kernelWidth in selectN(kernelWidths, N):
                                    for kernelDepth in selectN(kernelDepths,
                                                               N):
                                        paramsHidden.append(LayerParams(
                                            encoding,
                                            padding,
                                            strideY,
                                            strideX,
                                            kernelHeight,
                                            kernelWidth,
                                            kernelDepth,
                                            maxNumCx
                                        ))

        # Parameters of output layer.
        paramsOutput = []
        for maxNumCx in maxNumCxPerCore:
            for encoding in encodings:
                for padding in paddings:
                    for strideX in stridesX:
                        for strideY in stridesY:
                            for kernelHeight in selectN(kernelHeights, N):
                                for kernelWidth in selectN(kernelWidths, N):
                                    for kernelDepth in selectN(kernelDepths,
                                                               N):
                                        paramsOutput.append(LayerParams(
                                            encoding,
                                            padding,
                                            strideY,
                                            strideX,
                                            kernelHeight,
                                            kernelWidth,
                                            kernelDepth,
                                            maxNumCx
                                        ))

        print("Proposed {{{}, {}, {}}} points for {{input, hidden, output}} "
              "layer.".format(len(paramsInput), len(paramsHidden),
                              len(paramsOutput)))

        # Pick a combination of parameters from input, hidden and output layer.
        paramsAll = []
        for argsInput in paramsInput:
            for argsHidden in selectN(paramsHidden, len(paramsHidden)//20):
                for argsOutput in selectN(paramsOutput, len(paramsOutput)//20):
                    paramsAll.append((argsInput, argsHidden, argsOutput))

        print("Selected {} points in search space.".format(len(paramsAll)))

        # Iterate over selected parameters.
        # #################################

        for n, (argsInput, argsHidden, argsOutput) in enumerate(paramsAll):

            with self.subTest(argsInput=argsInput,
                              argsHidden=argsHidden,
                              argsOutput=argsOutput):

                print("Starting testrun {}.".format(n))

                runModelFromConfig(argsInput, argsHidden, argsOutput, n)


InputParams = namedtuple('InputParams', ['imgHeight', 'imgWidth',
                                         'imgDepth',
                                         'maxNumCxPerCore'])

LayerParams = namedtuple('LayerParams', ['encoding', 'padding',
                                         'strideY', 'strideX',
                                         'kernelHeight', 'kernelWidth',
                                         'kernelDepth',
                                         'maxNumCxPerCore'])


def runModelFromConfig(argsInput, argsHidden, argsOutput, n=None):
    """A helper function used by test_correlationUniform.

    Given a model configuration, this function creates a NxModel, checks
    that the model architecture is valid, then partitions, maps and runs the
    network with a constant white input image.

    The function probes voltage and current states, which can be plotted.
    Spikecountss are derived from the current trace.and are used to assert
    that the SNN activity equals the ANN activations.

    :param InputParams argsInput: Parameters for input layer.
    :param LayerParams argsHidden: Parameters for hidden layer.
    :param LayerParams argsOutput: Parameters for output layer.
    :param int n: The id of the current config. Only used for printing.
    """

    # Define stimulus and runtime settings.
    # #####################################

    inputScale = 1
    vThMantInput = 1
    vThMant = 1
    numSteps = (vThMantInput << 6) // inputScale + 1 + 1 + 1 + 1
    biasExp = 0
    visualizePartitions = False
    plotUV = False

    # Build model.
    # ############

    inputShape = (argsInput.imgHeight, argsInput.imgWidth, argsInput.imgDepth)
    batchInputShape = (1,) + inputShape

    inputLayer = NxInputLayer(batch_input_shape=batchInputShape,
                              biasExp=biasExp, vThMant=vThMantInput,
                              visualizePartitions=visualizePartitions)

    inputLayer._maxNumCompartments = argsInput.maxNumCxPerCore

    hiddenLayer = NxConv2D(
        argsHidden.kernelDepth,
        (argsHidden.kernelHeight, argsHidden.kernelWidth),
        strides=(argsHidden.strideY, argsHidden.strideX),
        padding=argsHidden.padding, vThMant=vThMant,
        synapseEncoding=argsHidden.encoding, kernel_initializer='ones',
        visualizePartitions=visualizePartitions, validatePartitions=True)

    hiddenLayer._maxNumCompartments = argsHidden.maxNumCxPerCore

    outputLayer = NxConv2D(
        argsOutput.kernelDepth,
        (argsOutput.kernelHeight, argsOutput.kernelWidth),
        strides=(argsOutput.strideY, argsOutput.strideX),
        padding=argsOutput.padding, vThMant=vThMant,
        synapseEncoding=argsOutput.encoding, kernel_initializer='ones',
        validatePartitions=True)

    outputLayer._maxNumCompartments = argsOutput.maxNumCxPerCore

    hiddenShape = hiddenLayer.compute_output_shape(batchInputShape)
    outputShape = outputLayer.compute_output_shape(hiddenShape)

    hiddenShape = hiddenShape[1:]
    outputShape = outputShape[1:]

    if np.any(np.array(hiddenShape) <= 0) or \
            np.any(np.array(outputShape) <= 0):
        print("Configuration resulted in negative layer shape; skip.")
        print("Input layer shape: {}".format(inputShape))
        print("Hidden layer shape: {}".format(hiddenShape))
        print("Output layer shape: {}".format(outputShape))
        return

    # Create NxModel from input to output layer.
    snn = NxModel(inputLayer.input,
                  outputLayer(hiddenLayer(inputLayer.input)),
                  numCandidatesToCompute=1)
    snn.compileModel()

    # Create plain Keras model from input to hidden layer so we
    # can read out hidden layer activations.
    model1 = Model(inputLayer.input, hiddenLayer(inputLayer.input))
    hiddenInput = Input(hiddenShape)
    model2 = Model(hiddenInput, outputLayer(hiddenInput))

    # Define probes to read out currents.
    # ###################################

    u = ProbableStates.CURRENT
    v = ProbableStates.VOLTAGE

    uProbes1 = []
    vProbes1 = []
    for i in range(int(np.asscalar(np.prod(hiddenShape)))):
        uProbes1.append(hiddenLayer[i].probe(u))
        if plotUV:
            vProbes1.append(hiddenLayer[i].probe(v))

    uProbes2 = []
    vProbes2 = []
    for i in range(int(np.asscalar(np.prod(outputShape)))):
        uProbes2.append(outputLayer[i].probe(u))
        if plotUV:
            vProbes2.append(outputLayer[i].probe(v))

    # Apply input image via bias currents.
    # ####################################

    inputImage = inputScale * np.ones(inputShape, int)
    for i, biasMant in enumerate(np.ravel(inputImage, 'F')):
        inputLayer[i].biasMant = biasMant
        inputLayer[i].phase = 2

    # Run model.
    # ##########

    snn.run(numSteps)
    snn.disconnect()

    # Get SNN spike counts.
    data1 = extract(uProbes1)
    spikecount1 = _data_to_img(data1 // (vThMant * 2**6), hiddenShape)

    data2 = extract(uProbes2)
    spikecount2 = _data_to_img(data2 // (vThMant * 2**6), outputShape)

    # Get ANN activations as target reference.
    batchInputImage = np.expand_dims(inputImage, 0)
    activations1 = model1.predict(batchInputImage)
    activations2 = model2.predict(activations1 > 0)

    # Remove batch dim.
    activations1 = activations1[0].astype(int)
    activations2 = activations2[0].astype(int)

    # Plotting.
    # #########

    if plotUV:
        plt.figure()
        _plot_stimulus_response(vProbes1, uProbes1)
        plt.show()

        plt.figure()
        _plot_stimulus_response(vProbes2, uProbes2)
        plt.show()

    # Validation.
    # ###########

    def getErrorMessage(spikecounts, activations):
        """Generate an error message for failed test.

        The error message contains a code snippet to reproduce failure.

        :param np.ndarray spikecounts: SNN spike counts of layer.
        :param np.ndarray activations: ANN activations of layer.
        :return: Error message.
        :rtype: str
        """

        return (
            "SNN spikecounts not equal to ANN activations.\n" +
            "Spikecounts: \n{}\n\n".format(spikecounts) +
            "Activations: \n{}\n\n".format(activations) +
            "Use the following code to reproduce the error.\n\n" +
            "def test_correlation{}():\n".format('' if n is None else n) +
            "\targsInput = {}\n".format(argsInput) +
            "\targsHidden = {}\n".format(argsHidden) +
            "\targsOutput = {}\n".format(argsOutput) +
            "\trunModelFromConfig(argsInput, argsHidden, argsOutput)\n\n")

    assert np.array_equal(spikecount1, activations1), \
        getErrorMessage(spikecount1, activations1)
    assert np.array_equal(spikecount2, activations2), \
        getErrorMessage(spikecount2, activations2)


def _plot_stimulus_response(vProbes, uProbes):
    """Plot voltage and current probes.

    :param list vProbes: Voltage probe.
    :param list uProbes: Current probe.
    """

    plt.subplot(2, 1, 1)
    for p in vProbes:
        p.plot(label="chipId={}, coreId={}, cxId={}".format(
            p.chipId, p.coreId, p.id))
    plt.ylabel("vProbes")
    # plt.legend(loc='center left')

    plt.subplot(2, 1, 2)
    for p in uProbes:
        plt.plot(p.data, linestyle="None", marker="o",
                 label="chipId={}, coreId={}, cxId={}".format(
                    p.chipId, p.coreId, p.id))
    plt.ylabel("uProbes")
    # plt.legend(loc='center left')


def _data_to_img(data, shape):
    """Sum probe data across time and reshape flat probe data into image shape.

    Reshaping is done in Fortran style because that is how the probes were
    created by the partitioner.

    :param np.ndarray data:
    :param list | tuple | np.ndarray shape: Image shape to transform data to.
    :return: The probe data summed over time and reshaped to image shape.
    :rtype: np.ndarray
    """

    return np.reshape(np.sum(data, 0), shape, 'F')


def runCorrelationRandom(layer, vThMant, insertFlatten=False, inputShape=None,
                         logger=None):
    """Run network to test correlation between ANN and SNN.

    :param NxLayer | Layer layer: NxLayer to test.
    :param int vThMant: Threshold of ``layer``; used to scale activations.
    :param bool insertFlatten: Whether to flatten input before applying it to
        ``layer``.
    :param np.ndarray | tuple | list inputShape: Shape of input to the network.
    :param logging.Logger logger: Logger.

    :return: Pearson correlation coefficient of ANN activations and SNN rates.
    :rtype: float
    """

    seed = 123
    np.random.seed(seed)

    visualizePartitions = False
    plotUV = False

    if inputShape is None:
        inputShape = (7, 7, 1)
    numInputNeurons = int(np.asscalar(np.prod(inputShape)))
    inputScale = numInputNeurons - 1

    thrToInputRatio = 2 ** 7
    thrGain = 2 ** 0

    vThMantInput = thrToInputRatio * inputScale // thrGain

    maxNumSpikes = 100
    numSteps = thrToInputRatio * maxNumSpikes

    inputImage = np.random.randint(0, inputScale, inputShape)

    inputLayer = NxInputLayer(batch_input_shape=(1,) + inputShape,
                              vThMant=vThMantInput,
                              visualizePartitions=visualizePartitions)

    out = layer(NxFlatten()(inputLayer.input)) \
        if insertFlatten else layer(inputLayer.input)

    model = NxModel(inputLayer.input, out, logger=logger)

    model.compileModel()

    outputShape = layer.output_shape[1:]

    # Define probes to read out currents.
    vProbes0 = []
    for i in range(numInputNeurons):
        vProbes0.append(inputLayer[i].probe(ProbableStates.VOLTAGE))

    vProbes = []
    sProbes = []
    for i in range(int(np.asscalar(np.prod(outputShape)))):
        vProbes.append(layer[i].probe(ProbableStates.VOLTAGE))
        sProbes.append(layer[i].probe(ProbableStates.ACTIVITY))

    # Set bias currents
    for i, b in enumerate(np.ravel(inputImage, 'F')):
        inputLayer[i].biasMant = b
        inputLayer[i].phase = 2

    model.run(numSteps)
    model.disconnect()

    data = extract(sProbes)
    spikecount = _data_to_img(data // 127, outputShape)
    spikerates = spikecount / numSteps * thrToInputRatio

    batchInputImage = np.expand_dims(inputImage, 0)
    activations = model.predict(batchInputImage)[0] / (vThMant * thrGain)

    if plotUV:
        plt.figure(1)
        _plot_stimulus_response(vProbes0, [])
        plt.show()

        plt.figure(2)
        _plot_stimulus_response(vProbes, sProbes)
        plt.show()

        plt.figure(3)
        plt.imshow(normalize_image_dims(inputImage))
        plt.show()

        plt.figure(6)
        plt.plot(activations.flatten(), spikerates.flatten(), '.')
        plt.show()

        plt.figure(7)
        plt.imshow(normalize_image_dims(activations))
        plt.show()

        plt.figure(8)
        plt.imshow(normalize_image_dims(spikerates))
        plt.show()

    return np.corrcoef(np.ravel(spikerates), np.ravel(activations))[0, 1]


def normalize_image_dims(image):
    """Add or remove dimensions from ``image`` to make it 2D.

    :param np.ndarray image: Image to transform.

    :return: Transformed image.
    :rtype: np.ndarray
    """

    if image.ndim == 1:
        return np.expand_dims(image, 1)
    if image.ndim == 3:
        return image[:, :, 0]
    return image


def kernel_initializer(shape, dtype=None, kernelScale=1):
    """Random integer kernel initializer for Keras layer.

    :param list | tuple | np.ndarray shape: Shape of kernel.
    :param str | type | None dtype: Data type of kernel.
    :param int kernelScale: Scale factor applied to the kernel.

    :return: Kernel tensor.
    """

    # Start with a small negative weight so we test using both signs. Could
    # center weight distribution around 0, but that would lead to low rates and
    # we'd have to run for more timesteps to get good correlation.
    kernel = np.random.randint(-1, kernelScale, shape)
    return k.constant(kernel, dtype)


def main():
    support.run_unittest(TestDnnCompiler)


if __name__ == '__main__':
    main()
