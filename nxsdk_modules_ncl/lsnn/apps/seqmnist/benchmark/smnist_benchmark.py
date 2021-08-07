# Copyright © 2018-2021 Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#   * Neither the name of Intel Corporation nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
import matplotlib.pyplot as plt
import six
from scipy.optimize import curve_fit

from nxsdk_modules.lsnn.datasets.mnist_dataset import loadMNIST
from nxsdk.utils.env_var_context_manager import setEnvWithinContext
from nxsdk_modules.lsnn.src.lsnn import LsnnNet, ModelParams, CompileParams
import nxsdk.api.n2a as nx

from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition


def loadMnistData(trainOrTestData='test'):
    """Loads MNIST data from sklearn or web.

    :param str trainOrTestData: Must be 'train' or 'test' and specifies which \
    part of the MNIST dataset to load.
    :return: images, targets
    """

    mnist = loadMNIST()
    if trainOrTestData == 'train':
        X = mnist.data[:60000, :].astype(np.uint8)
        y = mnist.target[:60000].astype(np.uint8)
    elif trainOrTestData == 'test':
        X = mnist.data[60000:, :].astype(np.uint8)
        y = mnist.target[60000:].astype(np.uint8)
    else:
        raise ValueError("trainOrTestData must be 'train' or 'test'.")

    return X, y


def loadWeightsAndDelays(dataDir, doLoadDelays=False):
    """Loads input, recurrent and output layer weight matrices and optionally \
    input and recurrent layer delay matrices. All matrices are expected to be \
    numpy arrays.

    :param str dataDir: Directory containing weight (and optional delay \
    matrices).
    :param bool doLoadDelays: If True, also expects delay matrices to be \
    present in directory and loads them as well.
    :return: (wIn, wRec, wOut) or (wIn, wRec, wOut, dIn, dRec) tuples \
    containing numpy weight and delay matrices. All matrices have shape \
    (dstDim, srcDim).
    :rtype: numpy.ndarray
    """

    # Load weights
    path = os.path.join(dataDir, 'w_in.npy')
    wIn = np.load(path)

    path = os.path.join(dataDir, 'w_rec.npy')
    wRec = np.load(path)

    path = os.path.join(dataDir, 'w_out.npy')
    wOut = np.load(path)

    out = (wIn.T, wRec.T, wOut.T)

    # Load delays
    if doLoadDelays:
        path = os.path.join(dataDir, 'delayInArray.npy')
        dIn = np.load(path)

        path = os.path.join(dataDir, 'delayRecArray.npy')
        dRec = np.load(path)

        out = (*out, dIn.T, dRec.T)

    return out


class SequentialImageClassifierLsnn_bench(LsnnNet):
    """Sequential image classifier class for benchmarking (can copy network to\
    multiple cores)
    """

    def __init__(self, wIn, wRec, wOut, numInput=80, numRegular=140,
                 numAdaptive=100, numOutput=10, cueDuration=56, imageSize=784,
                 batchSize=10, numCores=1):
        """Initializes the SequentialImageClassifier.

        :param array2d wIn: Input weight matrix used to connect input layer to\
                            recurrent layer
        :param array2d wRec: Recurrent weight matrix used to connect recurrent\
                             layer to itself
        :param array2d wOut: Output weight matrix used to connect recurrent\
                             layer to output layer
        :param int numInput: Number of input neurons or ports
        :param int numRegular: Number of regular neurons
        :param int numAdaptive: Number of adaptive neurons
        :param int numOutput: Number of output neurons
        :param int cueDuration: Duration in which the network decides the\
        classification. One dedicated input neuron is active during this cue.
        :param int imageSize: Size of the image to classify - determines the\
        time steps
        :param int batchSize: Amount of images which are transferred at once\
        to the LMT (input via SNIPs only)
        """

        self._numCores = numCores
        self.batchSize = batchSize
        self.cueDuration = cueDuration
        self.imageSize = imageSize
        self._numInput = numInput
        self._numOutput = numOutput
        self._inputs = None
        self._targets = None
        self._classifications = None
        self._snipsDir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'snips')

        # define parameters for LsnnNet
        self._modelP = ModelParams(numRegular, numAdaptive, wIn, wRec,
                                   wOut)
        self._modelP.thrAdaScale = 4
        self._modelP._numDelayBits = 4
        self._compileP = CompileParams()
        self._compileP.logicalCoreId = 0
        # need to create NxNet object to set up spike input ports used for SNIPs
        self._compileP.net = nx.NxNet()
        self._inputPortGroup = \
            self._compileP.net.createSpikeInputPortGroup(size=numInput)
        # configure output neuron group
        outputGroup = self._createOutputNeuronGroup(numOutput, logicalCoreId=0)

        # create input distribution layer
        self.inputDist = self._createInputNeuronGroup(numInput,
                                                      logicalCoreId=numCores)

        self._inputPortGroup.connect(
            dstGrp=self.inputDist,
            prototype=nx.ConnectionPrototype(weight=200),
            connectionMask=np.eye(numInput))

        # LSNN
        super().__init__(self.inputDist, outputGroup, self._modelP,
                         self._compileP)

        # create additional LSNNs # numCores and inputs etc.
        for i in range(1, numCores):
            compileP = CompileParams()
            compileP.logicalCoreId = i
            # use the same NxNet object
            compileP.net = self._compileP.net
            outputGroupCopy = self._createOutputNeuronGroup(numOutput,
                                                            logicalCoreId=i)
            lsnn = LsnnNet(self.inputDist, outputGroupCopy, self._modelP,
                           compileP)

        # Compile and setup snips and channels
        self.generateNetwork()

    # --------------------------------------------------------------------------
    # Helper
    @property
    def snipsDir(self):
        """Gives the path to the folder with the SNIP files"""
        return self._snipsDir

    @snipsDir.setter
    def snipsDir(self, val):
        """Gives the path to the folder with the SNIP files"""
        assert isinstance(val, str), "Directory must be of type string."
        self._snipsDir = val

    @property
    def _resourceMap(self):
        """The resourceMap of the network"""
        return self.compileParams.net.resourceMap

    def _getInputAxonMap(self):
        """Determines the port ids of the input axons (input via SNIPs)"""
        portIdToAxonsMap = {}
        for i, inputPort in enumerate(self._inputPortGroup):
            for inputAxon in inputPort.getAxon():
                axId = inputAxon.nodeId
                if i not in portIdToAxonsMap:
                    portIdToAxonsMap[i] = []
                portIdToAxonsMap[i] = portIdToAxonsMap[i] + \
                                      self._resourceMap.inputAxon(axId)

        for i in range(self._numInput):
            if i not in portIdToAxonsMap:
                portIdToAxonsMap[i] = [(-1, -1, -1, -1)]

        return portIdToAxonsMap

    def _getOutputNeuronMap(self):
        """Determines the portId of the output neurons (output via SNIPs)"""
        outMap = []
        for i in range(self._numOutput):
            cId = self.outputGroup[i].nodeId
            mapping = self._resourceMap.compartmentMap[cId]
            outMap.append([int(mapping.boardId), int(mapping.chipId),
                           int(mapping.coreId), int(mapping.cxId)])

        return outMap

    # --------------------------------------------------------------------------
    # Snip setup
    def _configureInitSnip(self):
        """Creates the process for the initialization SNIP channel"""
        self.initProcess = self.board.createProcess(
            name="initProcess",
            cFilePath=os.path.join(self.snipsDir, "initsnip.c"),
            includeDir=self.snipsDir,
            funcName="init_snip",
            guardName=None,
            phase="init")

    def _configureSpikingSnip(self):
        """Creates the process for the spiking SNIP channel"""
        self.spikingProcess = self.board.createProcess(
            name="spikingProcess",
            cFilePath=os.path.join(self.snipsDir, "spiking.c"),
            includeDir=self.snipsDir,
            funcName="run_spiking",
            guardName="do_spiking",
            phase="spiking")

    def _configureManagementSnip(self):
        """Creates the process for the management SNIP channel"""
        self.managementProcess = self.board.createProcess(
            name="resetSnip",
            cFilePath=os.path.join(self.snipsDir, "snip_mgmt.c"),
            includeDir=self.snipsDir,
            funcName="snip_mgmt",
            guardName="do_mgmt",
            phase="mgmt")

    def _configureSNIPs(self):
        """Configures init, spiking and management SNIPs and their channels"""
        self._configureInitSnip()
        self._configureSpikingSnip()
        self._configureManagementSnip()
        self._createChannels()

    # --------------------------------------------------------------------------
    # Channel setup
    def _setupInputParamsChannel(self):
        """Creates the channel for the input parameters (number of inputs,\
        image size and batch size)
        """
        self.inputParamsChannel = self.board.createChannel(
            b'nxinit_input_params', "int", 4)
        self.inputParamsChannel.connect(None, self.initProcess)

    def _setupInputPortsChannel(self):
        """Creates the channel for the input ports"""
        self.inputPortsChannel = self.board.createChannel(
            b'nxinit_input_ports', "int", 4 * self._numInput)
        self.inputPortsChannel.connect(None, self.initProcess)

    def _setupThresholdsChannel(self):
        """Creates the channel for the thresholds (threshold crossing spike\
        generation)
        """
        self.thresholdsChannel = self.board.createChannel(
            b'nxinit_thresholds', "int", self._numInput // 2)
        self.thresholdsChannel.connect(None, self.initProcess)

    def _setupOutputNeuronChannel(self):
        """Creates the channel for the output neuron ports"""
        self.outputNeuronChannel = self.board.createChannel(
            b'output_neurons', "int", 4 * self._numOutput)
        self.outputNeuronChannel.connect(None, self.initProcess)

    def _setupInputDataChannel(self):
        """Creates the channel for the input images"""
        self.inputDataChannel = self.board.createChannel(
            b'nxspk_img_data', "packed",
            self.numSamples * self.imageSize * self.batchSize)
        self.inputDataChannel.connect(None, self.spikingProcess)

    def _setupClassificationChannel(self):
        """Creates the channel for the classification results"""
        self.classificationChannel = self.board.createChannel(
            b'classifications', "int", self.batchSize)
        self.classificationChannel.connect(self.managementProcess, None)

    def _createChannels(self):
        """Creates all the channels"""
        self._setupInputParamsChannel()
        self._setupInputPortsChannel()
        self._setupThresholdsChannel()
        self._setupOutputNeuronChannel()
        self._setupInputDataChannel()
        self._setupClassificationChannel()

    # --------------------------------------------------------------------------
    # Execution helper
    def _configureRun(self):
        """Starts NxDriver and sends initialization parameters via different\
        channels to init snip on LMT.
        """

        self.board.startDriver()

        # Send size parameters to LMT
        self.inputParamsChannel.write(4, [self._numInput,
                                          self.imageSize, self.batchSize,
                                          self._numCores])

        # Send threshold values for spike-generator to LMT
        thresholds = np.floor(np.linspace(0, 255, self._numInput // 2)). \
            astype(np.uint8).tolist()
        self.thresholdsChannel.write(self._numInput // 2, thresholds)

        # Send portToAxonMap to init LMT
        portToAxonsMap = self._getInputAxonMap()
        for i in range(self._numInput):
            self.inputPortsChannel.write(4, list(portToAxonsMap[i][0]))

        # Send outputNeuronMap to LMT
        outputNeuronMap = self._getOutputNeuronMap()
        for i in range(self._numOutput):
            self.outputNeuronChannel.write(4, outputNeuronMap[i])

    def _start(self, imgIds):
        """Runs SequentialImageClassifier by sending numSamples amount of\
        images specified by imgIds from self.inputs to the LMT. Calls LsnnNet\
        run() function.

        :param range imgIds: ids of the images in self.inputs to process
        """

        # helpers to display a progress bar during execution
        count = int(self.numSamples / self.batchSize)
        print_update = int(np.ceil(count * 0.1))  # update print every 10%

        # write the images to the channel (packed mode)
        writeData = []
        for i in imgIds:
            img = self.inputs[i]
            img = np.reshape(img, np.product(img.shape))
            img = img.astype(np.uint8).tolist()
            writeData.extend(img)

        self.inputDataChannel.write(len(writeData), writeData)

        # Execute network in non-blocking aSync mode
        runtimePerImg = self.imageSize + self.cueDuration
        self.run(runtimePerImg * self.numSamples, aSync=True)

        # read classification results for each batch and update progress bar
        for k in range(1, count):

            out = range((k - 1) * self.batchSize, min(k * self.batchSize,
                                                      self.numSamples))

            # Read classification result from LMT at end of batch
            c = self.classificationChannel.read(
                self.batchSize)  # packed mode (16 integers)
            self._classifications[np.asarray(out)] = c  # [0:self.batchSize]

            if k % print_update == 0:
                v = int(k / print_update)
                print("\r[%-10s] %d%%" % ('=' * v, 10 * v), end='')

        print("\r[%-10s] %d%%" % ('=' * 10, 10 * 10))

        out = range((count - 1) * self.batchSize, self.numSamples)
        # Read classification result from LMT at end of batch
        c = self.classificationChannel.read(self.batchSize)

        self._classifications[np.asarray(out)] = c  # [0:self.batchSize]

    # --------------------------------------------------------------------------
    # Interface
    @property
    def numSamples(self):
        """Amount of images to process"""
        return len(self._targets)

    @property
    def inputs(self):
        """Input images"""
        assert self._inputs is not None, 'inputs not set.'
        return self._inputs

    @property
    def targets(self):
        """Target classes for classification"""
        assert self._targets is not None, 'targets not set.'
        return self._targets

    @property
    def classifications(self):
        assert self._classifications is not None, 'classifications not set.'
        return self._classifications[range(0, self.numSamples)]

    def classify(self, inputs, targets):
        """Classifies the given inputs (images) by comparing the processing\
        results of the images to the target classes targets.

        :param array2d inputs: array of arrays of pixel values representing an\
        image
        :param np.array targets: array of the target classes
        """
        assert inputs.shape[0] == targets.shape[0], \
            'Different number of inputs and targets provided. inputs.shape[0] ' \
            '' \
            'and targets.shape[0] must be identical.'
        assert inputs.shape[1] == self.imageSize, \
            'inputs have incorrect size. inputs.shape[1] must be %d.' % (
                self.imageSize)

        self._inputs = inputs
        self._targets = targets
        self._classifications = np.zeros(self._targets.shape, dtype=int)

        self._configureSNIPs()
        self._configureRun()

        # Compute number of batches
        assert self.numSamples % self.batchSize == 0, 'numSamples/targets ' \
                                                      'must be multiple of ' \
                                                      'batchSize.'

        imgIds = range(0, self.numSamples)
        self._start(imgIds)

        self.finish()

    def printClassification(self, showClasses=False):
        """Prints the classification accuracy and optional the targets and the\
        classification results.

        :param bool showClasses: if True, shows the target and classification\
        labels for each sample, additionally to the classification accuracy.
        """

        yt = np.asarray(self._targets[range(0, self.numSamples)])
        y = np.asarray(self._classifications[range(0, self.numSamples)])

        if showClasses:
            print('Targets:')
            print(yt)
            print('Classifications:')
            print(y)

        dy = yt - y
        numCorrect = int(np.sum(dy == 0))
        print('Classification accuracy of %d samples: %0.2f%%' % (
            self.numSamples, numCorrect / len(y) * 100))

    def _createOutputNeuronGroup(self, numOutput, logicalCoreId=-1):
        """Configures an generic output neuron group which does not spike.

        :param int numOutput: number of output neurons
        """
        outMantissa = 2 ** 17 - 1
        cxProto = nx.CompartmentPrototype(
            logicalCoreId=logicalCoreId,
            vThMant=outMantissa,
            compartmentCurrentDecay=self._modelP.decayU,
            compartmentVoltageDecay=self._modelP.decayV,
            refractoryDelay=self._modelP.refrac,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            numDendriticAccumulators=16
        )
        return self._compileP.net.createCompartmentGroup(
            size=numOutput, prototype=cxProto)

    def _createInputNeuronGroup(self, numInput, logicalCoreId=-1):
        """creates an input group on a given core."""
        inMantissa = 100
        cxProto = nx.CompartmentPrototype(logicalCoreId=logicalCoreId,
                                          vThMant=inMantissa,
                                          compartmentCurrentDecay=self._modelP.decayU,
                                          compartmentVoltageDecay=self._modelP.decayU,
                                          refractoryDelay=1,
                                          functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                          numDendriticAccumulators=16
                                          )
        return self._compileP.net.createCompartmentGroup(
            size=numInput, prototype=cxProto)


class SequentialImageClassifierLsnn_bench_breakdown(LsnnNet):
    """Sequential image classifier class for benchmarking (special SNIPs\
    measure breakdown of spiking and mngmt phase)
    """

    def __init__(self, wIn, wRec, wOut, numInput=80, numRegular=140,
                 numAdaptive=100, numOutput=10, cueDuration=56, imageSize=784,
                 batchSize=10):
        """Initializes the SequentialImageClassifier.

        :param array2d wIn: Input weight matrix used to connect input layer to\
                            recurrent layer
        :param array2d wRec: Recurrent weight matrix used to connect recurrent\
                             layer to itself
        :param array2d wOut: Output weight matrix used to connect recurrent\
                             layer to output layer
        :param int numInput: Number of input neurons or ports
        :param int numRegular: Number of regular neurons
        :param int numAdaptive: Number of adaptive neurons
        :param int numOutput: Number of output neurons
        :param int cueDuration: Duration in which the network decides the\
        classification. One dedicated input neuron is active during this cue.
        :param int imageSize: Size of the image to classify - determines the\
        time steps
        :param int batchSize: Amount of images which are transferred at once\
        to the LMT (input via SNIPs only)
        """
        # Initialize parameters
        self.batchSize = batchSize
        self.cueDuration = cueDuration
        self.imageSize = imageSize
        self._numInput = numInput
        self._numOutput = numOutput
        self._inputs = None
        self._targets = None
        self._classifications = None
        self._snipsDir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'snips')

        # define parameters for LsnnNet
        self._modelP = ModelParams(numRegular, numAdaptive, wIn, wRec,
                                   wOut)
        self._compileP = CompileParams()
        # need to create NxNet object to set up spike input ports used for SNIPs
        self._compileP.net = nx.NxNet()
        inputGroup = \
            self._compileP.net.createSpikeInputPortGroup(size=numInput)
        # configure output neuron group
        outputGroup = self._createOutputNeuronGroup(numOutput)

        # LSNN
        super().__init__(inputGroup, outputGroup, self._modelP,
                         self._compileP)

        # Compile and setup snips and channels
        self.generateNetwork()

    # --------------------------------------------------------------------------
    # Helper
    @property
    def snipsDir(self):
        """Gives the path to the folder with the SNIP files"""
        return self._snipsDir

    @snipsDir.setter
    def snipsDir(self, val):
        """Gives the path to the folder with the SNIP files"""
        assert isinstance(val, str), "Directory must be of type string."
        self._snipsDir = val

    @property
    def _resourceMap(self):
        """The resourceMap of the network"""
        return self.compileParams.net.resourceMap

    def _getInputAxonMap(self):
        """Determines the port ids of the input axons (input via SNIPs)"""
        portIdToAxonsMap = {}
        for i, inputPort in enumerate(self.inputGroup):
            for inputAxon in inputPort.getAxon():
                axId = inputAxon.nodeId
                if i not in portIdToAxonsMap:
                    portIdToAxonsMap[i] = []
                portIdToAxonsMap[i] = portIdToAxonsMap[i] + \
                                      self._resourceMap.inputAxon(axId)

        for i in range(self._numInput):
            if i not in portIdToAxonsMap:
                portIdToAxonsMap[i] = [(-1, -1, -1, -1)]

        return portIdToAxonsMap

    def _getOutputNeuronMap(self):
        """Determines the portId of the output neurons (output via SNIPs)"""
        outMap = []
        for i in range(self._numOutput):
            cId = self.outputGroup[i].nodeId
            mapping = self._resourceMap.compartmentMap[cId]
            outMap.append([int(mapping.boardId), int(mapping.chipId),
                           int(mapping.coreId), int(mapping.cxId)])

        return outMap

    # --------------------------------------------------------------------------
    # Snip setup
    def _configureInitSnip(self):
        """Creates the process for the initialization SNIP channel"""
        self.initProcess = self.board.createProcess(
            name="initProcess",
            cFilePath=os.path.join(self.snipsDir, "initsnip.c"),
            includeDir=self.snipsDir,
            funcName="init_snip",
            guardName=None,
            phase="init")

    def _configureSpikingSnip(self):
        """Creates the process for the spiking SNIP channel"""
        self.spikingProcess = self.board.createProcess(
            name="spikingProcess",
            cFilePath=os.path.join(self.snipsDir, "spiking_bench.c"),
            includeDir=self.snipsDir,
            funcName="run_spiking",
            guardName="do_spiking",
            phase="spiking")

    def _configureManagementSnip(self):
        """Creates the process for the management SNIP channel"""
        self.managementProcess = self.board.createProcess(
            name="resetSnip",
            cFilePath=os.path.join(self.snipsDir, "snip_mgmt_bench.c"),
            includeDir=self.snipsDir,
            funcName="snip_mgmt",
            guardName="do_mgmt",
            phase="mgmt")

    def _configureSNIPs(self):
        """Configures init, spiking and management SNIPs and their channels"""
        self._configureInitSnip()
        self._configureSpikingSnip()
        self._configureManagementSnip()
        self._createChannels()

    # --------------------------------------------------------------------------
    # Channel setup
    def _setupInputParamsChannel(self):
        """Creates the channel for the input parameters (number of inputs,\
        image size and batch size)
        """
        self.inputParamsChannel = self.board.createChannel(
            b'nxinit_input_params', "int", 4)
        self.inputParamsChannel.connect(None, self.initProcess)

    def _setupInputPortsChannel(self):
        """Creates the channel for the input ports"""
        self.inputPortsChannel = self.board.createChannel(
            b'nxinit_input_ports', "int", 4 * self._numInput)
        self.inputPortsChannel.connect(None, self.initProcess)

    def _setupThresholdsChannel(self):
        """Creates the channel for the thresholds (threshold crossing spike\
        generation)
        """
        self.thresholdsChannel = self.board.createChannel(
            b'nxinit_thresholds', "int", self._numInput // 2)
        self.thresholdsChannel.connect(None, self.initProcess)

    def _setupOutputNeuronChannel(self):
        """Creates the channel for the output neuron ports"""
        self.outputNeuronChannel = self.board.createChannel(
            b'output_neurons', "int", 4 * self._numOutput)
        self.outputNeuronChannel.connect(None, self.initProcess)

    def _setupInputDataChannel(self):
        """Creates the channel for the input images"""
        self.inputDataChannel = self.board.createChannel(
            b'nxspk_img_data', "packed",
            self.numSamples * self.imageSize * self.batchSize)
        self.inputDataChannel.connect(None, self.spikingProcess)

    def _setupClassificationChannel(self):
        """Creates the channel for the classification results"""
        self.classificationChannel = self.board.createChannel(
            b'classifications', "int", self.batchSize)
        self.classificationChannel.connect(self.managementProcess, None)

    def _setupMeasurementSpikingChannel(self):
        """Creates the channel for the measurements_spk results"""
        self.measurements_spkChannel = self.board.createChannel(
            b'measurements_spk', "packed", 16)
        self.measurements_spkChannel.connect(self.spikingProcess, None)

    def _setupMeasurementMgmtChannel(self):
        """Creates the channel for the measurements_mgmt results"""
        self.measurements_mgmtChannel = self.board.createChannel(
            b'measurements_mgmt', "packed", 16)
        self.measurements_mgmtChannel.connect(self.spikingProcess, None)

    def _createChannels(self):
        """Creates all the channels"""
        self._setupInputParamsChannel()
        self._setupInputPortsChannel()
        self._setupThresholdsChannel()
        self._setupOutputNeuronChannel()
        self._setupInputDataChannel()
        self._setupClassificationChannel()
        self._setupMeasurementSpikingChannel()
        self._setupMeasurementMgmtChannel()

    # --------------------------------------------------------------------------
    # Execution helper
    def _configureRun(self):
        """Starts NxDriver and sends initialization parameters via different\
        channels to init snip on LMT.
        """

        self.board.startDriver()

        # Send size parameters to LMT
        self.inputParamsChannel.write(4, [self._numInput,
                                          self.imageSize, self.batchSize, 1])

        # Send threshold values for spike-generator to LMT
        thresholds = np.floor(np.linspace(0, 255, self._numInput // 2)). \
            astype(np.uint8).tolist()
        self.thresholdsChannel.write(self._numInput // 2, thresholds)

        # Send portToAxonMap to init LMT
        portToAxonsMap = self._getInputAxonMap()
        for i in range(self._numInput):
            self.inputPortsChannel.write(4, list(portToAxonsMap[i][0]))

        # Send outputNeuronMap to LMT
        outputNeuronMap = self._getOutputNeuronMap()
        for i in range(self._numOutput):
            self.outputNeuronChannel.write(4, outputNeuronMap[i])

    def _start(self, imgIds):
        """Runs SequentialImageClassifier by sending numSamples amount of\
        images specified by imgIds from self.inputs to the LMT. Calls LsnnNet\
        run() function.

        :param range imgIds: ids of the images in self.inputs to process
        """

        # helpers to display a progress bar during execution
        count = int(self.numSamples / self.batchSize)
        print_update = int(np.ceil(count * 0.1))  # update print every 10%

        # write the images to the channel (packed mode)
        writeData = []
        for i in imgIds:
            img = self.inputs[i]
            img = np.reshape(img, np.product(img.shape))
            img = img.astype(np.uint8).tolist()
            writeData.extend(img)

        self.inputDataChannel.write(len(writeData), writeData)

        # Execute network in non-blocking aSync mode
        runtimePerImg = self.imageSize + self.cueDuration
        self.run(runtimePerImg * self.numSamples, aSync=True)

        # read classification results for each batch and update progress bar
        for k in range(1, count):

            out = range((k - 1) * self.batchSize, min(k * self.batchSize,
                                                      self.numSamples))

            # Read classification result from LMT at end of batch
            c = self.classificationChannel.read(self.batchSize)
            self._classifications[np.asarray(out)] = c  # [0:self.batchSize]

            if k % print_update == 0:
                v = int(k / print_update)
                print("\r[%-10s] %d%%" % ('=' * v, 10 * v), end='')

        print("\r[%-10s] %d%%" % ('=' * 10, 10 * 10))

        out = range((count - 1) * self.batchSize, self.numSamples)
        # Read classification result from LMT at end of batch
        c = self.classificationChannel.read(self.batchSize)

        self._classifications[np.asarray(out)] = c  # [0:self.batchSize]

        self.measures_spk = self.measurements_spkChannel.read(1)
        self.measures_mgmt = self.measurements_mgmtChannel.read(1)

    # --------------------------------------------------------------------------
    # Interface
    @property
    def numSamples(self):
        """Amount of images to process"""
        return len(self._targets)

    @property
    def inputs(self):
        """Input images"""
        assert self._inputs is not None, 'inputs not set.'
        return self._inputs

    @property
    def targets(self):
        """Target classes for classification"""
        assert self._targets is not None, 'targets not set.'
        return self._targets

    @property
    def classifications(self):
        assert self._classifications is not None, 'classifications not set.'
        return self._classifications[range(0, self.numSamples)]

    def classify(self, inputs, targets):
        """Classifies the given inputs (images) by comparing the processing\
        results of the images to the target classes targets.

        :param array2d inputs: array of arrays of pixel values representing an\
        image
        :param np.array targets: array of the target classes
        """
        assert inputs.shape[0] == targets.shape[0], \
            'Different number of inputs and targets provided. inputs.shape[0] '\
            '' \
            'and targets.shape[0] must be identical.'
        assert inputs.shape[1] == self.imageSize, \
            'inputs have incorrect size. inputs.shape[1] must be %d.' % (
                self.imageSize)

        self._inputs = inputs
        self._targets = targets
        self._classifications = np.zeros(self._targets.shape, dtype=int)

        self._configureSNIPs()
        self._configureRun()

        # Compute number of batches
        assert self.numSamples % self.batchSize == 0, 'numSamples/targets ' \
                                                      'must be multiple of ' \
                                                      'batchSize.'

        imgIds = range(0, self.numSamples)
        import time
        start = time.time()
        self._start(imgIds)
        end = time.time()
        self.pythonTime = (end - start)

        self.finish()

    def printClassification(self, showClasses=False):
        """Prints the classification accuracy and optional the targets and the\
        classification results.

        :param bool showClasses: if True, shows the target and classification\
        labels for each sample, additionally to the classification accuracy.
        """

        yt = np.asarray(self._targets[range(0, self.numSamples)])
        y = np.asarray(self._classifications[range(0, self.numSamples)])

        if showClasses:
            print('Targets:')
            print(yt)
            print('Classifications:')
            print(y)

        dy = yt - y
        numCorrect = int(np.sum(dy == 0))
        print('Classification accuracy of %d samples: %0.2f%%' % (
            self.numSamples, numCorrect / len(y) * 100))

    def _createOutputNeuronGroup(self, numOutput):
        """Configures an generic output neuron group which does not spike.

        :param int numOutput: number of output neurons
        """
        outMantissa = 2 ** 17 - 1
        cxProto = nx.CompartmentPrototype(
            logicalCoreId=self._compileP.logicalCoreId,
            vThMant=outMantissa,
            compartmentCurrentDecay=self._modelP.decayU,
            compartmentVoltageDecay=self._modelP.decayV,
            refractoryDelay=self._modelP.refrac,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            numDendriticAccumulators=16
        )
        return self._compileP.net.createCompartmentGroup(
            size=numOutput, prototype=cxProto)

    def _createInputNeuronGroup(self, numInput, logicalCoreId=-1):
        """creates an input group on a given core."""
        inMantissa = 100
        cxProto = nx.CompartmentPrototype(
                        logicalCoreId=logicalCoreId,
                        vThMant=inMantissa,
                        compartmentCurrentDecay=self._modelP.decayU,
                        compartmentVoltageDecay=self._modelP.decayU,
                        refractoryDelay=1,
                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                        numDendriticAccumulators=16
                        )
        return self._compileP.net.createCompartmentGroup(
            size=numInput, prototype=cxProto)


def runSequentialMnist(wgtDir, numSamples, batchSize, numCores=1):
    """Sets up and executes SequentialImageClassifier LSNN network to\
        classify MNIST images on the test set. Used for benchmarking.

        :param string wgtDir: directory which contains the weight matrices for\
        the network
        :param int numSamples: amount of images which should be processed
        :param int batchSize: batchSize of the SequentialImageClassifierLsnn
        :param int numCores: amount of cores the network should be copied
        """

    # Specify data directory
    dataDir = os.path.join(os.path.dirname(__file__), 'weights', wgtDir)
    # MNIST images are 28 x 28 in size
    imgDx = imgDy = 28

    # Load input, recurrent and output layer weights
    wIn, wRec, wOut = loadWeightsAndDelays(dataDir)

    # Initialize LSSN network
    sqic = SequentialImageClassifierLsnn_bench(wIn=wIn, wRec=wRec, wOut=wOut,
                                               numInput=80, numRegular=140,
                                               numAdaptive=100, numOutput=10,
                                               cueDuration=56,
                                               imageSize=imgDx * imgDy,
                                               batchSize=batchSize,
                                               numCores=numCores)

    sqic.snipsDir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'snips')

    # Load a random set of MNIST test set images and pass to LSNN
    inputs, targets = loadMnistData('test')
    np.random.seed(0)
    imgIdx = np.random.choice(range(0, 10000), numSamples, False)
    inputs, targets = inputs[imgIdx, :], targets[imgIdx]

    eProbe = sqic.board.probe(
        probeType=ProbeParameter.ENERGY,
        probeCondition=PerformanceProbeCondition(
            tStart=1,
            tEnd=numSamples * 840,
            bufferSize=1024,
            binSize=100))

    # Execute network: Generates spikes from images and injects into LSNN
    sqic.classify(inputs, targets)

    return eProbe


def runSequentialMnist_breakdown(wgtDir, numSamples, batchSize):
    """Sets up and executes SequentialImageClassifier LSNN network to\
        classify MNIST images on the test set. Used for benchmarking.

        :param string wgtDir: directory which contains the weight matrices for\
        the network
        :param int numSamples: amount of images which should be processed
        :param int batchSize: batchSize of the SequentialImageClassifierLsnn

        """

    # Specify data directory
    dataDir = os.path.join(os.path.dirname(__file__), 'weights', wgtDir)
    # MNIST images are 28 x 28 in size
    imgDx = imgDy = 28

    # Load input, recurrent and output layer weights
    wIn, wRec, wOut = loadWeightsAndDelays(dataDir)

    # Initialize LSSN network
    sqic = SequentialImageClassifierLsnn_bench_breakdown(wIn=wIn, wRec=wRec,
                                                         wOut=wOut,
                                                         numInput=80,
                                                         numRegular=140,
                                                         numAdaptive=100,
                                                         numOutput=10,
                                                         cueDuration=56,
                                                         imageSize=imgDx *
                                                                   imgDy,
                                                         batchSize=batchSize)

    sqic.snipsDir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'snips')

    # Load a random set of MNIST test set images and pass to LSNN
    inputs, targets = loadMnistData('test')
    np.random.seed(0)
    imgIdx = np.random.choice(range(0, 10000), numSamples, False)
    inputs, targets = inputs[imgIdx, :], targets[imgIdx]

    eProbe = sqic.board.probe(
        probeType=ProbeParameter.ENERGY,
        probeCondition=PerformanceProbeCondition(
            tStart=1,
            tEnd=numSamples * 840,
            bufferSize=1024,
            binSize=100))

    # Execute network: Generates spikes from images and injects into LSNN
    sqic.classify(inputs, targets)

    return eProbe, sqic.measures_spk, sqic.measures_mgmt, sqic.pythonTime


def powerFit(xdata, ydata, plot=False):
    """fits a line to the xdata/ydata based on func.

    :param list xdata: data points on the x axis
    :param list ydata: data points on the y axis
    :param bool plot: optional option to plot
    :return: passive and active parts of the power
    """

    def func(nc, passive, active):
        return passive + (active * nc)

    popt, pcov = curve_fit(func, xdata, ydata,
                           bounds=((0, 0), (np.inf, np.inf)))

    if plot:
        ydata_fit = func(xdata, popt[0], popt[1])

        plt.figure()
        plt.plot(xdata, ydata, 'b-', label='measured')
        plt.plot(xdata, ydata_fit, 'r--', label='fitted')
        plt.title("Power")
        plt.xlabel("# NeuroCores")
        plt.ylabel("Power in mW")
        plt.legend()
        plt.show()

    return popt[0], popt[1]


class ResultValues:
    """This class stores the benchmark results"""

    def __init__(self):
        """Initializes the parameters."""
        # derived static power (LMT, leakage) in mW
        self.static_power = None
        # derived active power (neuro cores) in mW
        self.active_power = None
        # time of image transfer in ms
        self.spk_transfer = None
        # time of spike generation and injection in ms
        self.spk_gen_inj = None
        # time of spike processing in neuro cores in ms
        self.spk_process = None
        # time of classification in ms
        self.mgmt_classification = None
        # time of reset the neuro cores in ms
        self.mgmt_reset = None
        # execution time per sample (calc) in ms
        self.time_sample_calc = None
        # execution time per sample (probe) in ms
        self.time_sample_probe = None
        # execution time per sample (python time measurements - control) in ms
        self.time_sample_python = None
        # time of image transfer per time step in µs
        self.spk_transfer_ts = None
        # time of spike generation and injection per time step in µs
        self.spk_gen_inj_ts = None
        # time of spike processing in neuro cores in ms per time step in µs
        self.spk_process_ts = None
        # time of classification per time step in µs
        self.mgmt_classification_ts = None
        # time of reset the neuro cores per time step in µs
        self.mgmt_reset_ts = None
        # execution time per time step in µs
        self.time_sample_ts = None
        # total power during inference in W
        self.total_power = None
        # total energy per inference in mJ
        self.energy_sample = None
        # static energy per inference in mJ
        self.static_energy_sample = None
        # active energy per inference in mJ
        self.active_energy_sample = None
        # partition
        self.partition = None


def plotResultTable(results):
    """Creates a pyplot table with the results

    :param ResultValue results: filled ResultValue objets
    """

    columns = ["System", "", "Execution time", "Average time per", "Power per",
               "Energy per"]
    cell_text = [
        ["", "", "per inference (ms)", "time step (µs)", "inference (W)",
         "inference (mJ)"],
        ["", "Total:", results.time_sample_calc, results.time_sample_ts, "",
         ""],
        ["", "SPK: Image transfer", results.spk_transfer,
         results.spk_transfer_ts, str(results.total_power) + "*",
         str(results.energy_sample) + "*"],
        ["", "SPK: Spike gen. & inj.", results.spk_gen_inj,
         results.spk_gen_inj_ts, "", ""],
        [results.partition, "SPK: Spike processing", results.spk_process,
         results.spk_process_ts,
         ("static: " + str(results.static_power) + " mW"),
         ("static: " + str(results.static_energy_sample))],
        ["", "MGMT: Classification", results.mgmt_classification,
         results.mgmt_classification_ts,
         ("active: " + str(results.active_power) + " mW"),
         ("active: " + str(results.active_energy_sample))],
        ["", "MGMT: State reset", results.mgmt_reset, results.mgmt_reset_ts, "",
         ""]
        ]

    size = (5 + np.array([0, 1])) * np.array([3.0, 0.4])
    fig, ax = plt.subplots(figsize=size)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, bbox=[0, 0, 1, 1],
                     cellLoc="center")

    header_rows = 2
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for k, cell in six.iteritems(table._cells):
        if k[1] == 0 or k[0] < header_rows:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor("#000000")

def performBenchmark(wgtDirName, numSamples=100, partitions=["loihi"], numCores=[1]):
    """Performs a extensive benchmark, scaling over "numCores" Cores and using
    "partitions" board.
    
    In order to give a good estimate of the power and energy consumption, this\
    example is copied to multiple neuro cores and the measured energy is\
    extrapolated within these measurement points. This is needed, because this\
    examle only needs one neuro core and the idle power of the other neuro\
    cores of the board cover the power usage of this example on one neuro core.

    :param int numSamples: amount of images to process
    :param string wgtDirName: path for directory with the weights
    :param list partitions: names of the partitions to compare
    :param list numCores: list of the number of cores to scale
    :return: list resultValueList: list of ResultValue objets
    """
    
    resultValuesList = []

    for partition in partitions:
        rs = ResultValues()
        spiking_list = []
        management_list = []
        power_list = []
        for cores in numCores:
            with setEnvWithinContext(PARTITION=partition):
                numSamples = 100
                eProbe = runSequentialMnist(wgtDirName, numSamples=numSamples,
                                            batchSize=10, numCores=cores)

            vddPower = np.mean(eProbe.rawPowerVdd)
            vddmPower = np.mean(eProbe.rawPowerVddm)

            power = vddPower + vddmPower

            spiking = eProbe.totalSpikingTime
            management = eProbe.totalManagementTime

            # normalised execution time

            spiking_norm = round(spiking / numSamples)
            management_norm = round(management / numSamples)

            spiking_list.append(spiking_norm)
            management_list.append(management_norm)
            power_list.append(int(power))

        static_power, active_power = powerFit(numCores, power_list)
        rs.static_power = round(static_power, 2)
        rs.active_power = round(active_power, 2)

        # SequentialImageClassifierLsnn_bench_breakdown
        with setEnvWithinContext(PARTITION=partition):
            numSamples = 100
            eProbe, spk, mgmt, pTime = \
                runSequentialMnist_breakdown(wgtDirName,
                                             numSamples=numSamples,
                                             batchSize=10)

            spiking = eProbe.totalSpikingTime
            management = eProbe.totalManagementTime

            # normalised execution time
            spiking_norm = round(spiking / numSamples)
            management_norm = round(management / numSamples)

        # breakdown of phases
        # time of image transfer in ms
        rs.spk_transfer = round((spk[0] / numSamples) / 1000, 2)
        # time of spike generation and injection in ms
        rs.spk_gen_inj = round((spk[1] / numSamples) / 1000, 2)
        # time of spike processing in neuro cores in ms (calculation)
        rs.spk_process = round(
            (spiking_norm / 1000) - rs.spk_transfer - rs.spk_gen_inj, 2)
        # time of classification in ms
        rs.mgmt_classification = round((mgmt[0] / numSamples) / 1000, 3)
        # time of reset the neuro cores in ms
        rs.mgmt_reset = round((mgmt[1] / numSamples) / 1000, 2)
        # execution time per sample (calc) in ms
        rs.time_sample_calc = round(
            rs.spk_transfer + rs.spk_gen_inj + rs.spk_process + \
            rs.mgmt_classification + rs.mgmt_reset, 2)
        # execution time per sample (probe) in ms
        rs.time_sample_probe = round((spiking_norm + management_norm) / 1000, 2)
        # Because we use binSize=100, the management_norm is larger than the
        # sum of
        # the mgmt_* parts. It adds up with binSize=1. The difference is only
        # around 50µs
        # execution time per sample (python time measurements - control) in ms
        rs.time_sample_python = round((pTime / numSamples) * 1000, 2)

        # average time per time step
        # time of image transfer per time step in µs
        rs.spk_transfer_ts = round((rs.spk_transfer / 840) * 1000, 2)
        # time of spike generation and injection per time step in µs
        rs.spk_gen_inj_ts = round((rs.spk_gen_inj / 840) * 1000, 2)
        # time of spike processing in neuro cores per time step in µs
        rs.spk_process_ts = round((rs.spk_process / 840) * 1000, 2)
        # time of classification per time step in µs
        rs.mgmt_classification_ts = round((rs.mgmt_classification / 840) * 1000,
                                          3)
        # time of reset the neuro cores per time step in µs
        rs.mgmt_reset_ts = round((rs.mgmt_reset / 840) * 1000, 2)
        # execution time per time step in µs
        rs.time_sample_ts = round((rs.time_sample_calc / 840) * 1000, 2)

        # calculation of energy from power and time measurements
        # total power during inference in W
        rs.total_power = round((rs.static_power + rs.active_power) / 1000, 3)
        # total energy per inference in mJ
        rs.energy_sample = round(rs.total_power * rs.time_sample_calc, 3)
        # static energy per inference in mJ
        rs.static_energy_sample = round(
            rs.static_power * rs.time_sample_calc / 1000, 3)
        # active energy per inference in mJ
        rs.active_energy_sample = round(
            rs.active_power * rs.time_sample_calc / 1000, 3)

        rs.partition = partition

        resultValuesList.append(rs)
        
    return resultValuesList

            
if __name__ == "__main__":
    numSamples = 100
    wgtDirName = 'v25_94per' # name of the folder with the trainend weights
    numCores = [1, 2, 5, 10, 20, 40]
    partitions = ["wm_perf", "nahuku08"]
    
    resultValueList = performBenchmark(wgtDirName, numSamples, partitions, numCores)

    for results in resultValueList:
        plotResultTable(results)

    plt.show()
