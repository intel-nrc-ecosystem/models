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
import nxsdk.api.n2a as nx
from nxsdk_modules.lsnn.src.lsnn import LsnnNet, ModelParams, CompileParams


class SequentialImageClassifierLsnn(LsnnNet):
    """This class demonstrates an application of LsnnNet module,\
    by performing a sequential image classification task. Instead of\
    presenting the image at once to the network, each pixel is presented\
    sequentially for one time step.

    The weights for the network were trainend off chip, with a TensorFlow model.
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
        # initialize parameters
        self.batchSize = batchSize
        self.cueDuration = cueDuration
        self.imageSize = imageSize
        self._numInput = numInput
        self._numOutput = numOutput
        self._inputs = None
        self._targets = None
        self._classifications = None
        self._snipsDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'snips')

        # define parameters for LsnnNet
        self._modelP = ModelParams(numRegular, numAdaptive, wIn, wRec,
                                   wOut)
        self._modelP.thrAdaScale = 4
        self._modelP._numDelayBits = 4
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

        # compile and setup snips and channels
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
            b'nxinit_input_params', numElements=3, messageSize=4)
        self.inputParamsChannel.connect(None, self.initProcess)

    def _setupInputPortsChannel(self):
        """Creates the channel for the input ports"""
        self.inputPortsChannel = self.board.createChannel(
            b'nxinit_input_ports', numElements=4 * self._numInput, messageSize=4)
        self.inputPortsChannel.connect(None, self.initProcess)

    def _setupThresholdsChannel(self):
        """Creates the channel for the thresholds (threshold crossing spike\
        generation)
        """
        self.thresholdsChannel = self.board.createChannel(
            b'nxinit_thresholds', numElements=self._numInput // 2, messageSize=4)
        self.thresholdsChannel.connect(None, self.initProcess)

    def _setupOutputNeuronChannel(self):
        """Creates the channel for the output neuron ports"""
        self.outputNeuronChannel = self.board.createChannel(
            b'output_neurons', numElements=4 * self._numOutput, messageSize=4)
        self.outputNeuronChannel.connect(None, self.initProcess)

    def _setupInputDataChannel(self):
        """Creates the channel for the input images"""
        self.inputDataChannel = self.board.createChannel(
            b'nxspk_img_data',
            numElements=self.numSamples * self.imageSize * self.batchSize,
            messageSize=64)
        self.inputDataChannel.connect(None, self.spikingProcess)

    def _setupClassificationChannel(self):
        """Creates the channel for the classification results"""
        self.classificationChannel = self.board.createChannel(
            b'classifications', numElements=1, messageSize=64)
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
        self.inputParamsChannel.write(3, [self._numInput,
                                          self.imageSize, self.batchSize])

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
            c = self.classificationChannel.read(1)  # packed mode (16 integers)
            self._classifications[np.asarray(out)] = c[0:self.batchSize]

            if k % print_update == 0:
                v = int(k / print_update)
                print("\r[%-10s] %d%%" % ('=' * v, 10 * v), end='')

        print("\r[%-10s] %d%%" % ('=' * 10, 10 * 10))

        out = range((count - 1) * self.batchSize, self.numSamples)
        # Read classification result from LMT at end of batch
        c = self.classificationChannel.read(1)

        self._classifications[np.asarray(out)] = c[0:self.batchSize]

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
