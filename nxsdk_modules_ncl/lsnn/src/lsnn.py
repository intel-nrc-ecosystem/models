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


import nxsdk.api.n2a as nx
import numpy as np
from collections import namedtuple


class ModelParams:
    """Contains all public model parameters for LsnnNet and computes derived\
    model parameters."""

    def __init__(self, numRegular, numAdaptive, wIn, wRec, wOut):
        """Initializes the ModelParams class for LsnnNet constructor.\
        These parameters are mandatory and need to be provided.

        :param int numRegular: Number of regular neurons
        :param int numAdaptive: Number of adaptive neurons
        :param array2d wIn: Input weight matrix used to connect input layer to\
                            recurrent layer
        :param array2d wRec: Recurrent weight matrix used to connect recurrent\
                             layer to itself
        :param array2d wOut: Output weight matrix used to connect recurrent\
                             layer to output layer
        """
        self.numRegular = numRegular
        self.numAdaptive = numAdaptive
        # Neural dynamics related
        self._vth = 127
        self._tauU = 0
        self._tauV = 20
        self._refrac = 1
        # The firing threshold of adapting neurons get increased by
        # self._beta/self._tauAdaption each time the neuron fires
        self._tauAdaption = 700  # adaption time constant
        self._beta = 1390000
        # _thrAdaScale: This is a scaling factor for the connection from the
        #  main compartment to the auxiliary compartment of an adaptive neuron.
        #  Look at getter method for a more detailed description.
        self._thrAdaScale = 0
        self._weightAux = None
        self._numDelayBits = 0

        # Connection/delay matrices of input/recurrent/output groups
        self._wIn = None
        self._wRec = None
        self._wOut = None
        self.wIn = wIn
        self.wRec = wRec
        self.wOut = wOut
        self._dlyIn = None
        self._dlyRec = None
        self._dlyOut = None

    # Helper methods
    def _assertIsDefined(self, field):
        assert getattr(self, field) is not None, '%s undefined' % (field[1:])

    # Getter/Setter interface
    @property
    def vth(self):
        """Spiking threshold of the membrane voltage"""
        self._assertIsDefined('_vth')
        return self._vth

    @vth.setter
    def vth(self, val):
        assert isinstance(val, int), 'vth must be a positive integer.'
        assert val < 2 ** 23, 'vth must be < 2**23'
        self._vth = val

    @property
    def tauU(self):
        """Time-constant for current decay in a neuron/compartment.\
        A smaller number refers to a faster decay with 0 being an instant decay.
        """
        self._assertIsDefined('_tauU')
        return self._tauU

    @tauU.setter
    def tauU(self, val):
        assert isinstance(val, int), 'tauU must be a positive integer.'
        assert val <= 2 ** 12, 'tauU must be < 2**12'
        self._tauU = val

    @property
    def decayU(self):
        """Inverse of time-constant for current decay in a neuron/compartment"""
        return int(1 / self.tauU * 2 ** 12) \
            if self.tauU > 0 else int(2 ** 12)

    @property
    def tauV(self):
        """Time-constant for voltage decay in a neuron/compartment.\
        A smaller number refers to a faster decay with 0 being an instant decay.
        """
        self._assertIsDefined('_tauV')
        return self._tauV

    @tauV.setter
    def tauV(self, val):
        assert isinstance(val, int), 'tauV must be a positive integer.'
        assert val <= 2 ** 12, 'tauV must be < 2**12'
        self._tauV = val

    @property
    def decayV(self):
        """Inverse of time-constant for current decay in a neuron/compartment"""
        return int(1 / self.tauV * 2 ** 12) \
            if self.tauV > 0 else int(2 ** 12)

    @property
    def refrac(self):
        """Refractory time steps after a neuron spiked."""
        self._assertIsDefined('_refrac')
        return self._refrac

    @refrac.setter
    def refrac(self, val):
        assert isinstance(val, int), 'refrac must be a positive integer.'
        assert val < 64, 'refrac must be < 64'
        self._refrac = val

    @property
    def tauAdaption(self):
        """Time-constant of the adaptive threshold. Is applied in the auxiliary\
        compartment of the adaptive neurons.\
        A smaller number refers to a faster decay with 0 being an instant decay.
        """
        self._assertIsDefined('_tauAdaption')
        return self._tauAdaption

    @tauAdaption.setter
    def tauAdaption(self, val):
        assert isinstance(val, int), 'tauAdaption must be a positive integer.'
        assert val < 2 ** 12, 'tauAdaption must be < 2**12'
        self._tauAdaption = val

    @property
    def decayAdaption(self):
        """Inverse of time-constant for adaptive threshold decay in an adaptive\
        neuron/compartment. Applied in the auxiliary compartments.
        """
        return int(1 / self.tauAdaption * 2 ** 12) \
            if self.tauAdaption > 0 else int(2 ** 12)

    @property
    def decayAuxU(self):
        """Current decay in the auxiliary compartment of an adaptive neuron."""
        return int(2 ** 12)

    @property
    def beta(self):
        """Factor to determine the dynamics of the adaptive threshold of\
        adaptive neurons. The threshold gets increased by beta/tauAdaptive,\
        each time the neuron spikes.
        """
        self._assertIsDefined('_beta')
        return self._beta

    @beta.setter
    def beta(self, val):
        assert isinstance(val, int), 'beta must be a positive integer.'
        self._beta = val

    @property
    def weightAux(self):
        """Weight of the connection between the main compartment and the\
        auxiliary compartment of an adaptive neuron to achieve an adaptive\
        threshold behavior.
        """
        weight = self._weightAux if self._weightAux is not None else int(self.beta * (1 / (self.tauAdaption * self.tauV) - 1 / (self.tauAdaption ** 2)))

        assert weight < 255, "The combination of beta, tauV and tauAdaption " \
                             "must be chosen in a way that the threshold " \
                             "increase does not exceed a weight of 255. " \
                             "Setting a smaller beta value is suggested."

        return weight

    @property
    def thrAdaScale(self):
        """This is a scaling factor for the connection between the main\
        compartment and the regular compartment of an adaptive neuron. As the\
        wgt of any connection is always multiplied by 2**6, this factor\
        allows to negate that. The available range of the membrane potential\
        of the auxiliary compartment of and adaptive neuron should be as high\
        as possbile - at the same time a small change does not reflect the\
        exponential decay, as the decay per time step is limited at the lower\
        range to 1 unit per time step. So a bigger change of the membrane\
        potential is needed to enable more states. Scaling the weight by 2**6\
        might be too high - we want to scale it by thrAdaScale.
        """
        self._assertIsDefined('_thrAdaScale')
        return self._thrAdaScale

    @thrAdaScale.setter
    def thrAdaScale(self, val):
        assert isinstance(val, int), 'thrAdaScale must be a positive integer.'
        assert val < 7, 'thrAdaScale must be < 7'
        self._thrAdaScale = val

    @property
    def vthAdaptive(self):
        """Spiking threshold mantissa of the membrane voltage of the main\
        compartment of adaptive neurons.
        """
        return self.vth * self.thrAdaScale

    @property
    def wIn(self):
        """Input weight matrix used to connect input layer to recurrent layer"""
        self._assertIsDefined('_wIn')
        return self._wIn

    @wIn.setter
    def wIn(self, val):
        assert np.max(val) < 256 and np.min(val) >= -256, 'wIn weights must' \
                                                          ' be with -255 and' \
                                                          ' 256.'
        self._wIn = val

    @property
    def wRec(self):
        """Recurrent weight matrix used to connect recurrent layer to itself"""
        self._assertIsDefined('_wRec')
        return self._wRec

    @wRec.setter
    def wRec(self, val):
        assert np.max(val) < 256 and np.min(val) >= -256, 'wIn weights must' \
                                                          ' be with -255 and' \
                                                          ' 256.'
        self._wRec = val

    @property
    def wOut(self):
        """Output weight matrix used to connect recurrent layer to output layer
        """
        self._assertIsDefined('_wOut')
        return self._wOut

    @wOut.setter
    def wOut(self, val):
        assert np.max(val) < 256 and np.min(val) >= -256, 'wIn weights must' \
                                                          ' be with -255 and' \
                                                          ' 256.'
        self._wOut = val

    @property
    def dlyIn(self):
        """Input delay matrix used for synaptic delays on the input to\
        recurrent layer connections
        """
        self._assertIsDefined('_dlyIn')
        return self._dlyIn

    @dlyIn.setter
    def dlyIn(self, val):
        assert np.max(val) < 64, 'dlyIn delays must be < 64'
        self._dlyIn = val

    @property
    def dlyRec(self):
        """Recurrent delay matrix used for synaptic delays on the recurrent to\
        recurrent layer connections
        """
        self._assertIsDefined('_dlyRec')
        return self._dlyRec

    @dlyRec.setter
    def dlyRec(self, val):
        assert np.max(val) < 64, 'dlyRec delays must be < 64'
        self._dlyRec = val

    @property
    def dlyOut(self):
        """Output delay matrix used for synaptic delays on the recurrent to\
        output layer connections
        """
        self._assertIsDefined('_dlyOut')
        return self._dlyOut

    @dlyOut.setter
    def dlyOut(self, val):
        assert np.max(val) < 64, 'dlyOut delays must be < 64'
        self._dlyOut = val


class CompileParams:
    """Parameter class for configuring compilation parameters for LsnnNet.\
    The parameters are optional.
    """

    def __init__(self):
        self._net = None
        self._logicalCoreId = -1

    # Helper methods
    def _assertIsDefined(self, field):
        assert getattr(self, field) is not None, '%s undefined' % (field[1:])

    # Getter/setter interface
    @property
    def net(self):
        """Defines the NxNet object of the LsnnNet.
        """
        self._assertIsDefined('_net')
        return self._net

    @net.setter
    def net(self, val):
        self._net = val

    @property
    def logicalCoreId(self):
        """Defines which NeuroCore should be used to configure the recurrent\
        network. Only applicable if the network fits on one NeuroCore. Use with\
        care.
        """
        self._assertIsDefined('_logicalCoreId')
        return self._logicalCoreId

    @logicalCoreId.setter
    def logicalCoreId(self, val):
        assert -1 <= val < 4096, "logicalCoreId must be with -1 to 4096"
        self._logicalCoreId = val


class LsnnNet:
    """Represents a neural network which implements a spiking version of a long\
    short-term memory (Long short-term memory spiking neural netwok) according\
    to: https://arxiv.org/abs/1803.09574

    The LSNN is a recurrent spiking neural networks (RSNN) and incoperates\
    neurons which model a prominent dynamical process of a biological neuron:\
    neuronal adaptation. The inclusion of adapting neurons increases the\
    computing powers of RSNNs.

    Such a network structure allows to identify patterns in time series data.

    The LsnnNet module provides an interface to set up a LSNN network of a\
    given size, connect the input and output layer, execute the network and\
    probe its states.

    Inputs have to be provided either via spike generators, spikeInputPorts\
    (SNIPs) or by configuring an input neuron group. These "input ports" get\
    connected to the recurrent portion of the network using a user-provided\
    connection matrix.

    The recurrent part of the network consists of a group of regular LIF\
    neurons and a group of adapting neurons. Both are recurrently connected via\
    a user-provided connection matrix.

    The output layer can be provided as a neuron group, which gets\
    connected to the recurrenct portion of the network by using a user-provided\
    connection matrix.

    In order to setup and configure a network, use the LsnnNet() constructor\
    and provide an instance of ModelParams and optional CompileParams.\
    ModelParams contains functionally relevant model parameters such as\
    number of regular and adaptive neurons, input and output layers as well as \
    input, recurrent and output connections matrices. Additionally the\
    parameters for the neural dynamics of the regular and adaptive neurons can\
    be changed. Optionally synaptic delays can be configured via delay matrices.
    CompileParams contains parameters relevant for compilation. Using defaults\
    is usually sufficient.

    After constructing the LsnnNet module, the network can be generated via\
    LsnnNet.generateNetwork(). This will configure the network according to\
    users settings.

    Once the network is generated, the network can be run via the LsnnNet.run()\
    method. One might also want to configure SNIPs beforehand. After the\
    network is generated the board is available through LsnnNet.board property.

    The module provides methods to probe certain compartment groups or a\
    probeAll() method, mainly to debug the network. It is not recommended to\
    probe always everything as it slows down the execution.

    """

    def __init__(self, inputGroup, outputGroup, modelParams,
                 compileParams=None):
        """Initializes LSNN network model using ModelParams and CompileParams\
        classes.

        :param var inputGroup: CompartmentGroup, inputPorts or spike generators\
                               which represent the input layer.
        :param CompartmentGroup outputGroup: CompartmentGroup, which represent\
                                             the output layer.
        :param ModelParams modelParams: An instance of ModelParams to configure\
         model-related parameters.
        :param CompileParams compileParams: An instance of CompileParams to\
        configure compilation-related parameters.
        """

        self._board = None
        self._inputGroup = None
        self._outputGroup = None
        self._regularNeuronProbes = None
        self._adaptiveMainProbes = None
        self._adaptiveAuxProbes = None
        self._outputProbes = None

        self.inputGroup = inputGroup  # could be spike gens/ports or neurons
        self.outputGroup = outputGroup
        self.modelParams = modelParams
        self._compileParams = compileParams

        if self.compileParams is None:
            self.compileParams = CompileParams()
            self.compileParams.net = nx.NxNet()

        if self.compileParams is not None and self.compileParams.net is None:
            self.compileParams.net = nx.NxNet()

        self._setupNetwork()

    # Helper method
    def _assertIsDefined(self, field):
        assert getattr(self, field) is not None, '%s undefined' % (field[1:])

    # Getter/setter interface
    @property
    def board(self):
        """Provides access to the board on which the LsnnNet was generated.\
        LsnnNet.generateNetwork() has to be called before the board property is\
        available.
        """
        self._assertIsDefined('_board')
        return self._board

    @property
    def compileParams(self):
        """Provides access to the compile parameters.
        """
        self._assertIsDefined('_compileParams')
        return self._compileParams

    @compileParams.setter
    def compileParams(self, val):
        self._compileParams = val

    @property
    def regularNeuronProbes(self):
        """The probes of the regular neurons. Need to be set up by calling\
         probeRegularNeurons() before running.
        """
        self._assertIsDefined('_regularNeuronProbes')
        return self._regularNeuronProbes

    @property
    def adaptiveMainProbes(self):
        """The probes of the main compartments of the adaptive neurons. Need to\
        be set up by calling probeAdaptiveMainCompartments() before running.
        """
        self._assertIsDefined('_adaptiveMainProbes')
        return self._adaptiveMainProbes

    @property
    def adaptiveAuxProbes(self):
        """The probes of the auxiliary compartments of the adaptive neurons.\
        Need to be set up by calling probeAdaptiveAuxCompartments() before\
        running.
        """
        self._assertIsDefined('_adaptiveAuxProbes')
        return self._adaptiveAuxProbes

    @property
    def outputProbes(self):
        """The probes of the output neurons. Need to be set up by calling\
         probeOutputNeurons() before running.
        """
        self._assertIsDefined('_outputProbes')
        return self._outputProbes

    @property
    def inputGroup(self):
        """CompartmentGroup, inputPorts or spike generators which represent the\
        input layer.
        """
        self._assertIsDefined('_inputGroup')
        return self._inputGroup

    @inputGroup.setter
    def inputGroup(self, val):
        self._inputGroup = val

    @property
    def outputGroup(self):
        """CompartmentGroup which represent the output layer.
        """
        self._assertIsDefined('_outputGroup')
        return self._outputGroup

    @outputGroup.setter
    def outputGroup(self, val):
        self._outputGroup = val

    def _setupNetwork(self):
        """Configures the compartmentGroups for regular and adaptive neurons\
        and creates the connections between the input, recurrent and output\
        layer.
        """

        # create compartmentGroups
        self._createRegularNeuronGroup()
        self._createAdaptiveNeuronGroup()
        self._createRecurrentNeuronGroup()

        ## create connections
        self._createInputToReccurentLayerConnections()
        self._createReccurentLayerConnections()
        self._createReccurentToOutputLayerConnections()

    def _createRegularNeuronGroup(self):
        """Configures the compuartmentGroup for the regular neurons."""

        # Create compartmentPrototype
        cxProto = nx.CompartmentPrototype(
            logicalCoreId=self.compileParams.logicalCoreId,
            vThMant=self.modelParams.vth,
            compartmentCurrentDecay=self.modelParams.decayU,
            compartmentVoltageDecay=self.modelParams.decayV,
            refractoryDelay=self.modelParams.refrac,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            numDendriticAccumulators=16
        )
        self.regularNeuronGroup = self.compileParams.net.createCompartmentGroup(
            size=self.modelParams.numRegular, prototype=cxProto)

    def _createAdaptiveNeuronGroup(self):
        """Configures the compuartmentGroup for the adaptive neurons."""

        # Create auxiliary compartment
        auxMantissa = 10000  # it should never spike and never get positive
        auxCxProto = nx.CompartmentPrototype(
            logicalCoreId=self.compileParams.logicalCoreId,
            vThMant=auxMantissa,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            compartmentCurrentDecay=self.modelParams.decayAuxU,
            compartmentVoltageDecay=self.modelParams.decayAdaption,
            stackOut=nx.COMPARTMENT_OUTPUT_MODE.PUSH,
            thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE
                .NO_SPIKE_AND_PASS_V_LG_VTH_TO_PARENT,
            numDendriticAccumulators=16
        )

        # Create main compartment
        mainCxProto = nx.CompartmentPrototype(
            logicalCoreId=self.compileParams.logicalCoreId,
            vThMant=int(self.modelParams.vthAdaptive),
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            compartmentCurrentDecay=self.modelParams.decayU,
            compartmentVoltageDecay=self.modelParams.decayV,
            compartmentJoinOperation=nx.COMPARTMENT_JOIN_OPERATION.ADD,
            stackIn=nx.COMPARTMENT_INPUT_MODE.POP_A,
            refractoryDelay=self.modelParams.refrac,
            numDendriticAccumulators=16
        )

        self.adaptiveNeuronGroup = \
            self.compileParams.net.createCompartmentGroup(
                size=self.modelParams.numAdaptive * 2,
                prototype=[auxCxProto, mainCxProto],
                prototypeMap=[0, 1] * self.modelParams.numAdaptive)

        # Connect main to auxiliary compartment
        mainToAuxConnProto = nx.ConnectionPrototype(
            weight=-self.modelParams.weightAux,
            weightExponent=-self.modelParams.thrAdaScale,
            signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE,
            numDelayBits=self.modelParams._numDelayBits
        )

        self.adaptiveMainCxGroup = \
            self.compileParams.net.createCompartmentGroup()
        self.adaptiveAuxCxGroup = \
            self.compileParams.net.createCompartmentGroup()

        for i in range(0, self.modelParams.numAdaptive * 2, 2):
            auxCx = self.adaptiveNeuronGroup[i]
            mainCx = self.adaptiveNeuronGroup[i + 1]
            mainCx.connect(auxCx, prototype=mainToAuxConnProto)
            self.adaptiveMainCxGroup.addCompartments(mainCx)
            self.adaptiveAuxCxGroup.addCompartments(auxCx)

    def _createRecurrentNeuronGroup(self):
        """Creates a compartment group which contains the regular and adaptive\
        compartment groups.
        """
        self.recurrentNeuronGroup = \
            self.compileParams.net.createCompartmentGroup()
        self.recurrentNeuronGroup.addCompartments(self.regularNeuronGroup)
        self.recurrentNeuronGroup.addCompartments(self.adaptiveMainCxGroup)

    @staticmethod
    def _createConnectionGroup(srcGrp, dstGrp, wgts, dlys, wgtExp=0):
        """creates connection gropus given a weight matrix, delay matrix and\
        optional weight exponent.

        :param CompartmentGroup srcGrp: Source compartment group which shoud\
        be connected to dstGrp.
        :param CompartmentGroup dstGrp: Compartment group which is connected to\
        srcGrp.
        :param array2d wgts: Weight matrix for the connections.
        :param array2d dlys: Delay matrix for delays on the connections.
        :param int wgtExp: Weight exponent for the connection weights.
        """

        # -ve weights will saturate to 0
        posConnGroup = srcGrp.connect(dstGrp,
                                      prototype=nx.ConnectionPrototype(
                                          signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                          compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                                          numDelayBits=4,
                                          weightExponent=wgtExp
                                      ),
                                      connectionMask=(wgts > 0),
                                      weight=wgts, delay=dlys)

        # +ve weights will saturate to 0
        negConnGroup = srcGrp.connect(dstGrp,
                                      prototype=nx.ConnectionPrototype(
                                          signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                                          compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                                          numDelayBits=4,
                                          weightExponent=wgtExp
                                      ),
                                      connectionMask=(wgts < 0),
                                      weight=wgts, delay=dlys)

        ConnGroup = namedtuple("ConnGroup", "positive negative")

        return ConnGroup(positive=posConnGroup, negative=negConnGroup)

    def _createReccurentLayerConnections(self):
        """Connects the recurrent neurons - regular and adaptive neurons - with\
        each other, based on the the connection matrix wRec and optional delay\
        matrix dlyRec.
        """

        wgt = self.modelParams.wRec
        try:
            dly = self.modelParams.dlyRec
        except AssertionError:
            dly = np.zeros(wgt.shape)

        n = self.modelParams.numRegular

        self.connRegularToRegularNeurons = self._createConnectionGroup(
            srcGrp=self.regularNeuronGroup,
            dstGrp=self.regularNeuronGroup,
            wgts=wgt[:n, :n],
            dlys=dly[:n, :n])

        self.connRegularToAdapativeNeurons = self._createConnectionGroup(
            srcGrp=self.regularNeuronGroup,
            dstGrp=self.adaptiveMainCxGroup,
            wgts=wgt[n:, :n],
            dlys=dly[n:, :n],
            wgtExp=6 - self.modelParams.thrAdaScale)

        self.connAdaptiveToAdapativeNeurons = self._createConnectionGroup(
            srcGrp=self.adaptiveMainCxGroup,
            dstGrp=self.adaptiveMainCxGroup,
            wgts=wgt[n:, n:],
            dlys=dly[n:, n:],
            wgtExp=6 - self.modelParams.thrAdaScale)

        self.connAdaptiveToRegularNeurons = self._createConnectionGroup(
            srcGrp=self.adaptiveMainCxGroup,
            dstGrp=self.regularNeuronGroup,
            wgts=wgt[:n, n:],
            dlys=dly[:n, n:])

    def _createReccurentToOutputLayerConnections(self):
        """Connects the recurrent neurons with the output neurons given the\
        connection matrix wOut and optional delay matrix dlyOut.
        """

        wgt = self.modelParams.wOut
        try:
            dly = self.modelParams.dlyOut
        except AssertionError:
            dly = np.zeros(wgt.shape)
        n = self.modelParams.numRegular

        self.connRegularToOutputNeurons = self._createConnectionGroup(
            srcGrp=self.regularNeuronGroup,
            dstGrp=self.outputGroup,
            wgts=wgt[:, :n],
            dlys=dly[:, :n])

        self.connAdaptiveToOutputNeurons = self._createConnectionGroup(
            srcGrp=self.adaptiveMainCxGroup,
            dstGrp=self.outputGroup,
            wgts=wgt[:, n:],
            dlys=dly[:, n:])

    def _createInputToReccurentLayerConnections(self):
        """Connects the input neurons with the recurrent neurons given the\
        connection matrix wIn and the optional delay matrix dlyIn.
        """

        n = self.modelParams.numRegular

        wgt = self.modelParams.wIn

        try:
            dly = self.modelParams.dlyIn
        except AssertionError:
            dly = np.zeros(wgt.shape)

        posConnGroupReg = self.inputGroup.connect(
            self.regularNeuronGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4),
            connectionMask=(wgt[:n, :] > 0),
            weight=wgt[:n, :], delay=dly[:n, :])

        posConnGroupAda = self.inputGroup.connect(
            self.adaptiveMainCxGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=0 if self.modelParams.thrAdaScale == 0 else 6 - self.modelParams.thrAdaScale),
            connectionMask=(wgt[n:, :] > 0),
            weight=wgt[n:, :], delay=dly[n:, :])

        negConnGroupReg = self.inputGroup.connect(
            self.regularNeuronGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4),
            connectionMask=(wgt[:n, :] < 0),
            weight=wgt[:n, :], delay=dly[:n, :])

        negConnGroupAda = self.inputGroup.connect(
            self.adaptiveMainCxGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=0 if self.modelParams.thrAdaScale == 0 else 6 - self.modelParams.thrAdaScale),
            connectionMask=(wgt[n:, :] < 0),
            weight=wgt[n:, :], delay=dly[n:, :])

        ConnGroup = namedtuple("ConnGroup", "positiveReg positiveAda "
                                            "negativeReg negativeAda")

        self.inputNeuronConnections = ConnGroup(positiveReg=posConnGroupReg,
                                                positiveAda=posConnGroupAda,
                                                negativeReg=negConnGroupReg,
                                                negativeAda=negConnGroupAda)

    @staticmethod
    def _isProbeParams(probeParams):
        """Checks if the probe parameters are valid.

        :param list of nxsdk.api.n2a.ProbeParameter probeParams: A single\
        object or list of objects of the type ProbeParameter
        """
        if isinstance(probeParams, list):
            for p in probeParams:
                assert isinstance(p, nx.ProbeParameter)
        else:
            assert isinstance(probeParams, nx.ProbeParameter)

    def probeRegularNeurons(self, probeParams):
        """Configures probes for the regular neurons with the given probe\
        parameters.

        :param list of nxsdk.api.n2a.ProbeParameter probeParams: A single\
        object or list of objects of the type ProbeParameter
        """
        self._isProbeParams(probeParams)
        self._regularNeuronProbes = self.regularNeuronGroup.probe(probeParams)

    def probeAdaptiveMainCompartments(self, probeParams):
        """Configures probes for the adaptive neurons main compartment with the\
        given probe parameters.

        :param list of nxsdk.api.n2a.ProbeParameter probeParams: A single\
        object or list of objects of the type ProbeParameter
        """
        self._isProbeParams(probeParams)
        self._adaptiveMainProbes = self.adaptiveMainCxGroup.probe(probeParams)

    def probeAdaptiveAuxCompartments(self, probeParams):
        """Configures probes for the adaptive neurons auxiliary compartment\
        with the given probe parameters.

        :param list of nxsdk.api.n2a.ProbeParameter probeParams: A single\
        object or list of objects of the type ProbeParameter
        """
        self._isProbeParams(probeParams)
        self._adaptiveAuxProbes = self.adaptiveAuxCxGroup.probe(probeParams)

    def probeOutputNeurons(self, probeParams):
        """Configures probes for the output neurons with the given probe\
        parameters.

        :param list of nxsdk.api.n2a.ProbeParameter probeParams: A single\
        object or list of objects of the type ProbeParameter
        """
        self._isProbeParams(probeParams)
        self._outputProbes = self.outputGroup.probe(probeParams)

    def probeAll(self):
        """Configures probes on all compartments with current, voltage and\
        spike probing parameters."""

        PP = nx.ProbeParameter
        params = [PP.COMPARTMENT_CURRENT, PP.COMPARTMENT_VOLTAGE, PP.SPIKE]

        self._regularNeuronProbes = self.regularNeuronGroup.probe(params)
        self._adaptiveMainProbes = self.adaptiveMainCxGroup.probe(params)
        self._adaptiveAuxProbes = self.adaptiveAuxCxGroup.probe(params)
        self._outputProbes = self.outputGroup.probe(params)

    def generateNetwork(self):
        """Compiles the network and sets the board. This is needed to be able\
        to configure channels/SNIPs.
        """
        self._board = nx.N2Compiler().compile(self.compileParams.net)

    def run(self, tSteps, aSync=False):
        """Runs the network for tSteps time steps.

        :param int tSteps: number of time steps the network should run
        :param bool aSync: flag if it should be run in asynchronous mode
        """
        self.board.run(tSteps, aSync)

    def finish(self):
        """Ends the execution and disconnects the board. Call after last\
        run() call.
        """
        self.board.finishRun()
        self.board.disconnect()


