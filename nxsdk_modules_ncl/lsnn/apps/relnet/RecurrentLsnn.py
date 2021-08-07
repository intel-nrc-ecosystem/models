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

import numpy as np
from collections import namedtuple
import nxsdk.api.n2a as nx
from nxsdk_modules.lsnn.src.lsnn import ModelParams, CompileParams

ConnGroup = namedtuple("ConnGroup", ["positive", "negative"])

class RecurrentLsnn:
    
    connection_cache = {}
    
    def __init__(self, lsnn_params, net, logicalCoreId):
        # TODO: write own parameters class or manage them inside this class (validators)
        self.modelParams = ModelParams(lsnn_params["n_reg"], lsnn_params["n_adap"], 0, 0, 0)

        # parameters
        self.modelParams._vth = lsnn_params["scaled_thr"]
        self.modelParams._tauU = lsnn_params["tau_I"]
        self.modelParams._tauV = lsnn_params["tau_V"]
        self.modelParams._refrac = lsnn_params["n_refractory"]
        self.modelParams._tauAdaption = lsnn_params["tau_adaptation"]
        self.weight_exp = lsnn_params["weight_exp"]
        self.aux_wgt_exp = lsnn_params["beta_exp"]
        self.aux_wgt = lsnn_params["weight_aux"]
        self.functional_state = lsnn_params.get('functional_state', nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
        self.modelParams._thrAdaScale = 1

        self.compileParams = CompileParams()
        self.compileParams.net = net
        self.compileParams.logicalCoreId = logicalCoreId
        
        # print("#######CoreID######## :", logicalCoreId)

    def _createRegularNeuronGroup(self):
        """Configures the compuartmentGroup for the regular neurons."""

        # Create compartmentPrototype
        cxProto = nx.CompartmentPrototype(
            logicalCoreId=self.compileParams.logicalCoreId,
            vThMant=self.modelParams.vth,
            compartmentCurrentDecay=self.modelParams.decayU,
            compartmentVoltageDecay=self.modelParams.decayV,
            refractoryDelay=self.modelParams.refrac,
            functionalState=self.functional_state,
            numDendriticAccumulators=16,
            
            enableHomeostasis=1,
            minActivity=0,
            maxActivity=127,
            homeostasisGain=0,
            activityImpulse=1,
            activityTimeConstant=1000000
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
            numDendriticAccumulators=16,
            
            enableHomeostasis=1,
            minActivity=0,
            maxActivity=127,
            homeostasisGain=0,
            activityImpulse=1,
            activityTimeConstant=1000000
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
            numDendriticAccumulators=16,
            
            enableHomeostasis=1,
            minActivity=0,
            maxActivity=127,
            homeostasisGain=0,
            activityImpulse=1,
            activityTimeConstant=1000000
        )

        self.adaptiveNeuronGroup = \
            self.compileParams.net.createCompartmentGroup(
                size=self.modelParams.numAdaptive * 2,
                prototype=[auxCxProto, mainCxProto],
                prototypeMap=[0, 1] * self.modelParams.numAdaptive)

        # Connect main to auxiliary compartment
        mainToAuxConnProto = nx.ConnectionPrototype(
            weight=-self.aux_wgt,
            weightExponent=self.aux_wgt_exp,
            signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
            numDelayBits=0
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

    def createRecurrentNeuronGroup(self):
        """Creates a compartment group which contains the regular and adaptive\
        compartment groups.
        """
        self._createRegularNeuronGroup()
        self._createAdaptiveNeuronGroup()
        
        self.recurrentNeuronGroup = \
            self.compileParams.net.createCompartmentGroup()
        self.recurrentNeuronGroup.addCompartments(self.regularNeuronGroup)
        self.recurrentNeuronGroup.addCompartments(self.adaptiveMainCxGroup)
        
    def createOutputNeuronGroup(self):
        """Configures an generic output neuron group which does not spike.

        :param int numOutput: number of output neurons
        """
        outMantissa = 2 ** 17 - 1
        cxProto = nx.CompartmentPrototype(
            logicalCoreId=self.compileParams.logicalCoreId,
            vThMant=outMantissa,
            compartmentCurrentDecay=self.modelParams.decayU,
            compartmentVoltageDecay=self.modelParams.decayV,
            refractoryDelay=self.modelParams.refrac,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            numDendriticAccumulators=16
        )
        self.outputNeuronGroup = self.compileParams.net.createCompartmentGroup(size=self.modelParams.numRegular, prototype=cxProto)
    

    @classmethod
    def createConnectionGroup(cls, srcGrp, dstGrp, wgts, dlys, wgtExp=0):
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
        conn_group = ConnGroup(positive=posConnGroup, negative=negConnGroup)

        return conn_group

    def createReccurentLayerConnections(self, wRec, dlyRec):
        """Connects the recurrent neurons - regular and adaptive neurons - with\
        each other, based on the the connection matrix wRec and optional delay\
        matrix dlyRec.
        """

        wgt = wRec
        dly = dlyRec

        n = self.modelParams.numRegular

        self.connRegularToRegularNeurons = self.createConnectionGroup(
            srcGrp=self.regularNeuronGroup,
            dstGrp=self.regularNeuronGroup,
            wgts=wgt[:n, :n],
            dlys=dly[:n, :n],
            wgtExp=self.weight_exp)

        self.connRegularToAdapativeNeurons = self.createConnectionGroup(
            srcGrp=self.regularNeuronGroup,
            dstGrp=self.adaptiveMainCxGroup,
            wgts=wgt[n:, :n],
            dlys=dly[n:, :n],
            wgtExp=self.weight_exp)

        self.connAdaptiveToAdapativeNeurons = self.createConnectionGroup(
            srcGrp=self.adaptiveMainCxGroup,
            dstGrp=self.adaptiveMainCxGroup,
            wgts=wgt[n:, n:],
            dlys=dly[n:, n:],
            wgtExp=self.weight_exp)

        self.connAdaptiveToRegularNeurons = self.createConnectionGroup(
            srcGrp=self.adaptiveMainCxGroup,
            dstGrp=self.regularNeuronGroup,
            wgts=wgt[:n, n:],
            dlys=dly[:n, n:],
            wgtExp=self.weight_exp)
        
    def createInputToReccurentLayerConnections(self, input_group, wIn, dlyIn, weight_exp=None):
        """Connects the input neurons with the recurrent neurons given the\
        connection matrix wIn and the optional delay matrix dlyIn.
        """
        
        self.inputGroup = input_group
        n = self.modelParams.numRegular
        
        wgt = wIn
        dly = dlyIn
        weight_exp = weight_exp if weight_exp is not None else self.weight_exp

        posConnGroupReg = self.inputGroup.connect(
            self.regularNeuronGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=weight_exp),
            connectionMask=(wgt[:n, :] > 0),
            weight=wgt[:n, :], delay=dly[:n, :])

        posConnGroupAda = self.inputGroup.connect(
            self.adaptiveMainCxGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=weight_exp),
            connectionMask=(wgt[n:, :] > 0),
            weight=wgt[n:, :], delay=dly[n:, :])

        negConnGroupReg = self.inputGroup.connect(
            self.regularNeuronGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=weight_exp),
            connectionMask=(wgt[:n, :] < 0),
            weight=wgt[:n, :], delay=dly[:n, :])

        negConnGroupAda = self.inputGroup.connect(
            self.adaptiveMainCxGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=weight_exp),
            connectionMask=(wgt[n:, :] < 0),
            weight=wgt[n:, :], delay=dly[n:, :])
        
        ConnGroup = namedtuple("ConnGroup", "positiveReg positiveAda "
                                            "negativeReg negativeAda")

        self.inputNeuronConnections = ConnGroup(positiveReg=posConnGroupReg,
                                                positiveAda=posConnGroupAda,
                                                negativeReg=negConnGroupReg,
                                                negativeAda=negConnGroupAda)

    def createInputToOutputLayerConnections(self, input_group, wIn, dlyIn):
        """Connects the input neurons with the recurrent neurons given the\
        connection matrix wIn and the optional delay matrix dlyIn.
        """
        
        self.inputGroup = input_group
        n = self.modelParams.numRegular

        wgt = wIn
        dly = dlyIn

        posConnGroupReg = self.inputGroup.connect(
            self.outputNeuronGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=self.weight_exp),
            connectionMask=(wgt > 0),
            weight=wgt, delay=dly)

        negConnGroupReg = self.inputGroup.connect(
            self.outputNeuronGroup,
            prototype=nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                numDelayBits=4,
                weightExponent=self.weight_exp),
            connectionMask=(wgt < 0),
            weight=wgt, delay=dly)

        
        ConnGroup = namedtuple("ConnGroup", "positiveReg "
                                            "negativeReg")

        self.inputNeuronConnections = ConnGroup(positiveReg=posConnGroupReg,
                                                negativeReg=negConnGroupReg)
        
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

        # TODO: we assume 180 inputs
        for i in range(180):
            if i not in portIdToAxonsMap:
                portIdToAxonsMap[i] = [(-1, -1, -1, -1)]

        return portIdToAxonsMap