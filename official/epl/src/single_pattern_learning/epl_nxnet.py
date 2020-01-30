# Copyright(c) 2019-2020 Intel Corporation All rights reserved.
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
# pylint: disable-all

import nxsdk.api.n2a as nx
import numpy as np
import random
import time
from collections import namedtuple
from nxsdk_modules.epl.src.single_pattern_learning.epl_parameters import \
    ParamsEPLSlots, ParamemtersForEPL


def timer(input_func):
    """ returns the execution time of a function wrapped by this decorator"""
    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        print("{0} took {1:0.5f} secs".format(input_func.__name__,
                                              end_time - start_time))
        return result
    return timed


class EplNxNet(ParamsEPLSlots):
    """NxNet implementation of the EPL network"""

    def __init__(self, eplParams):
        # make sure the type of the parameters class
        assert isinstance(eplParams, ParamemtersForEPL)
        super().__init__()
        # copy the parameters
        for attr in eplParams.__slots__:
            setattr(self, attr, getattr(eplParams, attr))
        self.net = nx.NxNet()
        # ToDo:Needs to be automated in future
        # The bias current value to be used for each MC input/stimulus
        self.stim2bias = [0, 34, 36, 38, 41, 43, 46, 50, 54, 59, 65, 72, 81,
                          92, 107, 129, 161, 214, 321, 641]
        if self.useRandomSeed:
            np.random.seed(self.randomGenSeed)
            random.seed(self.randomGenSeed)
        # probes related data structures
        self.allMCSomaProbes = None
        # self.cxProbeIdxs = cxProbeIdxs
        self.exc2InhConnProbes = None
        self.inh2ExcConnProbesPos = None
        self.inh2ExcConnProbesNeg = None
        self.mcADProbes = None
        self.mcSomaProbes = None
        self.gcProbes = None
        #self.probeIdxs = probeIdxs
        self.numStepsRan = 0

    @property
    def numMCs(self):
        """ returns the total number of MCs in the network"""
        return self.numColumns * self.numMCsPerColumn

    @property
    def numGCs(self):
        """ returns the total number of GCs in the network"""
        return self.numColumns * self.numGCsPerColumn

    @timer
    def setupNetwork(self):
        """ setups the EPL network """
        self.createMCAndSTONetwork()
        self.createMCToGCNetwork()
        # self.setupProbes()

    def createMCAndSTONetwork(self):
        """ setups the MC-AD, MC-Soma and oscillatory (STO) neurons and
        related connections"""
        self.createSTONeurons()
        self.createMCNeurons()
        self.connectSTOsWithMCADs()

    def createMCToGCNetwork(self):
        """ setups the MC->GC and GC->MC connections"""
        self.createGCNeurons()
        self.connectGCToMC()
        self.connectMCToGC()

    def createGCNeurons(self):
        """ configures the GC neurons """
        self.allGCsGroup = self.net.createCompartmentGroup()
        self.gcNeuronGrpPerCoreList = []

        for coreIdx in range(self.numColumns):
            gcGrpPerCore = self.net.createCompartmentGroup()
            gcProtoPerCore = nx.CompartmentPrototype(
                logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=4095,
                enableSpikeBackprop=1,
                enableSpikeBackpropFromSelf=1,
                biasMant=0,
                vThMant=3 * 200,
                refractoryDelay=25,
                vMinExp=0,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )

            for _ in range(self.numGCsPerColumn):
                gcCx = self.net.createCompartment(
                    prototype=gcProtoPerCore)
                gcGrpPerCore.addCompartments(gcCx)
                self.allGCsGroup.addCompartments(gcCx)
            self.gcNeuronGrpPerCoreList.append(gcGrpPerCore)

    def connectGCToMC(self, excDelay=20, inhDelay=20):
        """ creates the GC->MC inhibitory connections """
        ConnGroup = namedtuple("ConnGroup", "positive negative")
        self.gcToMCConnGrpsPerCore = list()
        self.inh2ExcConnProbesPos = list()
        self.inh2ExcConnProbesNeg = list()

        iSTDPLearningRule = self.net.createLearningRule(
            dd='4*2^-7*x1*y0 - 4*2^-7*y1*x0 + 4*x0 - 4*y0',
            x1Impulse=127,
            x1TimeConstant=25,
            y1Impulse=127,
            y1TimeConstant=25,
            r1Impulse=127,
            r1TimeConstant=25,
            tEpoch=40
        )

        for coreIdx in range(self.numColumns):
            excWgts = np.zeros(
                (self.numMCsPerColumn, self.numGCsPerColumn), int)
            excDlys = np.ones_like(excWgts) * excDelay
            inhWgts = np.zeros_like(excWgts)
            inhDlys = np.ones_like(excDlys) * inhDelay

            excConnProtoBox = nx.ConnectionPrototype(
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                enableLearning=1,
                learningRule=iSTDPLearningRule
            )

            inhConnProtoBox = nx.ConnectionPrototype(
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                enableLearning=1,
                learningRule=iSTDPLearningRule
            )

            posConnGrp = self.net.createConnectionGroup(
                src=self.gcNeuronGrpPerCoreList[coreIdx],
                dst=self.mcSomaGrpPerCoreList[coreIdx],
                prototype=excConnProtoBox,
                weight=excWgts,
                delay=excDlys
            )

            negConnGrp = self.net.createConnectionGroup(
                src=self.gcNeuronGrpPerCoreList[coreIdx],
                dst=self.mcSomaGrpPerCoreList[coreIdx],
                prototype=inhConnProtoBox,
                weight=inhWgts,
                delay=inhDlys
            )
            self.gcToMCConnGrpsPerCore.append(ConnGroup(positive=posConnGrp,
                                                        negative=negConnGrp))

    def connectMCToGC(self):
        """ creates the MC->GC excitatory connections """
        minDelay = self.minDelaysMCToGC
        numDelays = self.numDelaysMCToGC
        percent = int(self.connProbMCToGC * 100)

        eSTDPLearningRule = self.net.createLearningRule(
            dw='2^-6*x1*y0',
            x1Impulse=127,
            x1TimeConstant=1,
            y1Impulse=127,
            y1TimeConstant=25,
            tEpoch=40
        )

        self.mcToGCConnGrpsPerDelay = list()
        for delay in range(minDelay, minDelay + numDelays):
            wgtMat = np.zeros((self.numGCs, self.numMCs), int)
            rand = np.random.uniform(0, 100, size=wgtMat.shape)
            wgtMat[rand <= percent] = 201

            connProtoMCToGC = nx.ConnectionPrototype(
                delay=delay,
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                enableLearning=1,
                learningRule=eSTDPLearningRule,
            )

            connGrp = self.net.createConnectionGroup(
                dst=self.allGCsGroup,
                src=self.allMCSomaGrp,
                prototype=connProtoMCToGC,
                connectionMask=(wgtMat > 0),
                weight=wgtMat
            )
            self.mcToGCConnGrpsPerDelay.append(connGrp)

    def createMCNeurons(self):
        """ configures  the MC neurons"""

        self.allMCADGroup = self.net.createCompartmentGroup()
        for mcIdx in range(self.numMCs):
            mcADProto = nx.CompartmentPrototype(
                logicalCoreId=mcIdx//self.numMCsPerColumn,
                compartmentCurrentDecay=0,
                vThMant=10,  # i.e. 10 * 64 = 640
                biasMant=0,
                refractoryDelay=20,
                vMinExp=0,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            mcADCx = self.net.createCompartment(prototype=mcADProto)
            self.allMCADGroup.addCompartments(mcADCx)

        self.allMCSomaGrp = self.net.createCompartmentGroup()
        self.mcSomaGrpPerCoreList = []

        for coreIdx in range(self.numColumns):
            mcSomaNeuronProto = nx.CompartmentPrototype(
                logicalCoreId=coreIdx,
                compartmentCurrentDecay=0,
                compartmentVoltageDecay=4095,
                enableSpikeBackprop=1,
                enableSpikeBackpropFromSelf=1,
                vThMant=2,  # i.e. 2 * 64 = 128
                refractoryDelay=19,
                vMinExp=0,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            mcSomaGrpPerCore = self.net.createCompartmentGroup()
            for _ in range(self.numMCsPerColumn):
                mcSomaNeuronCx = self.net.createCompartment(
                    prototype=mcSomaNeuronProto)
                self.allMCSomaGrp.addCompartments(mcSomaNeuronCx)
                mcSomaGrpPerCore.addCompartments(mcSomaNeuronCx)
            self.mcSomaGrpPerCoreList.append(mcSomaGrpPerCore)

        # Connect each MC-AD neuron to its MC-Soma neuron
        mcADToSomaConnProtoBox = nx.ConnectionPrototype(
            weight=3,
            numWeightBits=8,
            delay=19,
            numDelayBits=6,
            enableDelay=1,
            signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
            postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE
        )

        self.mcADToSomaConns = list()
        for idx in range(self.numMCs):
            conn = self.net._createConnection(
                src=self.allMCADGroup[idx],
                dst=self.allMCSomaGrp[idx],
                prototype=mcADToSomaConnProtoBox)
            self.mcADToSomaConns.append(conn)

    def createSTONeurons(self):
        """ configures the sub-threshold oscillatory neurons """
        self.stoNeuronGroup = self.net.createCompartmentGroup()
        for i in range(self.numMCs):
            stoNeuronProto = nx.CompartmentPrototype(
                logicalCoreId=i,
                compartmentCurrentDecay=4095,
                vThMant=39,
                biasMant=64,
                numDendriticAccumulators=64,
                vMinExp=0,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )

            stoNeuronCx = self.net.createCompartment(prototype=stoNeuronProto)
            self.stoNeuronGroup.addCompartments(stoNeuronCx)

    def connectSTOsWithMCADs(self, wgt=-20):
        """ creates the STO->MC connections """
        connProtoBox = nx.ConnectionPrototype(
            weight=wgt,
            delay=20,
            numDelayBits=6,
            enableDelay=1,
            signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
            postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE
        )
        # ToDo: use only 1 STO per core
        # stoNeuronGroup.connect(dstGrp=eNeuronADGroup, prototype=connProtoBox)
        for idx in range(self.numMCs):
            self.net._createConnection(
                src=self.stoNeuronGroup[idx],
                dst=self.allMCADGroup[idx],
                prototype=connProtoBox)

            self.net._createConnection(
                src=self.stoNeuronGroup[idx],
                dst=self.allMCSomaGrp[idx],
                prototype=connProtoBox)

    @timer
    def compileAndGetBoard(self):
        """ compiles the network """
        self.board = nx.N2Compiler().compile(self.net)
        return self.board

    def setupMCSomaProbes(self):
        """ sets up MC soma spike probes """
        if self.useLMTSpikeCounters:
            pc = nx.SpikeProbeCondition(tStart=1000000)
        else:
            pc = None
        self.allMCSomaProbes = \
            self.allMCSomaGrp.probe(nx.ProbeParameter.SPIKE, pc)[0]

    def setupCxProbes(self):
        """ sets up the MC and GC compartment probes"""
        if self.cxProbeIdxs is None:
            return
        if "mc" in self.cxProbeIdxs.keys():
            self.setupMCProbes()
        if "gc" in self.cxProbeIdxs.keys():
            self.setupGCProbes()
        # self.setupMCToGCSynapseProbes()

    def setupGCProbes(self):
        """ setup GC compartment probes """
        probeParams = [nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE]
        self.gcProbes = dict()
        for coreIdx, gcIdx in self.cxProbeIdxs["gc"]:
            assert coreIdx < self.numColumns
            assert gcIdx < self.numGCsPerCore
            idx = coreIdx * self.numGCsPerCore + gcIdx
            cx = self.allGCsGroup[idx]
            prb = cx.probe(probeParams)
            self.gcProbes[(coreIdx, gcIdx)] = prb

    def setupMCProbes(self):
        """ setup MC compartment probes """
        probeParams = [nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE]
        self.mcADProbes = dict()
        self.mcSomaProbes = dict()
        for coreIdx, mcIdx in self.cxProbeIdxs["mc"]:
            assert coreIdx < self.numColumns
            assert mcIdx < self.numMCsPerCore
            idx = coreIdx * self.numMCsPerCore + mcIdx
            self.mcADProbes[(coreIdx, mcIdx)] = self.allMCADGroup[idx].probe(
                probeParams)
            self.mcSomaProbes[(coreIdx, mcIdx)] = self.allMCSomaGrp[idx].probe(
                probeParams)

    def setupGCToMCSynapseProbes(self):
        """ setup the GC->MC synapse probes """
        self.inh2ExcConnProbesPos = list()
        self.inh2ExcConnProbesNeg = list()
        prbCond = nx.IntervalProbeCondition(tStart=1, dt=100)
        prbParams = [nx.ProbeParameter.SYNAPSE_WEIGHT,
                     nx.ProbeParameter.SYNAPSE_DELAY,
                     nx.ProbeParameter.PRE_TRACE_X1]
        # nx.ProbeParameter.POST_TRACE_Y1],
        for idx in range(self.numColumns):
            ConnGrp = self.gcToMCConnGrpsPerCore[idx]

            posConnGrpPrb = ConnGrp.posConnGrp.probe(prbParams,
                                                     probeConditions=[prbCond] * len(prbParams))
            negConnGrpPrb = ConnGrp.negConnGrp.probe(prbParams,
                                                     probeConditions=[prbCond] * len(prbParams))
            self.inh2ExcConnProbesPos.append(posConnGrpPrb)
            self.inh2ExcConnProbesNeg.append(negConnGrpPrb)

    def setupMCToGCSynapseProbes(self):
        """ setup the MC->GC synapse probes """
        prbCond = nx.IntervalProbeCondition(tStart=1, dt=1)
        prbParams = [nx.ProbeParameter.SYNAPSE_WEIGHT,
                     nx.ProbeParameter.SYNAPSE_DELAY,
                     nx.ProbeParameter.PRE_TRACE_X1]
        # nx.ProbeParameter.POST_TRACE_Y1],
        self.exc2InhConnProbes = list()
        for delayIdx in range(self.numMCToGCDelays):
            connGrp = self.mcToGCConnGrpsPerDelay[delayIdx]
            connGrpPrb = connGrp.probe(prbParams,
                                       probeConditions=[prbCond] * len(prbParams))

            self.exc2InhConnProbes.append(connGrpPrb)
