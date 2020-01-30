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
import math
import random
import time
import os
from collections import namedtuple
from nxsdk_modules.epl.src.multi_pattern_learning.epl_parameters import \
    ParamemtersForEPL, ParamsEPLSlots
import matplotlib.pyplot as plt


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

    @property
    def numGCsPerColumn(self):
        """ returns the total number of GCs for all patterns in a column"""
        return self.numGCsPerPatternPerColumn * self.numPatterns

    @property
    def numGCsPerPattern(self):
        """returns the total number of GCs in all columns for a given
        pattern"""
        return self.numGCsPerPatternPerColumn * self.numColumns

    def setupNetwork(self):
        """ setups the EPL network """
        self.createMCAndSTONetwork()
        self.createMCToGCNetwork()
        # self.setupProbes()

    @timer
    def createMCAndSTONetwork(self):
        """ setups the MC-AD, MC-Soma and oscillatory (STO) neurons and
        related connections"""
        self.createSTONeurons()
        self.createMCNeurons()
        self.connectSTOsWithMCs()

    @timer
    def createMCToGCNetwork(self):
        """ setups the MC->GC and GC->MC connections"""
        self.allGCsPerPattern = dict()
        self.gcToMCConns = dict()
        self.mcToGCConns = dict()
        self.lastUsedLogicalCoreId = 2
        self.gcCoreIdRange = [self.lastUsedLogicalCoreId+1]
        for patternIdx in range(self.numPatterns):
            self.allGCsPerPattern[patternIdx] = \
                self.net.createCompartmentGroup()
            self.gcsPerPatternPerColumn = dict()
            self.createGCNeuronsPerPattern(patternIdx)
            self.connectGCToMC(patternIdx)
            self.connectMCToGC(patternIdx)
        self.gcCoreIdRange.append(self.lastUsedLogicalCoreId)
        #print(self.gcCoreIdRange)


    def createGCNeuronsPerPattern(self, patternIdx):
        """ configures the GC neurons for each pattern"""
        maxColsPerCore = 128//self.numGCsPerPatternPerColumn
        for colIdx in range(self.numColumns):
            self.gcsPerPatternPerColumn[patternIdx, colIdx] = \
                self.net.createCompartmentGroup()
            coreIdx = self.lastUsedLogicalCoreId + \
                math.ceil((colIdx+1)/maxColsPerCore)
            gcProto = nx.CompartmentPrototype(
                logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=4095,
                enableSpikeBackprop=1,
                enableSpikeBackpropFromSelf=1,
                biasMant=0,
                vThMant=3 * 200 if patternIdx == 0 else 2 ** 17 - 1,
                refractoryDelay=25,
                vMinExp=0,
                vMaxExp=17,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
            )
            for _ in range(self.numGCsPerPatternPerColumn):
                gcCx = self.net.createCompartment(prototype=gcProto)
                self.gcsPerPatternPerColumn[patternIdx,
                                            colIdx].addCompartments(gcCx)
                self.allGCsPerPattern[patternIdx].addCompartments(gcCx)
        self.lastUsedLogicalCoreId += math.ceil(self.numGCsPerPattern/128)
        #print(self.lastUsedLogicalCoreId)

    def connectGCToMC(self, patternIdx, excDelay=20, inhDelay=20):
        """ creates the GC->MC inhibitory connections for each pattern"""
        ConnGroup = namedtuple("ConnGroup", "positive negative")

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

        for colIdx in range(self.numColumns):
            excWgts = np.zeros((self.numMCsPerColumn,
                                self.numGCsPerPatternPerColumn), int)
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
                src=self.gcsPerPatternPerColumn[patternIdx, colIdx],
                dst=self.mcSomaGrpPerColumn[colIdx],
                prototype=excConnProtoBox,
                weight=excWgts,
                delay=excDlys
            )

            negConnGrp = self.net.createConnectionGroup(
                src=self.gcsPerPatternPerColumn[patternIdx, colIdx],
                dst=self.mcSomaGrpPerColumn[colIdx],
                prototype=inhConnProtoBox,
                weight=inhWgts,
                delay=inhDlys
            )
            self.gcToMCConns[patternIdx, colIdx] = \
                ConnGroup(positive=posConnGrp, negative=negConnGrp)

    def connectMCToGC(self, patternIdx):
        """ creates the MC->GC excitatory connections for each pattern"""
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

       # self.mcToGCConnGrpsPerDelay = list()
        for delayIdx in range(numDelays):
            wgtMat = np.zeros((self.numGCsPerPattern, self.numMCs), int)
            rand = np.random.uniform(0, 100, size=wgtMat.shape)
            wgtMat[rand <= percent] = 201

            connProtoMCToGC = nx.ConnectionPrototype(
                delay=minDelay + delayIdx,
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                enableLearning=1,
                learningRule=eSTDPLearningRule,
            )

            connGrp = self.net.createConnectionGroup(
                dst=self.allGCsPerPattern[patternIdx],
                src=self.allMCSomaGrp,
                prototype=connProtoMCToGC,
                connectionMask=(wgtMat > 0),
                weight=wgtMat
            )
            # self.mcToGCConnGrpsPerDelay.append(connGrp)
            self.mcToGCConns[patternIdx, delayIdx] = connGrp

    def createMCNeurons(self, biasMant=0):
        """ configures  the MC neurons"""
        mcADProto = nx.CompartmentPrototype(
            logicalCoreId=1,
            compartmentCurrentDecay=0,
            vThMant=10,  # i.e. 10 * 64 = 640
            biasMant=biasMant,
            refractoryDelay=20,
            vMinExp=0,
            numDendriticAccumulators=64,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
        )
        self.allMCADGroup = self.net.createCompartmentGroup(
            prototype=mcADProto, size=self.numMCs)

        mcSomaProto = nx.CompartmentPrototype(
            logicalCoreId=2,
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
        self.allMCSomaGrp = self.net.createCompartmentGroup()

        self.mcSomaGrpPerColumn = list()
        for colIdx in range(self.numColumns):
            self.mcSomaGrpPerColumn.append(self.net.createCompartmentGroup())
            for mcIdx in range(self.numMCsPerColumn):
                mcSomaCx = self.net.createCompartment(prototype=mcSomaProto)
                self.mcSomaGrpPerColumn[colIdx].addCompartments(mcSomaCx)
                self.allMCSomaGrp.addCompartments(mcSomaCx)

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
        stoNeuronProto = nx.CompartmentPrototype(
            logicalCoreId=0,
            compartmentCurrentDecay=4095,
            vThMant=39,
            biasMant=64,
            numDendriticAccumulators=64,
            vMinExp=0,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
        )
        self.stoNeuronGroup = self.net.createCompartmentGroup(
            prototype=stoNeuronProto, size=1)

    def connectSTOsWithMCs(self, wgt=-20):
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

        for mcIdx in range(self.numMCs):
            self.net._createConnection(src=self.stoNeuronGroup[0],
                                       dst=self.allMCADGroup[mcIdx], prototype=connProtoBox,)

            self.net._createConnection(src=self.stoNeuronGroup[0],
                                       dst=self.allMCSomaGrp[mcIdx], prototype=connProtoBox)

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

    def setupCxProbes(self, sto=False):
        """ sets up the MC and GC compartment probes"""
        if not hasattr(self, "cxProbeIdxs"):
            raise AttributeError("Can't set up compartment probes...")
        if "mc" in self.cxProbeIdxs.keys():
            self.setupMCProbes(sto)
        if "gc" in self.cxProbeIdxs.keys():
            self.setupGCProbes()
        # self.setupMCToGCSynapseProbes()

    def setupGCProbes(self):
        """ setup GC compartment probes """
        probeParams = [nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE]
        self.gcProbes = dict()
        for colIdx, gcIdx in self.cxProbeIdxs["gc"]:
            assert colIdx < self.numColumns
            assert gcIdx < self.numGCsPerCore
            idx = colIdx * self.numGCsPerCore + gcIdx
            cx = self.allGCsPerPattern[idx]
            prb = cx.probe(probeParams)
            self.gcProbes[(colIdx, gcIdx)] = prb

    def setupMCProbes(self, sto):
        """ setup MC compartment probes """
        probeParams = [nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE]
        self.mcADProbes = dict()
        self.mcSomaProbes = dict()
        for columnIdx, mcIdx in self.cxProbeIdxs["mc"]:
            assert columnIdx < self.numColumns
            assert mcIdx < self.numMCsPerColumn
            idx = columnIdx * self.numMCsPerColumn + mcIdx
            self.mcADProbes[(columnIdx, mcIdx)] = self.allMCADGroup[idx].probe(
                probeParams)
            self.mcSomaProbes[(columnIdx, mcIdx)] = self.allMCSomaGrp[idx].probe(
                probeParams)
        if sto:
            self.stoProbe = self.stoNeuronGroup[0].probe(probeParams)

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

