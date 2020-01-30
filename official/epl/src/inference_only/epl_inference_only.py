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
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pickle
import nxsdk_modules.epl.src.computeResults as computeResults
from collections import namedtuple


def timer(input_func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        print("{0} took {1:0.5f} secs".format(input_func.__name__,
                                              end_time - start_time))
        return result
    return timed


class MultiPatternInferenceEPL:

    def __init__(self, numCores, numExcNeuronsPerCore, numInhNeuronsPerCore,
                 inputBiases=None, gcInputBias=None,
                 conn_prob=0.2, delayMCToGC=16, numMCToGCDelays=4,
                 doOnlyInference=True, debug=False, log=True):
        self.net = nx.NxNet()
        self.numCores = numCores
        self.numExcNeuronsPerCore = numExcNeuronsPerCore
        self.numInhNeuronsPerCore = numInhNeuronsPerCore
        self.inputBiases = inputBiases
        self.gcInputBias = gcInputBias
        self.conn_prob = conn_prob
        self.numMCToGCDelays = numMCToGCDelays
        self.delayMCToGC = delayMCToGC
        self.stim2bias = [0, 34, 36, 38, 41, 43, 46, 50, 54, 59, 65, 72, 81,
                          92, 107, 129, 161, 214, 321, 641]
        self.cycleDuration = 40

        self.doOnlyInference = doOnlyInference
        self.debug = debug
        self.log = log
        self.numStepsRan = 0
        if not self.debug:
            self.setupNetwork()

    @property
    def numENeurons(self):
        return self.numCores * self.numExcNeuronsPerCore

    @property
    def numENeuronsPerCore(self):
        return self.numExcNeuronsPerCore

    @property
    def numINeurons(self):
        return self.numCores * self.numInhNeuronsPerCore

    @property
    def numINeuronsPerCore(self):
        return self.numInhNeuronsPerCore

    def setupNetwork(self):
        self.loadWeightsAndInputs()
        self.createMCAndSTONetwork()
        self.createMCToGCNetwork()
        self.setupProbes()

    @timer
    def createMCAndSTONetwork(self):
        self.createExcitatoryMCNeurons()
        self.createSTONeurons()
        self.connectSTONeuronsWithMCADNeurons()

    @timer
    def createMCToGCNetwork(self):
        self.createInhibitoryGCNeurons()
        self.connectInhibitoryGCToExcitatoryMCNeurons()
        self.connectExcitatoryMCToInhibitoryGCNeurons()

    @timer
    def loadWeightsAndInputs(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(dir_path, "../../data/")
        # print(data_dir)
        self.inhGCToExcMCWeights = np.load(os.path.join(data_dir,
                                                        "i2eWgtMat.npy"))

        self.inhGCToExcMCDelays = np.load(os.path.join(data_dir,
                                                       "i2eDlyMat.npy"))

        self.excMCToInhGCWeights = np.load(os.path.join(data_dir,
                                                        "e2iWgtMat.npy"))

        #print(os.path.join(data_dir, "windTunnelData.pi"))
        if not self.debug:
            windTunnelDataFile = "windTunnelData.pi"
            rf = open(os.path.join(data_dir, windTunnelDataFile), 'rb')
            self.trainingSet = pickle.load(rf)
            self.testSet = pickle.load(rf)
            rf.close()
        # print(self.inhGCToExcMCWeights.shape)
        # print(self.excMCToInhGCWeights.shape)

    def createInhibitoryGCNeurons(self):
        self.allGCNeuronsGroup = self.net.createCompartmentGroup()
        self.gcNeuronGrpPerCoreList = []

        if self.gcInputBias is None:
            self.gcInputBias = 0

        for coreIdx in range(self.numCores):
            gcNeuronGrpPerCore = self.net.createCompartmentGroup()
            gcNeuronProtoPerCore = nx.CompartmentPrototype(
                logicalCoreId=coreIdx,
                compartmentCurrentDecay=4095,
                compartmentVoltageDecay=4095,
                biasMant=0 if not self.debug else self.gcInputBias,
                vThMant=5*200 if not self.debug else self.gcInputBias//64,
                refractoryDelay=25,
                vMinExp=0,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )

            for i in range(self.numINeuronsPerCore):
                gcCx = self.net.createCompartment(
                    prototype=gcNeuronProtoPerCore)
                gcNeuronGrpPerCore.addCompartments(gcCx)
                self.allGCNeuronsGroup.addCompartments(gcCx)
            self.gcNeuronGrpPerCoreList.append(gcNeuronGrpPerCore)

    def connectInhibitoryGCToExcitatoryMCNeurons(self):
        ConnGroup = namedtuple("ConnGroup", "positive negative")
        self.inh2ExcConnGroups = list()

        for coreIdx in range(self.numCores):
            """
            wgts = np.zeros((self.numMCsPerCore, self.numGCsPerCore), int)
            delays = np.ones(wgts.shape)
            """
            if not self.debug:
                excWgts = self.inhGCToExcMCWeights[0, coreIdx]
                excDlys = self.inhGCToExcMCDelays[0, coreIdx]
                inhWgts = self.inhGCToExcMCWeights[1, coreIdx]
                inhDlys = self.inhGCToExcMCDelays[1, coreIdx]
            else:
                wgts = self.inhGCToExcMCWeights
                dlys = self.inhGCToExcMCDelays
                excWgts = np.ones_like(wgts[0, coreIdx])
                excDlys = np.ones_like(dlys[0, coreIdx]) * 2
                inhWgts = np.ones_like(wgts[1, coreIdx]) * -1
                inhDlys = np.ones_like(dlys[1, coreIdx]) * 1

            excConnProtoBox = nx.ConnectionPrototype(
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE
            )

            inhConnProtoBox = nx.ConnectionPrototype(
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE
            )

            posConnGrp = self.net.createConnectionGroup(
                src=self.gcNeuronGrpPerCoreList[coreIdx],
                dst=self.mcNeuronGrpPerCoreList[coreIdx],
                prototype=excConnProtoBox,
                connectionMask=(excWgts > 0),
                weight=excWgts,
                delay=excDlys
            )

            negConnGrp = self.net.createConnectionGroup(
                src=self.gcNeuronGrpPerCoreList[coreIdx],
                dst=self.mcNeuronGrpPerCoreList[coreIdx],
                prototype=inhConnProtoBox,
                connectionMask=(inhWgts < 0),
                weight=inhWgts,
                delay=inhDlys
            )
            self.inh2ExcConnGroups.append(ConnGroup(positive=posConnGrp,
                                                    negative=negConnGrp))

    def connectExcitatoryMCToInhibitoryGCNeurons(self):
        minDelay = self.delayMCToGC
        numDelays = self.numMCToGCDelays
        #percent = int(100 * self.conn_prob)

        """
        eSTDPLearningRule= net.createLearningRule(
                                            dw='2^-4*x1*y0',
                                            x1Impulse=20,
                                            x1TimeConstant=2,
                                            tEpoch=trainEpoch
                                            )
        """
        self.exc2InhConnGroups = list()
        for delay in range(minDelay, minDelay + numDelays):
            """
            wgtMat = np.zeros((self.numGCs, self.numMCs), int)
            rand = np.random.uniform(0, 100, size=wgtMat.shape)
            wgtMat[rand <= percent] = 10
            """
            wgtMat = self.excMCToInhGCWeights[delay-minDelay]

            connProtoE2I = nx.ConnectionPrototype(
                delay=delay if not self.debug else 0,
                numDelayBits=6,
                enableDelay=1,
                signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE,
                #enableLearning=1 if enableSTDP else 0,
                # learningRule=eSTDPLearningRule,
                # learningEnableMode=nx.SYNAPSE_LEARNING_ENABLE_MODE.SHARED
            )

            connGrp = self.net.createConnectionGroup(
                dst=self.allGCNeuronsGroup,
                src=self.allMCSomaNeuronsGrp,
                prototype=connProtoE2I,
                connectionMask=(wgtMat > 0),
                weight=wgtMat
            )

            self.exc2InhConnGroups.append(connGrp)

    def createExcitatoryMCNeurons(self):
        # Create MC-AD neurons recieve the input biases. The activity of
        # the MC-AD neurons is gated by the STO Neurons.
        if self.inputBiases is None:
            self.inputBiases = [0] * self.numCores

        self.mcADNeuronGroup = self.net.createCompartmentGroup()
        for coreIdx in range(self.numCores):
            mcADProto = nx.CompartmentPrototype(
                logicalCoreId=coreIdx,
                compartmentCurrentDecay=0,
                vThMant=10,  # i.e. 10 * 64 = 640
                biasMant=self.inputBiases[coreIdx],
                refractoryDelay=20,
                vMinExp=0,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )

            mcADCx = self.net.createCompartment(prototype=mcADProto)
            self.mcADNeuronGroup.addCompartments(mcADCx)

        # Create MC-Soma neurons which get input form MC-AD neurons. MC-Soma
        # neurons connect to the Inhibitory GC neurons.

        self.allMCSomaNeuronsGrp = self.net.createCompartmentGroup()
        self.mcNeuronGrpPerCoreList = []

        for coreIdx in range(self.numCores):
            mcSomaNeuronProto = nx.CompartmentPrototype(
                logicalCoreId=coreIdx,
                compartmentCurrentDecay=0,
                compartmentVoltageDecay=4095,
                vThMant=2,  # i.e. 2 * 64 = 128
                refractoryDelay=19,
                vMinExp=0,
                numDendriticAccumulators=64,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
            )
            mcNeuronGrpPerCore = self.net.createCompartmentGroup()
            for _ in range(self.numENeuronsPerCore):
                mcSomaNeuronCx = self.net.createCompartment(
                    prototype=mcSomaNeuronProto)
                self.allMCSomaNeuronsGrp.addCompartments(mcSomaNeuronCx)
                mcNeuronGrpPerCore.addCompartments(mcSomaNeuronCx)
            self.mcNeuronGrpPerCoreList.append(mcNeuronGrpPerCore)

        # Connect each MC-AD neuron to its MC-Soma neuron
        mcADToSomaConnProtoBox = nx.ConnectionPrototype(
            weight=3,
            delay=19,
            numDelayBits=6,
            enableDelay=1,
            signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
            postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE
        )

        for coreIdx in range(self.numENeurons):
            self.net._createConnection(src=self.mcADNeuronGroup[coreIdx],
                                       dst=self.allMCSomaNeuronsGrp[coreIdx],
                                       prototype=mcADToSomaConnProtoBox)

    def createSTONeurons(self):
        self.stoNeuronGroup = self.net.createCompartmentGroup()
        for i in range(self.numENeurons):
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

    def connectSTONeuronsWithMCADNeurons(self, wgt=20):
        connProtoBox = nx.ConnectionPrototype(
            weight=-wgt,
            delay=20,
            numDelayBits=6,
            enableDelay=1,
            signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
            postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.SPARSE
        )
        # stoNeuronGroup.connect(dst=eNeuronADGroup, prototype=connProtoBox)
        for coreIdx in range(self.numENeurons):
            self.net._createConnection(
                src=self.stoNeuronGroup[coreIdx],
                dst=self.mcADNeuronGroup[coreIdx],
                prototype=connProtoBox)

            for idx in range(self.numENeuronsPerCore):
                self.net._createConnection(
                    src=self.stoNeuronGroup[coreIdx],
                    dst=self.mcNeuronGrpPerCoreList[coreIdx][idx],
                    prototype=connProtoBox)

    def applyInputs(self, inputList, thethaReset=False):
        if len(inputList) != self.numENeurons:
            raise ValueError("Incorrect size of inputs list")

        if self.board is None:
            raise ValueError("There's no board as the network is not "
                             "compiled yet.")

        #uniqueCores = set()

        for mcIdx, inputVal in enumerate(inputList):
            cx = self.mcADNeuronGroup[mcIdx]
            _, chipId, coreId, cxId, _, _ = \
                self.net.resourceMap.compartment(cx.nodeId)
            n2Core = self.board.n2Chips[chipId].n2Cores[coreId]
            n2Core.cxCfg[np.asscalar(cxId)].bias = self.stim2bias[inputVal]
            n2Core.cxCfg.pushModified()
            if thethaReset:
                n2Core.cxState[np.asscalar(cxId)].v = 0
                n2Core.cxState.pushModified()

    def switchThetaState(self, state):
        for mcIdx in range(self.numCores):
            # MC soma
            cx = self.allMCSomaNeuronsGrp[mcIdx]
            _, chipId, coreId, cxId, _, vthProfileCfgId1 = \
                map(lambda x: int(x), self.net.resourceMap.compartment(cx.nodeId))
            n2Core = self.board.n2Chips[chipId].n2Cores[coreId]
            vth = 2 if state == 1 else 100
            n2Core.vthProfileCfg[vthProfileCfgId1].staticCfg.vth = vth
            n2Core.vthProfileCfg.pushModified()

    def sniff(self, inputList, numGammaCycles=5,
              numThetaCycles=1):
        self.applyInputs(inputList)
        numSteps = numGammaCycles * self.cycleDuration
        board.run(numSteps)
        self.applyInputs([0] * self.numCores, thethaReset=True)
        self.switchThetaState(state=0)
        # numSteps = numGammaCycles * self.cycleDuration
        board.run(numSteps)
        self.switchThetaState(state=1)
        self.numStepsRan += 2 * numSteps

    def dumpSpikesOutputForPostProcessing(self, nGamma):
        _, spikeProbes, _ = self.mcSomaProbes

        offset = 20 + 1  # 1 accounts the delay in spike probe counter

        gammaCode = []
        for _ in range(nGamma):
            gammaCode.append([0]*72)

        for i, spkProbe in enumerate(spikeProbes):
            data = spkProbe.data[offset:]
            spikes1 = np.nonzero(data)[0]
            for j in spikes1:
                gammaCycle = j//40
                rank = (gammaCycle*40 + 21) - (gammaCycle*40 + (j % 40))
                gammaCode[gammaCycle][i] = rank

        pickledfilename = "spikes.pi"
        wf = open(pickledfilename, 'wb')
        pickle.dump(gammaCode, wf)
        wf.close()

    @timer
    def setupProbes(self):
        self.setupMCAndSTOProbes()

    def setupMCAndSTOProbes(self):
        probeParams = [nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE,
                       nx.ProbeParameter.COMPARTMENT_CURRENT]

        self.mcADProbes = self.mcADNeuronGroup.probe(probeParams)
        self.mcSomaProbes = self.allMCSomaNeuronsGrp.probe(probeParams)
        self.stoProbes = self.stoNeuronGroup.probe(probeParams)

    def getProbesForNeuronIdx(self, probes, idx):
        vProbes, spikeProbes, uProbes = probes
        return vProbes[idx], spikeProbes[idx], uProbes[idx]

    def plotSTOVsMCNeuronProbes(self, idx):

        # plot the eNeuronProbes
        vProbeE, spikeProbeE, uProbeE = self.getProbesForNeuronIdx(
            self.mcSomaProbes, idx)
        vProbeSTO, spikeProbeSTO, uProbeSTO = self.getProbesForNeuronIdx(
            self.stoProbes, idx)

        plt.figure()
        ax1 = plt.subplot(321)
        vProbeE.plot()
        plt.title("E-NEURON(V_PROBE)")

        plt.subplot(323, sharex=ax1)
        spikeProbeE.plot()
        plt.title("E-NEURON(SPIKE_PROBE)")

        plt.subplot(325, sharex=ax1)
        uProbeE.plot()
        plt.title("E-NEURON(U_PROBE)")

        # plots for STO neurons
        plt.subplot(322, sharex=ax1)
        vProbeSTO.plot()
        plt.title("STO-NEURON(V_PROBE)")

        plt.subplot(324, sharex=ax1)
        spikeProbeSTO.plot()
        plt.title("STO-NEURON(SPIKE_PROBE)")

        plt.subplot(326, sharex=ax1)
        uProbeSTO.plot()
        plt.title("E-NEURON(U_PROBE)")

    def plotSpikeRaster(self, probes, offset=60):
        _, spikeProbes, _ = probes
        plt.figure()
        # probe[1] is spike probe
        data = [np.nonzero(spkProbe.data[offset:])[0]
                for spkProbe in spikeProbes]
        size = self.numENeurons

        plt.eventplot(positions=data, colors=[(1, 0, 0)],
                      lineoffsets=np.arange(size),
                      linelengths=np.ones(size) / 2.0)
        plt.title("E-Neurons (Spike Raster Plot)")
        plt.ylabel("# E-Neurons")
        plt.xlabel("Time + {} timesteps".format(offset))
        plt.tight_layout()

    @timer
    def compileAndGetBoard(self):
        self.board = nx.N2Compiler().compile(self.net)
        return self.board


if __name__ == '__main__':
    numCores = 72
    eplInference = MultiPatternInferenceEPL(numCores=numCores,
                                            numExcNeuronsPerCore=1,
                                            numInhNeuronsPerCore=46)
    board = eplInference.compileAndGetBoard()

    for i, trainSample in enumerate(eplInference.trainingSet):
        for _ in range(2):
            eplInference.sniff(inputList=trainSample)

    for i, testSample in enumerate(eplInference.testSet):
        eplInference.sniff(inputList=testSample)
    print("Ran the network for {} time steps".format(eplInference.numStepsRan))

    board.disconnect()

    nGamma = 10*len(eplInference.trainingSet)*2 + 10*len(eplInference.testSet)
    eplInference.dumpSpikesOutputForPostProcessing(nGamma)
    computeResults.computeResults(nGammaPerTraining=10,
                                  trainingSetSize=len(
                                      eplInference.trainingSet),
                                  testSetSize=len(eplInference.testSet),
                                  plotIDs=[0])
