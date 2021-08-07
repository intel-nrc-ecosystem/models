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

# -----------------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------------

import unittest
from test import support

import matplotlib.pyplot as plt
import numpy as np

import nxsdk.api.n2a as nx
from nxsdk_modules.lsnn.src.lsnn import LsnnNet, ModelParams, CompileParams


class TestLSNN(unittest.TestCase):
    """Unit tests for lsnn module."""

    def create_lsnn(self, numInput, numRegular, numAdaptive, numOutput, wIn,
                    wRec, wOut):
        """Creates a standard LsnnNet with a spike generator as input group."""

        # define parameters for LsnnNet
        modelP = ModelParams(numRegular, numAdaptive, wIn, wRec,
                                   wOut)
        compileP = CompileParams()
        # need to create NxNet object to set up spikegenerator for input
        compileP.net = nx.NxNet()
        inputGroup = compileP.net.createSpikeGenProcess(numPorts=numInput)

        # configure output neuron group
        outMantissa = 2 ** 17 - 1
        cxProto = nx.CompartmentPrototype(
            logicalCoreId=compileP.logicalCoreId,
            vThMant=outMantissa,
            compartmentCurrentDecay=modelP.decayU,
            compartmentVoltageDecay=modelP.decayV,
            refractoryDelay=modelP.refrac,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            numDendriticAccumulators=16
        )
        outputGroup = compileP.net.createCompartmentGroup(size=numOutput, prototype=cxProto)

        lsnn = LsnnNet(inputGroup, outputGroup, modelP, compileP)

        return lsnn


    def check_weights(self, w):
        """Check the weights."""
        R, C = w.shape
        #r, c = self.rand_int(R), self.rand_int(C)
        for r in range(R):
            for c in range(C):
                assert(w[r,c] == 10*(r+1) + c)


    def rand_int(self, max):
        """Create random integer."""
        return np.random.randint(0, max)


    def test_weight_matrices(self):
        """Check if the weight matrices stored correctly."""
        numInput=2
        numRegular=3
        numAdaptive=4
        numOutput=5

        wIn = np.array([[10, 11],
                             [20, 21],
                             [30, 31],
                             [40, 41],
                             [50, 51],
                             [60, 61],
                             [70, 71]])

        wRec = np.array([[10, 11, 12, 13, 14, 15, 16],
                             [20, 21, 22, 23, 24, 25, 26],
                             [30, 31, 32, 33, 34, 35, 36],
                             [40, 41, 42, 43, 44, 45, 46],
                             [50, 51, 52, 53, 54, 55, 56],
                             [60, 61, 62, 63, 64, 65, 66],
                             [70, 71, 72, 73, 74, 75, 76]])

        wOut = np.array([[10, 11, 12, 13, 14, 15, 16],
                             [20, 21, 22, 23, 24, 25, 26],
                             [30, 31, 32, 33, 34, 35, 36],
                             [40, 41, 42, 43, 44, 45, 46],
                             [50, 51, 52, 53, 54, 55, 56]])
        lsnn = self.create_lsnn(numInput=numInput, numRegular=numRegular,
                                numAdaptive=numAdaptive, numOutput=numOutput,
                                wIn=wIn, wRec=wRec, wOut=wOut)
        self.check_weights(lsnn.modelParams.wIn)
        self.check_weights(lsnn.modelParams.wOut)
        self.check_weights(lsnn.modelParams.wRec)
        self.check_matrix_equality(lsnn.connRegularToRegularNeurons,
                          wRec[:numRegular, :numRegular])
        
        self.check_matrix_equality(lsnn.connRegularToAdapativeNeurons,
                          wRec[numRegular:, :numRegular])

        self.check_matrix_equality(lsnn.connAdaptiveToAdapativeNeurons,
                          wRec[numRegular:, numRegular:])

        self.check_matrix_equality(lsnn.connAdaptiveToRegularNeurons,
                          wRec[:numRegular, numRegular:])
        

        self.check_matrix_equality(lsnn.connRegularToOutputNeurons,
                          wOut[:, :numRegular])

        self.check_matrix_equality(lsnn.connAdaptiveToOutputNeurons,
                          wOut[:, numRegular:])

        lsnn.generateNetwork()
        lsnn.run(1)
        lsnn.finish()

    def check_matrix_equality(self, conn, w):
        """Check if the weight matrix is equal."""
        pos = conn.positive
        neg = conn.negative
        wPos = pos.getConnectionState('weight')
        wNeg = neg.getConnectionState('weight')
        ww = wPos + wNeg
        self.assertEqual(np.array_equal(ww, w), True)


    def test_connectivity(self):
        """Check if the connectivity is set correctly."""
        numInput=2
        numRegular=4
        numAdaptive=6
        numOutput=3

        numRecurrent = numRegular + numAdaptive

        wIn = np.zeros((numRecurrent, numInput), int)
        wOut = np.ones((numOutput, numRecurrent), int)
        wRec = np.zeros((numRecurrent, numRecurrent), int)

        wRR=25
        wRA=15
        wAA=10
        wAR=20
        wRO=45
        wAO=55

        #only adaptive to adaptive connections. rest all are 0
        wRec[:numRegular, :numRegular] = wRR
        wRec[numRegular:, :numRegular] = wRA
        wRec[numRegular:, numRegular:] = wAA
        wRec[:numRegular, numRegular:] = wAR

        wOut[:, :numRegular] = wRO
        wOut[:,numRegular:] = wAO

        lsnn = self.create_lsnn(numInput=numInput, numRegular=numRegular,
                                numAdaptive=numAdaptive, numOutput=numOutput,
                                wIn=wIn, wRec=wRec, wOut=wOut)

        self.verify_connection_weights(lsnn, wRec, wOut)
        lsnn.generateNetwork()
        lsnn.run(10)
        lsnn.finish()


    def plotProbe(self, probes, title):
        """Plot information from probes."""
        #fig = plt.figure()
        n = len(probes)

        if n > 10:
            step = n//5
            n = 5
        else:
            step = 1

        fig, ax = plt.subplots(3, n)
        idx = 1
        #for prb in probes:
        for i in range(0, n * step, step):
            #prb = probes[i]
            uProbe, vProbe, spikeProbe = tuple(probes)
            #plots for E-Neurons
            ax1 = plt.subplot(3,n,idx)
            vProbe.plot()
            plt.title("V_" + str(idx))

            plt.subplot(3,n,n+idx, sharex=ax1)
            spikeProbe.plot()
            plt.title("SPIKE_" + str(idx))

            plt.subplot(3,n,2*n+idx, sharex=ax1)
            uProbe.plot()
            plt.title("U_" + str(idx))
            idx += 1

        fig.suptitle(title, size=16)
        fig.text(0.5, 0.04, 'Neuron #', ha='center')
        fig.text(0.04, 0.5, 'Probe Parameter', va='center',
                 rotation='vertical')
        fig.tight_layout()

    def verify_connection_weights(self, lsnn, wRec, wOut):
        """Check if the connection weights are correct."""
        numRegular = lsnn.modelParams.numRegular

        self.check_matrix_equality(lsnn.connRegularToRegularNeurons,
                          wRec[:numRegular, :numRegular])
        self.check_matrix_equality(lsnn.connRegularToAdapativeNeurons,
                          wRec[numRegular:, :numRegular])
        self.check_matrix_equality(lsnn.connAdaptiveToAdapativeNeurons,
                          wRec[numRegular:, numRegular:])
        self.check_matrix_equality(lsnn.connAdaptiveToRegularNeurons,
                          wRec[:numRegular, numRegular:])
        self.check_matrix_equality(lsnn.connRegularToOutputNeurons,
                          wOut[:, :numRegular])
        self.check_matrix_equality(lsnn.connAdaptiveToOutputNeurons,
                          wOut[:, numRegular:])
        print("######## verified connection weights")

    @unittest.skip
    def test_spike_injection(self):
        """Check the spike injection."""
        numInput = 5
        numRegular = 10
        numAdaptive = 20
        numOutput = 1

        numRecurrent = numRegular + numAdaptive
        nsteps = 5 * numInput

        wIn = np.ones((numRecurrent, numInput), int)
        for i in range(numInput):
            wIn[:, i] *= 5*i+5


        wOut = np.ones((numOutput, numRecurrent), int)
        wRec = np.zeros((numRecurrent, numRecurrent), int)

        lsnn = self.create_lsnn(numInput=numInput, numRegular=numRegular,
                                numAdaptive=numAdaptive, numOutput=numOutput,
                                wIn=wIn, wRec=wRec, wOut=wOut)

        spikeTimes = []
        for i in range(numInput):
            spikeTimes.append(list(np.arange(1 + i, nsteps, numInput)))
            lsnn.inputGroup.addSpikes(spikeInputPortNodeIds=i,
                                          spikeTimes=spikeTimes[i])

        lsnn.probeAll()
        lsnn.generateNetwork()
        lsnn.run(nsteps)
        lsnn.finish()
        #self.plotProbe(lsnn.adaptiveMainProbes, "Adaptive Main")
        #self.plotProbe(lsnn.regularNeuronProbes, "Regular")
        #plt.show()

        uProbe = lsnn.regularNeuronProbes[0][0]
        for i in range(numInput):
            for t in spikeTimes[i]:
                self.assertEqual(uProbe.data[t], 2**6 * wIn[0, i])

    @unittest.skip
    def test_network_sweep(self):
        """Check if the network configuration works."""
        numInput = 5
        numOutput = 5

        numRegular = 500
        numAdaptive = (512 - (numRegular + numOutput))//2
        self.run_spike_injection(numInput, numRegular, numAdaptive, numOutput)


    def run_spike_injection(self, numInput, numRegular, numAdaptive,
                            numOutput):
        """Helper function for spike injection."""
        numSteps = 10
        numRecurrent = numRegular + numAdaptive
        wIn = np.ones((numRecurrent, numInput), int) * 10
        wOut = np.zeros((numOutput, numRecurrent), int)
        wRec = np.zeros((numRecurrent, numRecurrent), int)

        prob = 0.05

        if numRegular >= 600:
            prob = 0.03
        if numRegular >=800:
            prob = 0.01

        w = 5
        self.connect_with_prob(wRec, w)
        self.connect_with_prob(wOut, w)



        lsnn = self.create_lsnn(numInput=numInput, numRegular=numRegular,
                                numAdaptive=numAdaptive, numOutput=numOutput,
                                wIn=wIn, wRec=wRec, wOut=wOut)

        lsnn.spikeTimes = []
        for i in range(numInput):
            lsnn.spikeTimes.append(list(np.arange(1 + i, numSteps, numInput)))
            lsnn.inputGroup.addSpikes(spikeInputPortNodeIds=i,
                                          spikeTimes=lsnn.spikeTimes[i])

        lsnn.probeAll()
        lsnn.generateNetwork()
        lsnn.run(numSteps)
        lsnn.finish()

        self.check_deltaU(lsnn.regularNeuronProbes, [0, 495, 0])
        self.check_deltaU(lsnn.adaptiveMainProbes, [0, 1983, 0])
        self.check_deltaU(lsnn.outputProbes, [0, 0, 0])


    def connect_with_prob(self, wgtMat, wgt, prob=0.03):
        """Helper function for configuring connections with probes."""
        isInRange = True if (0.0 <= prob and prob <=1.0) else \
            False
        self.assertEqual(isInRange, True)
        percent = int(100 * prob)
        rand = np.random.uniform(-100, 100, size=wgtMat.shape)
        wgtMat[np.logical_and(rand <= percent, rand > 0)] = wgt
        wgtMat[np.logical_and(rand >= -percent, rand < 0)] = -wgt


    def check_deltaU(self, probes, deltaUExpected):
        """Helper function to compare currents."""
        for i, prb in enumerate(probes):
            uPrb = prb[0]
            deltaU = uPrb.data[6]-uPrb.data[5]
            self.assertEqual(deltaU, deltaUExpected[i])


if __name__ == '__main__':
    support.run_unittest(TestLSNN)

