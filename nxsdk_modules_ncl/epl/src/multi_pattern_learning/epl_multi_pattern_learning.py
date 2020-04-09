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

"""
EPL network for learning multiple patterns
"""

import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
from nxsdk_modules_ncl.epl.src.multi_pattern_learning.epl_snips_utils import \
    EplWithSNIPs
from nxsdk_modules_ncl.epl.src.multi_pattern_learning.epl_parameters import \
    ParamemtersForEPL
from nxsdk_modules_ncl.epl.src.epl_utils import EplUtils
from matplotlib.lines import Line2D


class EPLMultiPatternLearning(EplUtils, EplWithSNIPs):
    """
    Create an EPL network to learn multiple patterns and test if the network
    can recall the learned patterns even when noise corrupted test samples of
    the learned patterns are presented
    """

    def __init__(self, eplParams):
        """ initialize the EPL network """
        super().__init__(eplParams=eplParams)
        self.setupNetwork()
        self.setupMCSomaProbes()
        self.board = self.compileAndGetBoard()
        self.numStepsRan = None
        self.trainingSet = None
        self.testingSet = None
        self.cxIdToSpikeTimeMap = None

    @property
    def numTrainSamples(self):
        """ returns the number of training patterns presented to the network"""
        return len(self.trainingSet)

    @property
    def numTestSamples(self):
        """ returns the number of noise corrupted test samples for
        each pattern presented to the network"""
        return len(self.testingSet)

    @property
    def numLabelSamples(self):
        """ returns the number of patterns presented in the labeling phase.
        In the labeling phase, the learned patters are presented as test
        samples"""
        return len(self.trainingSet)

    @property
    def numTotalTestSamples(self):
        """ returns the total number of the samples presented to the network
        in test phase. The first sample is always the learned odor (labeling
        phase). Thus labeling is testing without any noise corruption i.e.
        the trained odor. Next in the actual test phase, we present various
        noise corrupted versions of the trained odor"""
        return self.numTestSamples + self.numLabelSamples

    @property
    def trainDuration(self):
        """ returns the total number of timesteps in the train phase """
        gamma = self.gammaCycleDuration
        positiveThetaPeriod = self.numGammaCyclesTrain * gamma
        negativeThetaPeriod = self.numGammaCyclesIdle * gamma
        trainingDuration = positiveThetaPeriod + negativeThetaPeriod
        trainingDuration *= self.numTrainSamples
        return trainingDuration

    @property
    def testDuration(self):
        """ returns the total number of timesteps in the test phase """
        gamma = self.gammaCycleDuration
        negativeThetaPeriod = self.numGammaCyclesIdle * gamma
        testDuration = self.numGammaCyclesTest * gamma
        testDuration += negativeThetaPeriod
        testDuration *= self.numTotalTestSamples
        return testDuration

    @property
    def numStepsToRun(self):
        """ returns the total number of steps for which the network will run"""
        return self.trainDuration + self.testDuration

    def _readSpikeCounterData(self):
        """ post process the LMT spike counter data sent from the chip """
        # ToDo: remove magic numbers, use logger
        print("processing LMT spike counter data...takes time...")
        cxIdToTimeMap = dict()
        for cx in range(self.numMCs):
            cxIdToTimeMap[cx] = list()

        while True:
            t = self.spkCounterMgmtChannel.read(1)[0]
            #print("Processing spike at t={}".format(t))
            cx = -1
            while True:
                cx = self.spkCounterMgmtChannel.read(1)[0]
                #print("Processing for cx={}".format(cx))
                if cx >= self.numStepsToRun + 10:
                    break
                else:
                    cxIdToTimeMap[cx].append(t)
            if cx == self.numStepsToRun + 11:
                break
        # pprint.pprint(cxIdToTimeMap)
        return cxIdToTimeMap

    def genTrainingData(self, numPatterns=1):
        """generates a synthetic training dataset of odor sensor readings"""
        return self.generateTrainingData(numOdors=numPatterns,
                                         numSensors=self.numMCs)

    def genTestingData(self, trainingData, numTestSamples=10,
                       occlusionFactor=0.5):
        """generates a synthetic testing dataset of noise corrupted
        versions of odor sensor readings"""
        return self.generateTestingData(trainingData=trainingData,
                                        occlusionPercent=occlusionFactor,
                                        numTestSamples=numTestSamples)

    def fit(self, trainingSet, testingSet):
        """trains the network with the given training set"""
        self.trainingSet = trainingSet
        self.testingSet = testingSet
        self.gatherDataForSNIPs(trainingSet, testingSet)
        self.addSNIPs(totalTrainSamples=self.numTrainSamples,
                      totalTestSamples=self.numTotalTestSamples,
                      totalTimeSteps=self.numStepsToRun)

        self.board.startDriver()
        self.sendDataToSNIP()
        self.board.run(self.trainDuration, aSync=True)
        # SNIP halts after training; must do this read to proceed to testing
        status = self.boardStatusMgmtChannel.read(1)[0]
        self.board.finishRun()

    def predict(self):
        """tests the network with the given testing set"""
        self.board.run(self.testDuration, aSync=True)
        if self.useLMTSpikeCounters:
            self.cxIdToSpikeTimeMap = self._readSpikeCounterData()
        self.board.finishRun()
        self.board.disconnect()

    def evaluate(self, verbose=False, similarityThreshold=0.75):
        """evaluates the performance of the network"""

        spikesData = self.cxIdToSpikeTimeMap if self.useLMTSpikeCounters else \
            self.allMCSomaProbes
        gammaCode = self.spikesDataToGammaCode(
            spikesData=spikesData,
            numStepsRan=self.numStepsToRun,
            numSensors=self.numMCs,
            cycleDuration=self.gammaCycleDuration,
            useLMTCounters=self.useLMTSpikeCounters,
            dumpToFile=False)

        nGammaTrain = self.numGammaCyclesTrain + self.numGammaCyclesIdle
        accuracy_percent = self.computeResults(
            gammaCode=gammaCode,
            nGammaPerTraining=nGammaTrain,
            trainingSetSize=len(self.trainingSet),
            testSetSize=len(self.testingSet),
            verbose=verbose, nsensors=self.numMCs,
            similarityThreshold=similarityThreshold)
        return accuracy_percent

    def showRasterPlot(self, patternIdx, sampleIdx):
        """ displays the MC output spike raster during the test phase for a
        particular test sample (sampleIdx) belonging to a particular pattern(
        patternIdx)"""
        probes = self.allMCSomaProbes
        numGammaPattern = self.numGammaCyclesTest + self.numGammaCyclesIdle
        testDuration = self.gammaCycleDuration * numGammaPattern

        beginLabel = self.trainDuration + (patternIdx * testDuration)
        endLabel = beginLabel + testDuration
        dataLabel = [np.nonzero(probe.data[beginLabel:endLabel])[0]
                            for probe in  probes]

        beginSample = self.trainDuration + (self.numPatterns * testDuration)
        beginSample += (patternIdx * self.numTestSamples + sampleIdx) * \
                       testDuration
        endSample = beginSample + testDuration
        dataSample = [np.nonzero(probe.data[beginSample:endSample])[0]
                     for probe in probes]

        size = self.numMCs
        fig, _ = plt.subplots()
        plt.eventplot(positions=dataLabel, colors='blue',
                      lineoffsets=np.arange(size),
                      linelengths=0.8)
        plt.eventplot(positions=dataSample, colors='red',
                      lineoffsets=np.arange(size),
                      linelengths=0.5)
        plt.title("""MC Output Spike Raster (patternIdx={}, sampleIdx={})
                    """.format(patternIdx, sampleIdx))
        plt.ylabel("#MC Neurons")
        plt.xlabel("""Time ({} gamma cycles; gamma cycle={} timesteps)
                """.format(self.numGammaCyclesTest, self.gammaCycleDuration))
        xticks = [self.gammaCycleDuration * i for i in
                                range(self.numGammaCyclesTest)]
        plt.xticks(xticks)
        legend_elements = [
            Line2D([0], [0], color='blue', lw=1, label='Trained Pattern'),
            Line2D([1], [0], color='red', lw=1, label='Test Pattern')
            ]
        fig.legend(handles=legend_elements, loc='center right')
        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        plt.show()


def test1(numTestSamples, useLMTSpkCtr=False):
    """Test something"""
    #os.environ["PARTITION"] = "wm_perf"
    # user specifies the dimensions, compute the cores internally
    eplParams = ParamemtersForEPL()
    eplParams.numPatterns = 5
    eplParams.numColumns = 72
    eplParams.numMCsPerColumn = 1
    eplParams.numGCsPerPatternPerColumn = 5
    eplParams.connProbMCToGC = 0.2
    eplParams.numDelaysMCToGC = 1
    eplParams.useRandomSeed = True
    eplParams.randomGenSeed = 100
    eplParams.useLMTSpikeCounters = useLMTSpkCtr
    eplParams.logSNIPs = False
    eplParams.executionTimeProbe = False
    eplParams.numGammaCyclesTrain = 45

    epl = EPLMultiPatternLearning(eplParams=eplParams)

    x = epl.genTrainingData(numPatterns=eplParams.numPatterns)
    y = epl.genTestingData(trainingData=x,
                           numTestSamples=numTestSamples,
                           occlusionFactor=0.5)
    epl.fit(trainingSet=x, testingSet=y)
    epl.predict()
    epl.evaluate(verbose=True)
    epl.showRasterPlot(patternIdx=0, sampleIdx=2)


if __name__ == '__main__':
    test1(numTestSamples=5, useLMTSpkCtr=False)
    #test2()
