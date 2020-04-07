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

from official.epl.src.single_pattern_learning.epl_snips_utils import \
    EplWithSNIPs
from official.epl.src.epl_utils import EplUtils
from official.epl.src.single_pattern_learning.epl_parameters import \
    ParamemtersForEPL


class EPLSinglePatternLearning(EplUtils, EplWithSNIPs):
    """
    Create an EPL network to learn one pattern (odor in case of olfactory
    sensor data) and test if it can recall the learned pattern when presented
    with noise corrupted test samples of the same pattern
    """

    def __init__(self, eplParams, verbose=False):
        self.executionTime = None
        self.numStepsRan = None
        self.trainingSet = None
        self.testingSet = None
        self.verbose = verbose

        super().__init__(eplParams=eplParams)
        self.setupNetwork()
        self.setupMCSomaProbes()
        # if not self.disableProbes:
        #     self.setupCxProbes()
        self.board = self.compileAndGetBoard()

    @property
    def _numTestSamples(self):
        """ returns the number of noise corrupted test samples presented to
        the network"""
        return len(self.testingSet)

    @property
    def _numLabelSamples(self):
        """ returns the number of noise free samples presented to the
        network. Since this network learns only 1 pattern, therefore
        numLabelSamples is 1
        """
        return len(self.trainingSet)

    @property
    def _numTotalTestSamples(self):
        """ returns the total number of the samples presented to the network
        in the test phase. The first sample is always the learned pattern (
        labeling phase). Thus labeling is testing without any noise corruption
        i.e. the trained pattern.
        Next in the actual test phase, we present various noise corrupted
        versions of the trained pattern to see how well the network can
        recall them"""
        return self._numTestSamples + self._numLabelSamples

    @property
    def _trainDuration(self):
        """ returns the number of algorithmic timesteps for which the network
        is trained
        """
        gamma = self.gammaCycleDuration
        positiveThetaPeriod = self.numGammaCyclesTrain * gamma
        negativeThetaPeriod = self.numGammaCyclesIdle * gamma
        trainingDuration = positiveThetaPeriod + negativeThetaPeriod
        return trainingDuration

    @property
    def _testDuration(self):
        """ returns the number of algorithmic timesteps for which the network
        is presented test samples
        """
        gamma = self.gammaCycleDuration
        negativeThetaPeriod = self.numGammaCyclesIdle * gamma
        testDuration = self.numGammaCyclesTest * gamma
        testDuration += negativeThetaPeriod
        testDuration *= self._numTotalTestSamples
        return testDuration

    @property
    def _numStepsToRun(self):
        """returns the total number of algorithm timesteps for which the
        network runs for
        """
        return self._trainDuration + self._testDuration

    def _readSpikeCounterData(self):
        print("processing LMT spike counter data...takes few seconds...")
        cxIdToTimeMap = {cx: [] for cx in range(self.numMCs)}
        while True:
            t = self.mgmtChannel3.read(1)[0]
            cx = -1
            while True:
                cx = self.mgmtChannel3.read(1)[0]
                if cx >= self._numStepsToRun + 10:
                    break
                else:
                    cxIdToTimeMap[cx].append(t)
            if cx == self._numStepsToRun + 11:
                break
        return cxIdToTimeMap

    def genTrainingData(self, numOdors=1):
        """generates a synthetic training dataset of odor sensor readings"""
        if numOdors > 1:
            raise ValueError("Can't learn more than 1 odor for now...")
        return self.generateTrainingData(numOdors=numOdors,
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
        self.setupSNIPs(self._numTotalTestSamples, self._numStepsToRun)
        self.board.startDriver()
        self.sendDataToSNIP()
        self.board.run(self._trainDuration, aSync=True)
        FINISHED_TRAINING = 1
        status = self.mgmtChannel4.read(1)[0]
        if self.verbose and status == FINISHED_TRAINING:
            print("EPL NETWORK: COMPLETED TRAINING...")

    def predict(self):
        """tests the network with the given testing set"""
        self.board.run(self._testDuration, aSync=True)
        if self.useLMTSpikeCounters:
            self.cxIdToSpikeTimeMap = self._readSpikeCounterData()
        self.board.finishRun()
        self.board.disconnect()
        if self.verbose:
            print("EPL NETWORK: COMPLETED TESTING...")

    def evaluate(self, plotIds=None):
        """evaluates the performance of the network"""
        if plotIds is None:
            plotIds = []

        spikesData = self.cxIdToSpikeTimeMap if self.useLMTSpikeCounters else \
            self.allMCSomaProbes

        gammaCode = self.spikesDataToGammaCode(
            spikesData=spikesData,
            numStepsRan=self._numStepsToRun,
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
            nsensors=self.numMCs,
            verbose=False)
        return accuracy_percent


if __name__ == '__main__':
    # import os
    # os.environ["PARTITION"] = "wm_perf"
    eplParams = ParamemtersForEPL()
    eplParams.numColumns = 72
    eplParams.numMCsPerColumn = 1
    eplParams.numGCsPerColumn = 3
    eplParams.connProbMCToGC = 0.2
    eplParams.numDelaysMCToGC = 1
    eplParams.useRandomSeed = True
    eplParams.randomGenSeed = 100
    eplParams.useLMTSpikeCounters = False
    eplParams.logSNIPs = False
    eplParams.executionTimeProbe = False
    eplParams.numGammaCyclesTrain = 45
    numTestSamples = 5

    epl = EPLSinglePatternLearning(eplParams=eplParams, verbose=True)

    x = epl.genTrainingData()
    y = epl.genTestingData(trainingData=x,
                           numTestSamples=numTestSamples,
                           occlusionFactor=0.5)
    epl.fit(trainingSet=x, testingSet=y)
    epl.predict()
    epl.evaluate()
