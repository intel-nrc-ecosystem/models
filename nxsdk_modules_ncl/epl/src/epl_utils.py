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

import random
import numpy as np
import pickle
import nxsdk_modules_ncl.epl.src.computeResults as compResults


class EplUtils:
    """ implements utility functions to: (1) generate synthetic training and
    testing data, (2) post process the MC output spikes data (3)evaluate the
    recall performance of the EPL network for the test samples"""
    @staticmethod
    def generateTrainingData(numOdors=1, numSensors=72, sparsity=0.5, offset=0):
        """ generate synthetic training data for EPL network"""
        trainingData = []
        for i in range(0, numOdors):
            trainingData.append([])
            for j in range(0, numSensors):
                if random.random() > sparsity:
                    trainingData[i].append(4 * random.randint(1, 4) - offset)
                else:
                    trainingData[i].append(0)
        return trainingData

    @staticmethod
    def generateTestingData(trainingData, occlusionPercent, numTestSamples,
                            sensorDynamicRange=16):
        """ generate synthetic testing data for EPL network"""
        data = trainingData
        p = occlusionPercent
        n = numTestSamples
        occludedData = []
        nsensors = len(data[0])

        for i in range(0, len(data)):
            ndim = len(data[i])  # dimension of data
            for j in range(0, n):
                occludedData.append([])
                affected_ids = random.sample(range(ndim), int(p * ndim))
                for k in range(0, ndim):
                    if k in affected_ids:
                        occludedData[i * n + j].append(
                            random.randint(0, sensorDynamicRange))
                    else:
                        occludedData[i * n + j].append(data[i][k])
        return occludedData

    @staticmethod
    def spikesDataToGammaCode(spikesData, numStepsRan,
                              cycleDuration, offset=60,
                              numSensors=72, verbose=False,
                              useLMTCounters=False, dumpToFile=True):
        """converts MC output spikes data to gamma code"""
        nGamma = numStepsRan // cycleDuration
        offset = 0 #60
        gammaCode = []
        for _ in range(nGamma):
            gammaCode.append([0] * numSensors)

        for i in range(numSensors):
            if useLMTCounters:
                spikes1 = spikesData[i]
            else:
                spkProbe = spikesData[i]
                data = spkProbe.data
                spikes1 = np.nonzero(data)[0]
            for j in spikes1:
                # if i==0: print(j);
                spikeLatency = j - offset + 1
                gammaCycle = spikeLatency // cycleDuration
                # ToDo: replace 21 with a parameter
                rank = (gammaCycle * cycleDuration + 21) - \
                       (gammaCycle * cycleDuration +
                        (spikeLatency % cycleDuration))
                gammaCode[gammaCycle][i] = rank

        if dumpToFile:
            pickledfilename = 'spikes.pi'
            wf = open(pickledfilename, 'wb')
            pickle.dump(gammaCode, wf)
            wf.close()
        if verbose:
            for i in range(len(gammaCode)):
                print(gammaCode[i])
        return gammaCode

    @staticmethod
    def computeResults(gammaCode, nGammaPerTraining, trainingSetSize,
                       testSetSize, nsensors, similarityThreshold=0.75,
                       verbose=False):
        """ evaluate the EPL network performance for the test samples """
        compResults.computeResults(nGammaPerTraining, trainingSetSize,
                                   testSetSize, gammaCode=gammaCode,
                                   verbose=verbose, nsensors=nsensors,
                                   similarityThreshold=similarityThreshold)
