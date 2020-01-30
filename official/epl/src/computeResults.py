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

import matplotlib.pyplot as plt
import math
import pickle
import numpy as np


def dot(l1=[], l2=[]):
    """ computes dot product of 2 vectors"""
    sum1 = 0
    for i in range(0, len(l1)):
        sum1 = (l1[i] * l2[i]) + sum1
    return sum1


# Function to compute 2-norm of a list
def norm(l1=[]):
    """ computes norm of 2 vectors"""
    d = 0
    for i in range(0, len(l1)):
        d = d + ((l1[i]) ** 2)
    return math.sqrt(d)


# Function to compute cosine similarity index of two lists
def cosine_similarity(l1=[], l2=[]):
    """ computes cosine similarity"""
    dot_product = dot(l1, l2)
    norm_a = norm(l1)
    norm_b = norm(l2)
    denom = norm_a * norm_b
    if (denom != 0):
        out = round(float(dot_product) / (denom), 4)
    else:
        out = 0
    return out


def hammingSimilarity(l1=[], l2=[]):
    """ computes hamming similarity """
    hammingD = 0
    nsensors = len(l1)  # using the total number of non-zero sensors per odor
    nNonZero = len(l1)
    # nsensors = len(l1)
    for i in range(0, nsensors):
        if l1[i] != l2[i]:
            hammingD += 1
        if l1[i] == 0 and l2[i] == 0:
            nNonZero = nNonZero - 1
    # ratio = float(hammingD)/nNonZero
    ratio = float(hammingD) / nsensors
    hammingS = round(1 - ratio, 2)
    return hammingS


def jaccardSimilarity(l1=[], l2=[]):
    """ computes Jaccard similarity"""
    list1 = []
    list2 = []
    for i in range(0, len(l1)):
        list1.append((i, l1[i]))
        list2.append((i, l2[i]))
    set1 = set(list1)
    set2 = set(list2)
    intersectionSize = len(set.intersection(set1, set2))
    unionSize = len(set.union(set1, set2))
    # print intersectionSize, unionSize;
    return round(intersectionSize/float(unionSize), 4)


def computeSimilarity(l1, l2):
    """ computes similarity index """
    return jaccardSimilarity(l1, l2)


def findPrediction(SImatrix_gamma, nACh=1, pThreshold=0.75):
    """ computes the correct classifications"""
    pValues = []
    pValuesNaive = []
    maxSI = 0
    maxSIindex = 'x'
    maxSInaive = 0
    maxSInaiveIndex = 'x'
    k = 0
    gammaIndex = 0
    AChCnt = [0] * nACh  # counts number of correct classifications at each ACh level
    for i in range(0, len(SImatrix_gamma)):
        for j in range(0, len(SImatrix_gamma[i])):
            if (SImatrix_gamma[i][j] > maxSI):
                maxSI = SImatrix_gamma[i][j]
                maxSIindex = j
                AChID = k // 10
                if (gammaIndex == 0):
                    maxSInaive = SImatrix_gamma[i][j]
                    maxSInaiveIndex = j
        k += 1
        if (k == 10 * nACh):
            if (maxSI >= pThreshold):
                pValues.append(maxSIindex)
                AChCnt[AChID] += 1
            else:
                pValues.append('x')
                # if(maxSInaive>=pThreshold):
            if (maxSInaive >= pThreshold):
                pValuesNaive.append(maxSInaiveIndex)
            else:
                pValuesNaive.append('x')
            maxSI = 0
            maxSIindex = 'x'
            maxSInaive = 0
            maxSInaiveIndex = 'x'
            k = 0
        gammaIndex += 1
        if (gammaIndex == 10):
            gammaIndex = 0

    return pValues, AChCnt, pValuesNaive


def computeClassification(pValues, nTestPerOdor, nodors):
    """ computes the classification accuracy"""
    currentOdorId = 0
    k = 0
    percentCorrect = 0
    for i in range(0, len(pValues)):
        if (pValues[i] == currentOdorId):
            percentCorrect += 1
        k += 1
        if (k == nTestPerOdor):
            currentOdorId += 1
            k = 0
    return percentCorrect

def computeResults(nGammaPerTraining, trainingSetSize, testSetSize,
                   nsensors=72, verbose=False, gammaCode=None,
                   similarityThreshold=0.75):
    """evaluates the performance of the EPL network"""
    nodors = trainingSetSize
    nNoiseLevels = 1
    nTestPerOdor = testSetSize/nodors
    nACh = 1

    precedenceCodeLearned = []
    # this stores results from 1st gamma cycle to measure performance with naive representation
    precedenceCodeNaive = []

    if gammaCode is None:
        pickedfilename = '/.spikes.pi'
        rf = open(pickedfilename, 'rb')
        precedenceCodeGamma = pickle.load(rf)
        rf.close()
    else:
        precedenceCodeGamma = gammaCode

    # Find learned precedence codes
    for i in range(0, nodors):
        # labelGamma = 2 * i * 2 * 5 + 2 * 5  # labeling period
        labelGamma = nGammaPerTraining*nodors + 5*2*i
        precedenceCodeNaive.append(precedenceCodeGamma[labelGamma])
        labelGamma = labelGamma + 4  # last gamma cycle of label
        # -1 because first gamma missing in simulation
        precedenceCodeLearned.append(precedenceCodeGamma[labelGamma])

    # Compute similarity of test odors to learned odors at every gamma
    # -1 because first gamma in simulation is missing
    testThetaStart = nGammaPerTraining*nodors + 5*2*nodors

    SImatrix_gamma = []
    SImatrix_gammaNaive = []
    gammaIndex = 0
    # for i in range(testThetaStart, len(precedenceCodeGamma)-1):
    for i in range(testThetaStart, len(precedenceCodeGamma)):
        similarityIndices = []
        similarityIndicesNaive = []
        for k in range(0, nodors):
            # SI = cosine_similarity(precedenceCodeGamma[i], precedenceCodeLearned[k])
            if (gammaIndex < 5):
                SI = computeSimilarity(precedenceCodeGamma[i],
                                       precedenceCodeLearned[k])
                similarityIndices.append(SI)
                # SInaive = cosine_similarity(precedenceCodeGamma[i], precedenceCodeNaive[k])
                SInaive = computeSimilarity(precedenceCodeGamma[i],
                                            precedenceCodeNaive[k])
                similarityIndicesNaive.append(SInaive)
            else:
                similarityIndices.append(0)
                similarityIndicesNaive.append(0)
        gammaIndex += 1
        if (gammaIndex == 10):
            gammaIndex = 0
        SImatrix_gamma.append(similarityIndices)
        SImatrix_gammaNaive.append(similarityIndicesNaive)

    # Printing
    for i in precedenceCodeGamma:
        # print(i[0:10])
        pass

    if verbose:
        for i in SImatrix_gamma:
            print(i)
            pass

        # Find predictions and compute classification of EPL results
    pValues, AChCnt, pValuesNaive = findPrediction(SImatrix_gamma, nACh=nACh,
                                            pThreshold=similarityThreshold)

    for i in range(0, len(pValues)):
        # print(pValues[i])
        pass

    percentCorrect = []
    percentCorrectNaive = []
    for i in range(0, nNoiseLevels):
        indexStart = int(nodors * nTestPerOdor * i)
        indexEnd = int(indexStart + nodors * nTestPerOdor)
        percentCorrect.append(
            computeClassification(pValues[indexStart:indexEnd], nTestPerOdor,
                                  nodors=nodors))
        percentCorrectNaive.append(
            computeClassification(pValuesNaive[indexStart:indexEnd],
                                  nTestPerOdor, nodors=nodors))

    for i in range(0, len(percentCorrect)):
        percentCorrect[i] = 100 * round(
            percentCorrect[i] / float(nodors * nTestPerOdor), 2)

    # Printing info
    print("*****Execution Report*****")
    print("{} patterns presented. {} test samples for each pattern".format(
        nodors, int(nTestPerOdor)))
    print("""Classification performance = {}%; for similarity threshold = {} 
            """.format(percentCorrect[0], similarityThreshold))
    return percentCorrect[0]
