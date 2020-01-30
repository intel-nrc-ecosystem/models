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

import csv
import numpy as np
import os


class MCGCWeightsGen:

    def __init__(self, inhToExcWgtsFile="inhToExcWgtsFile.txt",
                 excToInhWgtsFile="excToInhWgtsFile.txt",
                 numCores=72, numGCPerCore=46, numMCPerCore=1, numDelays=4,
                 minDelay=16):

        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.i2eWgtsFile = os.path.join(dir_path, inhToExcWgtsFile)
        self.e2iWgtsFile = os.path.join(dir_path, excToInhWgtsFile)

        self.numCores = numCores
        self.numGCPerCore = numGCPerCore
        self.numMCPerCore = numMCPerCore
        self.numDelays = numDelays
        self.minDelay = minDelay
        self.saveWgtsE2I()
        self.saveWgtsAndDelaysI2E()

    @property
    def numGC(self):
        return self.numCores * self.numGCPerCore

    @property
    def numMC(self):
        return self.numCores * self.numMCPerCore

    def saveWgtsE2I(self):

        e2iWgtMat = np.zeros((self.numDelays, self.numGC, self.numMC))
        print(e2iWgtMat.shape)
        with open(self.e2iWgtsFile) as e2iFile:
            csvReader = csv.reader(e2iFile, delimiter=',')
            for row in csvReader:
                int_row = [int(item) for item in row]
                coreIdx, gcIdx, mcIdx, wgt, dly = tuple(int_row)
                gcIdx = coreIdx * self.numGCPerCore + gcIdx
                mcIdx = self.numMCPerCore * mcIdx
                dlyIdx = dly - self.minDelay
                if e2iWgtMat[dlyIdx, gcIdx, mcIdx] != 0:
                    raise ValueError("Duplicate weights")

                if wgt != 0:
                    e2iWgtMat[dlyIdx, gcIdx, mcIdx] = wgt

        #print(np.where(e2iWgtMat > 0))
        np.save("e2iWgtMat", e2iWgtMat)

    def saveWgtsAndDelaysI2E(self):

        i2eWgtMat = np.zeros((2, self.numCores, self.numMCPerCore,
                              self.numGCPerCore))
        i2eDlyMat = np.zeros(i2eWgtMat.shape)
        print(i2eWgtMat.shape)
        with open(self.i2eWgtsFile) as i2eFile:
            csvReader = csv.reader(i2eFile, delimiter=',')
            for row in csvReader:
                int_row = [int(item) for item in row]
                coreIdx, gcIdx, mcIdx, wgt, dly = tuple(int_row)
                boxIdx = 0 if wgt > 0 else 1

                if i2eWgtMat[boxIdx, coreIdx, mcIdx, gcIdx] != 0:
                    raise ValueError("Duplicate weights for core, gc , mc",
                                     coreIdx, gcIdx, mcIdx, i2eWgtMat[coreIdx, mcIdx, gcIdx])

                if wgt != 0:
                    i2eWgtMat[boxIdx, coreIdx, mcIdx, gcIdx] = wgt
                    i2eDlyMat[boxIdx, coreIdx, mcIdx, gcIdx] = dly

        #print(np.where(i2eWgtMat > 0))
        np.save("i2eWgtMat", i2eWgtMat)
        np.save("i2eDlyMat", i2eDlyMat)


if __name__ == '__main__':
    wgen = MCGCWeightsGen()
