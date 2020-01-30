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

import os
import inspect
from nxsdk_modules.epl.src.single_pattern_learning.epl_nxnet import EplNxNet,\
    timer
from jinja2 import Environment, FileSystemLoader


class EplWithSNIPs(EplNxNet):
    """ This class has all the functions/methods to setup, gather and transfer
    the data required by the LMT SNIPs code which manages the execution of the
    EPL network"""

    def __init__(self, eplParams):
        super().__init__(eplParams=eplParams)
        self.totalTimeSteps = None
        self.totalTestSamples = None
        self.lenData = 0

    def _updateLength(self, axonMap):
        """ updates the length of the data being sent """
        if self.lenData == 0:
            # precompute the size of data sent in small chunks
            self.lenData = 1 + 2 + 4 * self.numColumns
        for l in axonMap.values():
            self.lenData += len(l)

    def _updateMap(self, connGrp, coreIdToAxonIdMap):
        """ updates the input axon ids for each core for a given connection
        group """
        for conn in connGrp:
            iaxonId = conn.inputAxon.nodeId
            hwAxonIds = self.net.resourceMap.inputAxon(iaxonId)
            for _, _, coreId, axonId in hwAxonIds:
                if coreId not in coreIdToAxonIdMap.keys():
                    coreIdToAxonIdMap[coreId] = set()
                coreIdToAxonIdMap[coreId].add(axonId)

    def _getCoreIdToAxonIdMapForMCToGCConns(self):
        """ creates a mapping of input axon ids for each core for the MC->GC
        connections """
        coreIdToAxonIdMap = dict()
        for connGrp in self.mcToGCConnGrpsPerDelay:
            self._updateMap(connGrp, coreIdToAxonIdMap)
        self.mcToGCInputAxonMap = coreIdToAxonIdMap
        self._updateLength(self.mcToGCInputAxonMap)

    def _getCoreIdToAxonIdMapForGCToMCConns(self):
        """ creates a mapping of input axon ids for each core for the GC->MC
        connections """
        self.gcToMCExcConnInputAxonMap = dict()
        self.gcToMCInhConnInputAxonMap = dict()
        for connGrp in self.gcToMCConnGrpsPerCore:
            self._updateMap(connGrp.positive, self.gcToMCExcConnInputAxonMap)
            # one map is enough as the input axon ids are shared between
            # excitatory and inhibitory synapses
            #self._updateMap(connGrp.negative, self.gcToMCInhConnInputAxonMap)
        self._updateLength(self.gcToMCExcConnInputAxonMap)

    def _genCodeForConstants(self):
        """ write the constants (known before SNIP compilation) to a
        C header file using Jinja2"""
        cfile = self.snipsDir + "/constants.h"
        templatesDir = self.currDir + "/templates"
        env = Environment(loader=FileSystemLoader(templatesDir))
        template = env.get_template('constants_template.txt')
        consts = dict()
        consts['NUM_CORES'] = self.numColumns
        consts['NUM_MCS_PER_CORE'] = self.numMCsPerColumn
        consts['NUM_GCS_PER_CORE'] = self.numGCsPerColumn
        consts['NUM_MC_TO_GC_DELAYS'] = self.numDelaysMCToGC
        consts['MCAD_CXGRP_ID'] = self.allMCADGroup.groupId
        consts['MCSOMA_CXGRP_ID'] = self.allMCSomaGrp.groupId
        consts['GAMMA_CYCLE_DURATION'] = self.gammaCycleDuration
        consts['NUM_GAMMA_CYCLES_TRAIN'] = self.numGammaCyclesTrain
        consts['NUM_GAMMA_CYCLES_TEST'] = self.numGammaCyclesTest
        consts['NUM_GAMMA_CYCLES_IDLE'] = self.numGammaCyclesIdle
        consts['NO_LEARNING_PERIOD'] = 20
        consts['NUM_TEST_SAMPLES'] = self.totalTestSamples
        consts['USE_LMT_SPIKE_COUNTERS'] = 1 if self.useLMTSpikeCounters else 0
        consts['RUN_TIME'] = self.totalTimeSteps
        consts['LOG_SNIP'] = 1 if self.logSNIPs else 0
        output = template.render(data=consts)
        with open(cfile, "w") as fh:
            fh.write(output)

    def _initSnip(self):
        """ setups the init SNIP"""
        includeDir = self.snipsDir
        cFilePath = includeDir + "/initsnip.c"
        self.initProcess = self.board.createProcess("self.initProcess",
                                                    includeDir=includeDir,
                                                    cFilePath=cFilePath,
                                                    funcName="initParamsAndInputs",
                                                    guardName=None,
                                                    phase="init")
        # lenData = len(self.initData) + self.numMCs + 1
        lenData = self.numMCs + 1
        self.initChannel = self.board.createChannel(b'nxinit', "int", lenData)
        self.initChannel.connect(None, self.initProcess)

    def _mgmtSnip(self):
        """ setups up the management SNIP """
        includeDir = self.snipsDir
        cFilePath = includeDir + "/mgmtsnip.c"
        self.mgmtProcess = self.board.createProcess("self.mgmtProcess",
                                                    includeDir=includeDir,
                                                    cFilePath=cFilePath,
                                                    funcName="runMgmt",
                                                    guardName="doMgmt",
                                                    phase="mgmt")
        lendata = self.lenData
        self.mgmtChannel = self.board.createChannel(b'nxmgmt', "int", lendata)
        self.mgmtChannel.connect(None, self.mgmtProcess)
        self.mgmtChannel2 = self.board.createChannel(b'nxmgmt_mc_inputs', "int",
                                                     self.totalTestSamples * self.numMCs)
        self.mgmtChannel2.connect(None, self.mgmtProcess)

        self.mgmtChannel3 = self.board.createChannel(
            b'nxspkcntr', "int", 2**18)
        self.mgmtChannel3.connect(self.mgmtProcess, None)

        self.mgmtChannel4 = self.board.createChannel(b'status', "int", 10)
        self.mgmtChannel4.connect(self.mgmtProcess, None)

    def _idxToBases(self, inputList):
        """ maps the input data/sensor reading to an MC-AD bias current"""
        return [self.stim2bias[i] for i in inputList]

    def _sendInputAxonMapData(self, axonMap):
        """ send the input axon map data to mgmt SNIP via a read channel"""
        self.mgmtChannel.write(1, [self.numColumns])
        for coreId in range(self.numColumns):
            data = list(axonMap[coreId])
            self.mgmtChannel.write(2, [coreId, len(data)])
            self.mgmtChannel.write(len(data), data)

    def _sendDataToSwitchToInference(self):
        """ send the data required to switch from training mode to inference
        mode by the mgmt SNIP via a read channel"""
        mode = 1  # testing
        self.mgmtChannel.write(1, [mode])
        self._sendInputAxonMapData(axonMap=self.mcToGCInputAxonMap)
        self._sendInputAxonMapData(axonMap=self.gcToMCExcConnInputAxonMap)

    def sendDataToSNIP(self):
        """ send the data needed for the init and mgmt SNIPs"""
        l = len(self.trainData)
        self.initChannel.write(l, self.trainData)
        self._sendDataToSwitchToInference()
        self.mgmtChannel2.write(l, self.trainData)
        self.mgmtChannel2.write(len(self.testData), self.testData)

    def gatherDataForSNIPs(self, trainingSet, testingSet):
        """ collect all the data that needs to transfered to the SNIPs via
        channels"""
        self.trainData = []
        for trainSample in trainingSet:
            self.trainData += self._idxToBases(trainSample)
        self.testData = []
        for testSample in testingSet:
            testSample = self._idxToBases(testSample)
            self.testData += testSample
        self._getCoreIdToAxonIdMapForMCToGCConns()
        self._getCoreIdToAxonIdMapForGCToMCConns()

    def setupSNIPs(self, totalTestSamples, totalTimeSteps):
        self.totalTestSamples = totalTestSamples
        self.totalTimeSteps = totalTimeSteps
        self.currDir = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.snipsDir = self.currDir + "/snips"
        self._genCodeForConstants()
        self._initSnip()
        self._mgmtSnip()
