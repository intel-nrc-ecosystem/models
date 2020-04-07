# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

"""Test DNN partitioning code."""
import os
import unittest

import numpy as np
from scipy.sparse import lil_matrix
from test import support

from nxsdk.graph.nxprobes import N2Probe, N2SpikeProbe
from official.dnn.src.optimization import getDummyLayer
from official.dnn.src.utils import getCoreIdMapFromCoreShape
from official.dnn.src.plotting import plotMat
from official.dnn.src.data_structures import Layer
from official.dnn.src.dnn_layers import NxInputLayer, NxConv2D, \
    NxModel, ProbableStates, NxAveragePooling2D, NxDepthwiseConv2D, NxDense, \
    NxFlatten, NxConv1D, loadNxModel
from official.dnn.src.synapse_compression import SynapseEncoder


class TestSynapseEncoding(unittest.TestCase):
    """Test various synapse encoding methods."""

    def setUp(self):
        """Define common parameters."""

        self.verbose = False
        self.numWeightBits = 8
        self.maxNumSynPerSynEntry = 60

    def test_encodeSynSparse1(self):
        """Check SPARSE encoding of synapses."""

        syn = lil_matrix([1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4])
        synIds = np.array(syn.rows[0])
        synVals = np.array(syn.data[0])

        synapseEncoder = SynapseEncoder(self.numWeightBits,
                                        self.maxNumSynPerSynEntry, 'sparse',
                                        True)
        synapseEncoder.encode(synIds, synVals, 0, 0)

        synEntries = synapseEncoder.getSynEntries()
        synFmts = synapseEncoder.getSynFmts()

        numBits = []
        numPrefixBits = []
        numSynBits = []
        for synEntry in synEntries:
            numBits.append(synEntry.numSynBits + synEntry.numPrefixBits)
            numPrefixBits.append(synEntry.numPrefixBits)
            numSynBits.append(synEntry.numSynBits)

        self.assertEqual(len(synEntries), 1)
        self.assertEqual(len(synFmts), 1)

        if self.verbose:
            synEntries[0].print()

        self.assertEqual(synEntries[0].prefixOffset, 0)
        self.assertEqual(tuple(synEntries[0].idxs), (0, 5, 6, 11))
        self.assertEqual(tuple(synEntries[0].weights), (1, 2, 3, 4))
        self.assertEqual(synEntries[0].synFmtId, 0)
        self.assertEqual(synEntries[0].numSyn, 4)
        self.assertEqual(numPrefixBits[0], 10)
        self.assertEqual(numSynBits[0], 56)
        self.assertEqual(numBits[0], 66)

        self.assertEqual(synFmts[0].numIdxBits, 6)
        self.assertEqual(synFmts[0].numSkipBits, 0)
        self.assertEqual(synFmts[0].numWgtBits, 8)

    def test_encodeSynSparse2(self):
        """Check SPARSE encoding of synapses into multiple synEntries."""

        syn = lil_matrix([1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4])
        synIds = np.array(syn.rows[0])
        synVals = np.array(syn.data[0])

        synapseEncoder = SynapseEncoder(self.numWeightBits, 2, 'sparse', True)
        synapseEncoder.encode(synIds, synVals, 0, 0)

        synEntries = synapseEncoder.getSynEntries()
        synFmts = synapseEncoder.getSynFmts()

        numBits = []
        numPrefixBits = []
        numSynBits = []
        for synEntry in synEntries:
            numBits.append(synEntry.numSynBits + synEntry.numPrefixBits)
            numPrefixBits.append(synEntry.numPrefixBits)
            numSynBits.append(synEntry.numSynBits)

        self.assertEqual(len(synEntries), 2)
        self.assertEqual(len(synFmts), 2)

        if self.verbose:
            synEntries[0].print()

        self.assertEqual(synEntries[0].prefixOffset, 0)
        self.assertEqual(tuple(synEntries[0].idxs), (0, 5))
        self.assertEqual(tuple(synEntries[0].weights), (1, 2))
        self.assertEqual(synEntries[0].synFmtId, 0)
        self.assertEqual(synEntries[0].numSyn, 2)
        self.assertEqual(numPrefixBits[0], 10)
        self.assertEqual(numSynBits[0], 28)
        self.assertEqual(numBits[0], 38)

        self.assertEqual(synFmts[0].numIdxBits, 6)
        self.assertEqual(synFmts[0].numSkipBits, 0)
        self.assertEqual(synFmts[0].numWgtBits, 8)

        if self.verbose:
            synEntries[1].print()

        self.assertEqual(synEntries[1].prefixOffset, 0)
        self.assertEqual(tuple(synEntries[1].idxs), (6, 11))
        self.assertEqual(tuple(synEntries[1].weights), (3, 4))
        self.assertEqual(synEntries[1].synFmtId, 1)
        self.assertEqual(synEntries[1].numSyn, 2)
        self.assertEqual(numPrefixBits[0], 10)
        self.assertEqual(numSynBits[0], 28)
        self.assertEqual(numBits[0], 38)

        self.assertEqual(synFmts[1].numIdxBits, 6)
        self.assertEqual(synFmts[1].numSkipBits, 0)
        self.assertEqual(synFmts[1].numWgtBits, 8)

    def test_encodeSynRunLength1(self):
        """Check RUNLENGTH encoding of synapses."""

        syn = lil_matrix([0] * 100 + [1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4])
        synIds = np.array(syn.rows[0])
        synVals = np.array(syn.data[0])

        synapseEncoder = SynapseEncoder(self.numWeightBits,
                                        self.maxNumSynPerSynEntry, 'runlength',
                                        True)
        synapseEncoder.encode(synIds, synVals, 1, 2)

        synEntries = synapseEncoder.getSynEntries()
        synFmts = synapseEncoder.getSynFmts()

        numBits = []
        numPrefixBits = []
        numSynBits = []
        for synEntry in synEntries:
            numBits.append(synEntry.numSynBits + synEntry.numPrefixBits)
            numPrefixBits.append(synEntry.numPrefixBits)
            numSynBits.append(synEntry.numSynBits)

        self.assertEqual(len(synEntries), 1)
        self.assertEqual(len(synFmts), 1)

        if self.verbose:
            synEntries[0].print()

        self.assertEqual(synEntries[0].prefixOffset, 100)
        self.assertEqual(tuple(synEntries[0].idxs), (0, 5, 1, 5))
        self.assertEqual(tuple(synEntries[0].weights), (1, 2, 3, 4))
        self.assertEqual(synEntries[0].synFmtId, 0)
        self.assertEqual(synEntries[0].numSyn, 4)
        self.assertEqual(numPrefixBits[0], 10 + 7)
        self.assertEqual(numSynBits[0], 44)
        self.assertEqual(numBits[0], 61)

        self.assertEqual(synFmts[0].numIdxBits, 7)
        self.assertEqual(synFmts[0].numSkipBits, 3)
        self.assertEqual(synFmts[0].numWgtBits, 8)
        self.assertEqual(synFmts[0].cIdxOffset, 1)
        self.assertEqual(synFmts[0].cIdxMult, 2)

    def test_encodeSynRunLength2(self):
        """Check RUNLENGTH encoding of synapses into multiple synEntries."""

        syn = lil_matrix([0] * 100 + [1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4])
        synIds = np.array(syn.rows[0])
        synVals = np.array(syn.data[0])
        
        synapseEncoder = SynapseEncoder(self.numWeightBits, 2, 'runlength',
                                        True)
        synapseEncoder.encode(synIds, synVals, 0, 0)

        synEntries = synapseEncoder.getSynEntries()
        synFmts = synapseEncoder.getSynFmts()

        numBits = []
        numPrefixBits = []
        numSynBits = []
        for synEntry in synEntries:
            numBits.append(synEntry.numSynBits + synEntry.numPrefixBits)
            numPrefixBits.append(synEntry.numPrefixBits)
            numSynBits.append(synEntry.numSynBits)
            
        self.assertEqual(len(synEntries), 2)
        self.assertEqual(len(synFmts), 2)

        if self.verbose:
            synEntries[0].print()

        self.assertEqual(synEntries[0].prefixOffset, 100)
        self.assertEqual(tuple(synEntries[0].idxs), (0, 5))
        self.assertEqual(tuple(synEntries[0].weights), (1, 2))
        self.assertEqual(synEntries[0].synFmtId, 0)
        self.assertEqual(synEntries[0].numSyn, 2)
        self.assertEqual(numPrefixBits[0], 10 + 7)
        self.assertEqual(numSynBits[0], 22)
        self.assertEqual(numBits[0], 22 + 17)

        self.assertEqual(synFmts[0].numIdxBits, 7)
        self.assertEqual(synFmts[0].numSkipBits, 3)
        self.assertEqual(synFmts[0].numWgtBits, 8)

        if self.verbose:
            synEntries[1].print()

        self.assertEqual(synEntries[1].prefixOffset, 106)
        self.assertEqual(tuple(synEntries[1].idxs), (0, 5))
        self.assertEqual(tuple(synEntries[1].weights), (3, 4))
        self.assertEqual(synEntries[1].synFmtId, 1)
        self.assertEqual(synEntries[1].numSyn, 2)
        self.assertEqual(numPrefixBits[1], 10 + 7)
        self.assertEqual(numSynBits[1], 22)
        self.assertEqual(numBits[1], 22 + 17)

        self.assertEqual(synFmts[1].numIdxBits, 7)
        self.assertEqual(synFmts[1].numSkipBits, 3)
        self.assertEqual(synFmts[1].numWgtBits, 8)

    def test_encodeSynDense1(self):
        """Check DENSE encoding of synapses into multiple synEntries."""

        syn = lil_matrix([0] * 200 + [1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4])

        synIds = np.array(syn.rows[0])
        synVals = np.array(syn.data[0])

        synapseEncoder = SynapseEncoder(self.numWeightBits, 2, 'dense1', True)
        synapseEncoder.encode(synIds, synVals, 0, 0)

        synEntries = synapseEncoder.getSynEntries()
        synFmts = synapseEncoder.getSynFmts()

        numBits = []
        numPrefixBits = []
        numSynBits = []
        for synEntry in synEntries:
            numBits.append(synEntry.numSynBits + synEntry.numPrefixBits)
            numPrefixBits.append(synEntry.numPrefixBits)
            numSynBits.append(synEntry.numSynBits)

        self.assertEqual(len(synEntries), 3)
        self.assertEqual(len(synFmts), 3)

        if self.verbose:
            synEntries[0].print()

        self.assertEqual(synEntries[0].prefixOffset, 200)
        self.assertEqual(tuple(synEntries[0].idxs), (0,))
        self.assertEqual(synEntries[0].weights, 1)
        self.assertEqual(synEntries[0].synFmtId, 0)
        self.assertEqual(synEntries[0].numSyn, 1)
        self.assertEqual(numPrefixBits[0], 10 + 8)
        self.assertEqual(numSynBits[0], 8)
        self.assertEqual(numBits[0], 18 + 8)

        self.assertEqual(synFmts[0].numIdxBits, 8)
        self.assertEqual(synFmts[0].numSkipBits, 0)
        self.assertEqual(synFmts[0].numWgtBits, 8)

        if self.verbose:
            synEntries[1].print()

        self.assertEqual(synEntries[1].prefixOffset, 205)
        self.assertEqual((tuple(synEntries[1].idxs)), (0, 1))
        self.assertEqual(tuple(synEntries[1].weights), (2, 3))
        self.assertEqual(synEntries[1].synFmtId, 1)
        self.assertEqual(synEntries[1].numSyn, 2)
        self.assertEqual(numPrefixBits[1], 10 + 8)
        self.assertEqual(numSynBits[1], 16)
        self.assertEqual(numBits[1], 18 + 16)

        self.assertEqual(synFmts[1].numIdxBits, 8)
        self.assertEqual(synFmts[1].numSkipBits, 0)
        self.assertEqual(synFmts[1].numWgtBits, 8)

        if self.verbose:
            synEntries[2].print()

        self.assertEqual(synEntries[2].prefixOffset, 211)
        self.assertEqual((tuple(synEntries[2].idxs)), (0,))
        self.assertEqual(tuple(synEntries[2].weights), (4,))
        self.assertEqual(synEntries[2].synFmtId, 2)
        self.assertEqual(synEntries[2].numSyn, 1)
        self.assertEqual(numPrefixBits[2], 10 + 8)
        self.assertEqual(numSynBits[2], 8)
        self.assertEqual(numBits[2], 18 + 8)

        self.assertEqual(synFmts[2].numIdxBits, 8)
        self.assertEqual(synFmts[2].numSkipBits, 0)
        self.assertEqual(synFmts[2].numWgtBits, 8)

    def test_encodeSynDense2(self):
        """Check DENSE encoding of synapses with dummy connections."""

        syn = lil_matrix([0] * 200 +
                         [1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4, 0, 0])

        synIds = np.array(syn.rows[0])
        synVals = np.array(syn.data[0])

        synapseEncoder = SynapseEncoder(self.numWeightBits,
                                        self.maxNumSynPerSynEntry, 'dense2',
                                        True)
        synapseEncoder.encode(synIds, synVals, 0, 0)

        synEntries = synapseEncoder.getSynEntries()
        synFmts = synapseEncoder.getSynFmts()

        numBits = []
        numPrefixBits = []
        numSynBits = []
        for synEntry in synEntries:
            numBits.append(synEntry.numSynBits + synEntry.numPrefixBits)
            numPrefixBits.append(synEntry.numPrefixBits)
            numSynBits.append(synEntry.numSynBits)

        self.assertEqual(len(synEntries), 1)
        self.assertEqual(len(synFmts), 1)

        if self.verbose:
            synEntries[0].print()

        self.assertEqual(synEntries[0].prefixOffset, 200)
        self.assertEqual(tuple(synEntries[0].idxs),
                         (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
        self.assertEqual(tuple(synEntries[0].weights),
                         (1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4))
        self.assertEqual(synEntries[0].synFmtId, 0)
        self.assertEqual(synEntries[0].numSyn, 12)
        self.assertEqual(numPrefixBits[0], 10 + 8)
        self.assertEqual(numSynBits[0], 12 * 8)
        self.assertEqual(numBits[0], 18 + 12 * 8)

        self.assertEqual(synFmts[0].numIdxBits, 8)
        self.assertEqual(synFmts[0].numSkipBits, 0)
        self.assertEqual(synFmts[0].numWgtBits, 8)

    def test_compressConnections(self):
        """Check axon compression."""

        inputShapes = np.array([
            [7, 7, 2],
            [32, 32, 3],
            [32, 32, 3],
            [27, 26, 2],
            [32, 30, 2],
            [30, 35, 2],
            [7, 7, 2],
            [20, 25, 2],
            [7, 7, 2],
            [30, 35, 2],
            [7, 7, 2],
            [30, 35, 2]
        ])

        layerArgs = np.array([
            [2, 3],
            [2, 3],
            [2, 3],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 3],
            [2, 3],
            [2, 3],
            [2, 3],
            [2, 3],
            [2, 3]
        ])

        layerKwargs = np.array([
            {'synapseEncoding': 'sparse'},
            {'synapseEncoding': 'sparse'},
            {'synapseEncoding': 'sparse', 'strides': 2},
            {'synapseEncoding': 'sparse', 'strides': 2},
            {'synapseEncoding': 'sparse', 'strides': 2},
            {'synapseEncoding': 'sparse', 'strides': 2, 'padding': 'same'},
            {'synapseEncoding': 'runlength'},
            {'synapseEncoding': 'runlength', 'strides': 2, 'padding': 'same'},
            {'synapseEncoding': 'dense1'},
            {'synapseEncoding': 'dense1', 'strides': 2, 'padding': 'same'},
            {'synapseEncoding': 'dense2'},
            {'synapseEncoding': 'dense2', 'strides': 2, 'padding': 'same'},
        ])

        coreShapes = np.array([
            [5, 5, 2],
            [30, 10, 2],
            [10, 5, 2],
            [7, 13, 2],
            [16, 5, 2],
            [8, 9, 2],
            [5, 5, 2],
            [6, 9, 2],
            [5, 5, 2],
            [15, 10, 2],
            [5, 5, 2],
            [6, 9, 2]
        ])

        # numOutputAxons (second column) is expected to be zero because we are
        # looking at the output layer.
        expectedResults = np.array([
            [14, 0, 900, 281, 378],
            [108, 0, 29160, 6873, 10080],
            [198, 0, 4050, 2430, 2592],
            [52, 0, 56, 42, 28],
            [60, 0, 768, 528, 384],
            [144, 0, 1408, 808, 868],
            [14, 0, 900, 288, 371],
            [104, 0, 928, 524, 580],
            [14, 0, 900, 660, 427],
            [72, 0, 1408, 1226, 840],
            [14, 0, 3587, 290, 700],
            [216, 0, 2920, 874, 1052]
        ])

        doTest = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool)

        np.random.seed(0)
        for inputShape, args, kwargs, coreShape, expectedResult in \
                zip(inputShapes[doTest],
                    layerArgs[doTest],
                    layerKwargs[doTest],
                    coreShapes[doTest],
                    expectedResults[doTest]):

            with self.subTest(inputShape=inputShape,
                              args=args,
                              kwargs=kwargs,
                              coreShape=coreShape,
                              expectedResult=expectedResult):

                # The Keras layer initializer does not know how to handle
                # np.int types. Convert to builtin int manually.
                args = list(args)
                args[1] = int(args[1])

                inputLayer = NxInputLayer(tuple(inputShape))
                layer = NxConv2D(*args, **kwargs)
                layer(inputLayer.input)

                # Assumes that this layer will never go beyond one chip.
                layer.coreCounter = 0

                layerShape = layer.output_shape[1:]
                numCoresPerAxis = np.ceil(layerShape / coreShape).astype(int)
                coreIdMap = getCoreIdMapFromCoreShape(coreShape, layerShape,
                                                      numCoresPerAxis)

                multiplicityMapPre = np.ones(layer.input_shape[1:-1], int)
                postLayerPartition = getDummyLayer(layer.output_shape[1:])

                partitionCandidate = Layer(layer.name, '',
                                           layer.compartmentKwargs,
                                           layer.connectionKwargs, coreIdMap,
                                           multiplicityMapPre,
                                           postLayerPartition)

                partitionCandidate = layer.compile(partitionCandidate)

                try:
                    layer.validatePartition(partitionCandidate)
                finally:
                    layer.deleteKernelIdMap()

                layer.deleteKernelIdMap()

                result = [partitionCandidate.numInputAxons,
                          partitionCandidate.numOutputAxons,
                          partitionCandidate.numSyn,
                          partitionCandidate.numSynEntries,
                          partitionCandidate.numSynMemWords]
                self.verbose = True
                if self.verbose:
                    print(result)

                for er, r in zip(expectedResult, result):
                    self.assertEqual(er, r)


class TestNxConv2D(unittest.TestCase):
    """Test methods for partitioning a Nx convolution layer."""

    def setUp(self):
        """Define common parameters."""

        self.verbose = False

    def test_getCoreIdMapFromCoreShape(self):
        """Test generating the coreIdMap given a coreShape."""

        inputShape = (9, 9, 2)
        inputLayer = NxInputLayer(inputShape)
        layer = NxConv2D(2, 3, strides=2)
        layer(inputLayer.input)

        coreShape = np.array([1, 4, 2])
        layerShape = layer.output_shape[1:]
        numCoresPerAxis = np.ceil(layerShape / coreShape).astype(int)
        coreIds = getCoreIdMapFromCoreShape(coreShape, layerShape,
                                            numCoresPerAxis)
        target = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]],
                           [[1, 1], [1, 1], [1, 1], [1, 1]],
                           [[2, 2], [2, 2], [2, 2], [2, 2]],
                           [[3, 3], [3, 3], [3, 3], [3, 3]]])
        self.assertTrue(np.array_equal(coreIds, target))

        if self.verbose:
            plotMat(coreIds[:, :, 0], title='coreIds[:, :, 0]')
            plotMat(coreIds[:, :, 1], title='coreIds[:, :, 1]')

    def test_getCoreIdMapFromCoreShape2(self):
        """Test generating the coreIdMap given a coreShape."""

        inputShape = (4, 4, 2)
        inputLayer = NxInputLayer(inputShape)
        layer = NxConv2D(2, 3)
        layer(inputLayer.input)

        coreShape = np.array([2, 1, 2])
        layerShape = layer.output_shape[1:]
        numCoresPerAxis = np.ceil(layerShape / coreShape).astype(int)
        coreIds = getCoreIdMapFromCoreShape(coreShape, layerShape,
                                            numCoresPerAxis)
        target = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]])
        self.assertTrue(np.array_equal(coreIds, target))

        if self.verbose:
            plotMat(coreIds[:, :, 0], title='coreIds[:, :, 0]')
            plotMat(coreIds[:, :, 1], title='coreIds[:, :, 1]')

    def test_getMultiplicityMap(self):
        """Test generating a multiplicity map from a coreIdMap."""

        inputShape = (10, 10, 2)
        inputLayer = NxInputLayer(inputShape)
        layer = NxConv2D(2, 3)
        layer(inputLayer.input)

        coreShape = np.array([4, 3, 2])
        layerShape = layer.output_shape[1:]
        numCoresPerAxis = np.ceil(layerShape / coreShape).astype(int)
        coreIdMap = getCoreIdMapFromCoreShape(coreShape, layerShape,
                                              numCoresPerAxis)
        multiplicityMap = layer.getMultiplicityMap(coreIdMap)

        if self.verbose:
            plotMat(multiplicityMap, title='MultiplicityMap')

        target = [[1, 1, 1, 2, 2, 1, 2, 2, 1, 1],
                  [1, 1, 1, 2, 2, 1, 2, 2, 1, 1],
                  [1, 1, 1, 2, 2, 1, 2, 2, 1, 1],
                  [1, 1, 1, 2, 2, 1, 2, 2, 1, 1],
                  [2, 2, 2, 4, 4, 2, 4, 4, 2, 2],
                  [2, 2, 2, 4, 4, 2, 4, 4, 2, 2],
                  [1, 1, 1, 2, 2, 1, 2, 2, 1, 1],
                  [1, 1, 1, 2, 2, 1, 2, 2, 1, 1],
                  [1, 1, 1, 2, 2, 1, 2, 2, 1, 1],
                  [1, 1, 1, 2, 2, 1, 2, 2, 1, 1]]

        self.assertTrue(np.array_equal(multiplicityMap, target))


class TestNxAveragePooling2D(unittest.TestCase):
    """Test partitioning NxPooling layers."""

    def setUp(self):
        """Define commom parameters."""

        self.verbose = False

    def test_partitionPooling(self):
        """Test partitioning a single pooling layer."""

        inputShape = (5, 5, 4)
        inputLayer = NxInputLayer(inputShape)
        outputLayer = NxAveragePooling2D(2, validatePartitions=True)
        model = NxModel(inputLayer.input, outputLayer(inputLayer.input))

        model.partition()

        model.clearTemp()

    def test_partitionPooling2(self):
        """Test partitioning two pooling layers."""

        inputShape = (30, 40, 4)
        inputLayer = NxInputLayer(inputShape)
        hiddenLayer = NxAveragePooling2D(2, padding='same',
                                         validatePartitions=True)
        outputLayer = NxAveragePooling2D(3, strides=(1, 1),
                                         validatePartitions=True)
        model = NxModel(inputLayer.input,
                        outputLayer(hiddenLayer(inputLayer.input)))

        model.partition()

        model.clearTemp()


class TestNxDepthwiseConv2D(unittest.TestCase):
    """Test partitioning NxDepthwiseConv2D layers."""

    def setUp(self):
        """Define commom parameters."""

        self.verbose = False

    def test_partition1(self):
        """Test partitioning a single DepthwiseConv2D layer."""

        inputShape = (5, 5, 4)
        inputLayer = NxInputLayer(inputShape)
        outputLayer = NxDepthwiseConv2D(2, validatePartitions=True)
        model = NxModel(inputLayer.input, outputLayer(inputLayer.input))

        model.partition()

        model.clearTemp()

    def test_partition2(self):
        """Test partitioning two DepthwiseConv2D layers."""

        inputShape = (50, 20, 4)
        inputLayer = NxInputLayer(inputShape)
        hiddenLayer = NxDepthwiseConv2D(3, strides=(2, 2), padding='same',
                                        validatePartitions=True)
        outputLayer = NxDepthwiseConv2D(2, padding='same',
                                        validatePartitions=True)
        model = NxModel(inputLayer.input,
                        outputLayer(hiddenLayer(inputLayer.input)))

        model.partition()

        model.clearTemp()


class TestNxConv1D(unittest.TestCase):
    """Test partitioning NxConv1D layers."""

    def setUp(self):
        """Define commom parameters."""

        self.verbose = False

    def test_partition1(self):
        """Test partitioning a single NxConv1D layer."""

        inputShape = (5, 4)
        inputLayer = NxInputLayer(inputShape)
        outputLayer = NxConv1D(2, 3, validatePartitions=True)
        model = NxModel(inputLayer.input, outputLayer(inputLayer.input))

        model.partition()

        model.clearTemp()

    def test_partition2(self):
        """Test partitioning two NxConv1D layers."""

        inputShape = (500, 4)
        inputLayer = NxInputLayer(inputShape)
        hiddenLayer = NxConv1D(20, 3, strides=2, padding='same',
                               validatePartitions=True)
        outputLayer = NxConv1D(10, 2, padding='same', validatePartitions=True)
        model = NxModel(inputLayer.input,
                        outputLayer(hiddenLayer(inputLayer.input)))

        model.partition()

        model.clearTemp()


class TestNxDense(unittest.TestCase):
    """Test partitioning NxDense layers."""

    def setUp(self):
        """Define commom parameters."""

        self.verbose = False

    def test_partition1(self):
        """Test partitioning a single fully-connected layer."""

        inputShape = (3, 3, 2)
        inputLayer = NxInputLayer(inputShape)
        flattenLayer = NxFlatten()(inputLayer.input)
        outputLayer = NxDense(10, validatePartitions=True)
        model = NxModel(inputLayer.input, outputLayer(flattenLayer))

        model.partition()

        model.clearTemp()

    def test_partition2(self):
        """Test partitioning two fully-connected layers."""

        inputShape = (40, 32, 2)
        inputLayer = NxInputLayer(inputShape)
        flattenLayer = NxFlatten()(inputLayer.input)
        hiddenLayer = NxDense(100, validatePartitions=True)
        outputLayer = NxDense(10, validatePartitions=True)
        model = NxModel(inputLayer.input,
                        outputLayer(hiddenLayer(flattenLayer)))

        model.partition()

        model.clearTemp()


class TestNxModel(unittest.TestCase):
    """Test partitioning NxModels."""

    def setUp(self):
        """Define commom parameters."""

        self.verbose = False

    def test_compileNxModel(self):
        """Check that NxModel can be compiled with Keras."""

        inputShape = (7, 7, 1)
        inputLayer = NxInputLayer(inputShape)
        outputLayer = NxConv2D(2, 3)(inputLayer.input)
        model = NxModel(inputLayer.input, outputLayer)
        model.clearTemp()

    def test_evaluateNxModel(self):
        """Check that NxModel can be evaluated like a Keras Model."""

        batchSize = 10
        batchInputShape = (batchSize, 7, 7, 1)
        inputLayer = NxInputLayer(batch_input_shape=batchInputShape)
        layer = NxConv2D(2, 3)(inputLayer.input)
        model = NxModel(inputLayer.input, layer)

        model.compile('sgd', 'mse')
        model.evaluate(np.random.random_sample(model.input_shape),
                       np.random.random_sample(model.output_shape))
        model.clearTemp()

    def test_saveLoadNxModel(self):
        """Check that NxModel can be saved and loaded like a Keras Model."""

        inputLayer = NxInputLayer(batch_input_shape=(1, 10, 10, 3))
        layer = NxConv2D(2, 3)(inputLayer.input)
        model1 = NxModel(inputLayer.input, layer)
        model1.compile('sgd', 'mse')
        model1.compileModel()
        model1.clearTemp()
        filename = os.path.abspath(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '../../..', 'temp',
            str(hash(model1))))
        model1.save(filename)
        model2 = loadNxModel(filename)
        os.remove(filename)

        x = np.random.random_sample(model1.input_shape)
        y1 = model1.predict(x)
        y2 = model2.predict(x)

        self.assertTrue(np.array_equal(y1, y2))

    def test_partitionModel(self):
        """Test partitioning of a NxModel.

        After completion of the algorithm, the partitioner reconstructs the
        kernelIdMap from the synapses and axons generated during partitioning.
        An exception is thrown if the reconstructed map does not equal the
        original map.
        """

        inputShape = (33, 51, 2)
        inputLayer = NxInputLayer(inputShape)
        hiddenLayer = NxConv2D(7, 4, padding='same',
                               validatePartitions=True)(inputLayer.input)
        hiddenLayer = NxConv2D(9, 3, strides=2,
                               validatePartitions=True)(hiddenLayer)
        hiddenLayer = NxFlatten()(hiddenLayer)
        outputLayer = NxDense(4, validatePartitions=True)(hiddenLayer)

        model = NxModel(inputLayer.input, outputLayer)

        model.partition()

        model.clearTemp()

    def test_partitionModel2(self):
        """Test partitioning of a NxModel.

        After completion of the algorithm, the partitioner reconstructs the
        kernelIdMap from the synapses and axons generated during partitioning.
        An exception is thrown if the reconstructed map does not equal the
        original map.
        """

        inputShape = (2, 2, 128)
        inputLayer = NxInputLayer(inputShape)
        hiddenLayer = NxConv2D(128, 2, padding='same',
                               validatePartitions=True)
        hiddenLayer.exclusionCriteria.maxNumCompartments /= 4
        outputLayer = NxConv2D(256, 3, padding='same',
                               validatePartitions=True)
        outputLayer.exclusionCriteria.maxNumCompartments /= 4

        model = NxModel(inputLayer.input,
                        outputLayer(hiddenLayer(inputLayer.input)))

        model.partition()

        model.clearTemp()

    def test_partitionModel3(self):
        """Test partitioning of a NxModel.

        After completion of the algorithm, the partitioner reconstructs the
        kernelIdMap from the synapses and axons generated during partitioning.
        An exception is thrown if the reconstructed map does not equal the
        original map.
        """

        inputShape = (73, 81, 3)
        inputLayer = NxInputLayer(inputShape)
        hiddenLayer = NxConv2D(11, 3, strides=(2, 2), padding='same',
                               validatePartitions=True)(inputLayer.input)
        hiddenLayer = NxAveragePooling2D(4,
                                         validatePartitions=True)(hiddenLayer)
        hiddenLayer = NxFlatten()(hiddenLayer)
        outputLayer = NxDense(50, validatePartitions=True)(hiddenLayer)

        model = NxModel(inputLayer.input, outputLayer)

        model.partition()

        model.clearTemp()

    def test_partitionModel4(self):
        """Test partitioning of a NxModel.

        After completion of the algorithm, the partitioner reconstructs the
        kernelIdMap from the synapses and axons generated during partitioning.
        An exception is thrown if the reconstructed map does not equal the
        original map.
        """

        inputShape = (15, 15, 3)
        inputLayer = NxInputLayer(inputShape)
        outputLayer = NxConv2D(3, 3, strides=(2, 2), padding='same',
                               validatePartitions=True)(inputLayer.input)

        model = NxModel(inputLayer.input, outputLayer,
                        numCandidatesToCompute=10)

        model.partition()

        model.clearTemp()

    def test_partitionModel5(self):
        """Test partitioning of a NxModel.

        After completion of the algorithm, the partitioner reconstructs the
        kernelIdMap from the synapses and axons generated during partitioning.
        An exception is thrown if the reconstructed map does not equal the
        original map.
        """

        inputShape = (1235, 2)
        inputLayer = NxInputLayer(inputShape, validatePartitions=True)
        hiddenLayer = NxConv1D(11, 3, padding='same',
                               validatePartitions=True)(inputLayer.input)
        hiddenLayer = NxConv1D(5, 4, strides=2,
                               validatePartitions=True)(hiddenLayer)
        hiddenLayer = NxFlatten()(hiddenLayer)
        outputLayer = NxDense(40, validatePartitions=True)(hiddenLayer)

        model = NxModel(inputLayer.input, outputLayer)

        model.partition()

        model.clearTemp()


class TestCompartmentInterface(unittest.TestCase):
    """Test CompartmentInterface."""

    def test_setter_getter(self):
        """Check setter and getter methods of CompartmentInterface."""

        # Create an arbitrary model
        inputShape = (16, 16, 1)
        inputLayer = NxInputLayer(inputShape)
        outputLayer = NxConv2D(1, 3, padding='same',
                               validatePartitions=True)
        model = NxModel(inputLayer.input, outputLayer(inputLayer.input),
                        numCandidatesToCompute=3)

        model.compileModel()

        # Set ome arbitrary values
        outputLayer[0].current = 1
        outputLayer[0].voltage = 2
        outputLayer[0].activity = 3
        outputLayer[0].biasMant = 4
        outputLayer[0].biasExp = 5
        outputLayer[0].phase = 1
        outputLayer[1].phase = 2
        outputLayer[2].phase = 3
        outputLayer[3].phase = 4
        outputLayer[4].phase = 5

        # Check values
        self.assertEqual(outputLayer[0].current, 1)
        self.assertEqual(outputLayer[0].voltage, 2)
        self.assertEqual(outputLayer[0].activity, 3)
        self.assertEqual(outputLayer[0].biasMant, 4)
        self.assertEqual(outputLayer[0].biasExp, 5)
        self.assertEqual(outputLayer[0].phase, 1)
        self.assertEqual(outputLayer[1].phase, 2)
        self.assertEqual(outputLayer[2].phase, 3)
        self.assertEqual(outputLayer[3].phase, 4)
        self.assertEqual(outputLayer[4].phase, 5)

    def test_CompartmentInterface_probe(self):
        """Check probe generation."""

        # Create an arbitrary model
        inputShape = (16, 16, 1)
        inputLayer = NxInputLayer(inputShape)
        outputLayer = NxConv2D(1, 3, padding='same',
                               validatePartitions=True)
        model = NxModel(inputLayer.input, outputLayer(inputLayer.input),
                        numCandidatesToCompute=3)

        model.compileModel()

        uProbe = outputLayer[0].probe(ProbableStates.CURRENT)
        sProbe = outputLayer[0].probe(ProbableStates.SPIKE)
        aProbe = outputLayer[0].probe(ProbableStates.ACTIVITY)
        pProbe = outputLayer[2].probe(ProbableStates.PHASE)

        self.assertTrue(isinstance(uProbe, N2Probe))
        self.assertTrue(isinstance(sProbe, N2SpikeProbe))
        self.assertTrue(isinstance(aProbe, N2Probe))
        self.assertTrue(isinstance(pProbe, N2Probe))

        self.assertEqual(uProbe.chipId, outputLayer._cxResourceMap[0, 0])
        self.assertEqual(uProbe.coreId, outputLayer._cxResourceMap[0, 1])
        self.assertEqual(uProbe.nodeId, outputLayer._cxResourceMap[0, 2])

        self.assertEqual(sProbe.chipId, outputLayer._cxResourceMap[0, 0])
        self.assertEqual(sProbe.coreId, outputLayer._cxResourceMap[0, 1])
        self.assertEqual(sProbe.cxId, outputLayer._cxResourceMap[0, 2])

        self.assertEqual(aProbe.chipId, outputLayer._cxResourceMap[0, 0])
        self.assertEqual(aProbe.coreId, outputLayer._cxResourceMap[0, 1])
        self.assertEqual(aProbe.nodeId, outputLayer._cxResourceMap[0, 2])

        self.assertEqual(pProbe.chipId, outputLayer._cxResourceMap[2, 0])
        self.assertEqual(pProbe.coreId, outputLayer._cxResourceMap[2, 1])
        self.assertEqual(pProbe.nodeId, outputLayer._cxResourceMap[2, 2] // 4)


def main():
    support.run_unittest(TestSynapseEncoding)
    support.run_unittest(TestNxConv2D)
    support.run_unittest(TestNxAveragePooling2D)
    support.run_unittest(TestNxDepthwiseConv2D)
    support.run_unittest(TestNxConv1D)
    support.run_unittest(TestNxDense)
    support.run_unittest(TestNxModel)
    support.run_unittest(TestCompartmentInterface)


if __name__ == '__main__':
    main()
