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

"""
This module contains deep neural network layer objects derived from Keras
and augmented with functionality for deployment on Loihi hardware.
"""

import collections
import logging
import os
import shutil
import time
from abc import abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, \
    AveragePooling2D, Flatten, Conv1D, ZeroPadding2D, Reshape, InputLayer
from tensorflow.keras.models import Model, load_model
from scipy.sparse import lil_matrix, load_npz, save_npz

# Import plotting modules first to ensure plt backend is set correctly.
from nxsdk_modules_ncl.dnn.src.plotting import plot_multiplicity, \
    plot_coreIdMap, plot_core_occupancy, plot_layer_partition, \
    plot_core_utilization, plot_exclusion_criteria_hit_count, \
    visualize_partitions, plot_cost_graph, plot_cost_terms, plot_cx_syn
from nxsdk_modules_ncl.dnn.src.data_structures import Layer, Partition, \
    SynapseGroup, OutputAxonGroup, CompartmentGroup, InputAxonGroup, \
    serializeLayer, deserializeLayer
from nxsdk_modules_ncl.dnn.src.dnn_mapper import DnnMapper
from nxsdk_modules_ncl.dnn.src.optimization import ExclusionCriteria, \
    PartitionOptimizer, getDummyLayer
from nxsdk_modules_ncl.dnn.src.synapse_compression import SynapseEncoder, \
    compressSynFmts, remapSynEntries, reconstructKMapFromPartitions
from nxsdk_modules_ncl.dnn.src.utils import getWeightsFromIds, _interleave, \
    _getMultiplicityMapConvlike, _getUniqueSourceGroups, _getSizeInterleaved, \
    _getDestinationGroups, _getPadding, _genKernelIdMap, \
    getPartitionCandidates, getS, logMemTime
from nxsdk.graph.nxboard import N2Board
from nxsdk.logutils.nxlogging import get_logger

if TYPE_CHECKING:
    from nxsdk.arch.n2a.graph.n2acore import N2ACore
    from nxsdk.graph.monitor.probes import Probe, ProbeCondition
    from nxsdk_modules_ncl.dnn.src.data_structures import SynEntry


NX_KWARGS = ['numWeightBits', 'synapseEncoding', 'biasExp', 'vThMant',
             'weightExponent', 'useSharedSign', 'visualizePartitions',
             'validatePartitions', 'logger', 'probeSpikes', 'threshOp',
             'saveOutput', 'compartmentKwargs', 'connectionKwargs',
             'resetMode', 'weightExpSoftReset', 'numBiasBits', '_padding',
             '_zeroPadding', '_signed']

# Todo: Add and test soft-reset for all spiking layers.
SOFT_RESET_LAYERS = {'NxConv2D', 'NxInputLayer', 'NxConv1D',
                     'NxDepthwiseConv2D', 'NxAveragePooling2D',
                     'NxDense'}

CxAddr = collections.namedtuple('CxAddr', ['chipId', 'coreId', 'cxId'])


def fix_input_layer_shape(shape):
    """
    tf.keras.models.load_model function introduced a bug that wraps the input
    tensors and shapes in a single-entry list, i.e.
    output_shape == [(None, 1, 28, 28)]. Thus we have to apply [0] here.
    """

    if len(shape) == 1:
        return shape[0]
    return shape


def removeNxKwargs(kwargs):
    """Remove keyword arguments not understood by Keras layer constructors.

    :param dict kwargs: Keyword arguments to NxLayers.

    :return: Possibly reduced set of keyword arguments.
    :rtype: dict
    """

    for kwarg in NX_KWARGS:
        kwargs.pop(kwarg, None)

    return kwargs


class ProbableStates(IntEnum):
    """Enumeration of states that can be probed in Loihi compartments."""

    CURRENT = 0
    VOLTAGE = 1
    SPIKE = 2
    ACTIVITY = 3
    PHASE = 4


class CompartmentInterface:
    """Interface to Loihi compartments.

    Facilitates setting registers interactively via NxCore.

    :param N2Board board: The board the compartment belongs to.
    :param CxAddr addr: The compartment address tuple (chip, core, cx).
    """

    __slots__ = ["_board", "_addr"]

    def __init__(self, board, addr):

        self._board = board
        self._addr = addr

    def _getCore(self):
        """Return core associated with address.

        :rtype: N2ACore
        """

        return \
            self._board.chipMap[self._addr.chipId].coreMap[self._addr.coreId]

    @property
    def _cxGroupId(self):
        """Return the compartment group id."""

        return self._addr.cxId // 4

    @property
    def _cxIdMod4(self):
        """Return the compartment id modulo 4."""

        return self._addr.cxId % 4

    @property
    def current(self):
        """Return current of compartment."""

        reg = self._getCore().cxState[self._addr.cxId]
        reg.fetch()
        return reg.u

    @current.setter
    def current(self, val):
        """Set current of compartment."""

        self._getCore().cxState[self._addr.cxId].u = val

    @property
    def voltage(self):
        """Return voltage of compartment."""

        reg = self._getCore().cxState[self._addr.cxId]
        reg.fetch()
        return reg.v

    @voltage.setter
    def voltage(self, val):
        """Set voltage of compartment."""

        self._getCore().cxState[self._addr.cxId].v = val

    @property
    def activity(self):
        """Return soma trace of compartment."""

        reg = self._getCore().somaState[self._addr.cxId]
        reg.fetch()
        return reg.a

    @activity.setter
    def activity(self, val):
        """Set somaTrace of compartment."""

        self._getCore().somaState[self._addr.cxId].a = val

    @property
    def biasMant(self):
        """Return bias mantissa of compartment."""

        reg = self._getCore().cxCfg[self._addr.cxId]
        reg.fetch()
        return reg.bias

    @biasMant.setter
    def biasMant(self, val):
        """Set bias mantissa of compartment."""

        self._getCore().cxCfg[self._addr.cxId].bias = val

    @property
    def biasExp(self):
        """Return bias exponent of compartment."""

        reg = self._getCore().cxCfg[self._addr.cxId]
        reg.fetch()
        return reg.biasExp

    @biasExp.setter
    def biasExp(self, val):
        """Set bias exponent of compartment."""

        self._getCore().cxCfg[self._addr.cxId].biasExp = val

    @property
    def phase(self):
        """Return phase of compartment."""

        cxGroupId = self._cxGroupId
        cxIdMod4 = self._cxIdMod4
        reg = self._getCore().cxMetaState[cxGroupId]
        reg.fetch()
        if cxIdMod4 == 0:
            return reg.phase0
        elif cxIdMod4 == 1:
            return reg.phase1
        elif cxIdMod4 == 2:
            return reg.phase2
        else:
            return reg.phase3

    @phase.setter
    def phase(self, val):
        """Set phase of compartment."""

        cxGroupId = self._cxGroupId
        cxIdMod4 = self._cxIdMod4
        reg = self._getCore().cxMetaState[cxGroupId]
        if cxIdMod4 == 0:
            reg.phase0 = val
        elif cxIdMod4 == 1:
            reg.phase1 = val
        elif cxIdMod4 == 2:
            reg.phase2 = val
        else:
            reg.phase3 = val

    def probe(self, state, probeCondition=None):
        """Generates a probe for given state and with given probeCondition.

        .. note:: Spike probes should only be added to layers without shared
                  output axons like output layer of CNN because spike probes
                  are incompatible with neurons already having shared axons.
                  This requirement is currently not enforced!

        :param ProbableStates state: Enum that specifies which state to probe.
        :param ProbeCondition probeCondition: Condition under which to probe
            state from package 'nxsdk.graph.monitor.probes'. By default, state
            is probed every time step.

        :return: Generated probe.
        :rtype: Probe
        """

        mon = self._board.monitor
        if state is ProbableStates.CURRENT:
            prb = mon.probe(self._getCore().cxState, [self._addr.cxId],
                            'u', probeCondition)
        elif state is ProbableStates.VOLTAGE:
            prb = mon.probe(self._getCore().cxState, [self._addr.cxId],
                            'v', probeCondition)
        elif state is ProbableStates.SPIKE:
            prb = mon.probe(self._getCore().cxState, [self._addr.cxId],
                            'spike', probeCondition)
        elif state is ProbableStates.ACTIVITY:
            prb = mon.probe(self._getCore().somaState, [self._addr.cxId],
                            'a', probeCondition)
        elif state is ProbableStates.PHASE:
            reg = self._getCore().cxMetaState
            phases = ('phase0', 'phase1', 'phase2', 'phase3')
            prb = mon.probe(reg, [self._cxGroupId],
                            phases[self._cxIdMod4], probeCondition)
        else:
            raise Exception("Illegal state to probe: {}".format(state))

        return prb[0]


def compileConvlike(self, partitionCandidate):
    """Compile partition candidate for a convolution-like layer.

    Tries to create compartment, synapse and axon groups for each partition in
    the layer.

    Returns ``None`` if the partition candidate exceed some resource limit.

    :param NxConv2D | NxDepthwiseConv2D | NxAveragePooling2D self: Nx layer.
    :param Layer partitionCandidate: The layer partition candidate.

    :return: ``partitionCandidate``.
    :rtype: Layer | None
    """

    # Input variables #
    ###################

    inputShape = self.input_shape[1:]

    # When using signed spikes the number of channels in the input
    # is doubled.
    signedInput = False
    if len(self._inbound_nodes[:1]):
        layer = self._inbound_nodes[0].inbound_layers
        # layer may be wrapped in a list of lenght 1:
        if isinstance(layer, (list, tuple)):
            layer = layer[0]
        if hasattr(layer, 'signed'):
            if layer.signed:
                signedInput = True
                inputShape = inputShape[:-1] + (2 * inputShape[-1],)

    # Subtract zero padding from inputShape if previous layer is ZeroPadding.
    # Not used in actual spiking layer.
    if self._zeroPadding is not None:
        py0, py1, px0, px1 = self._zeroPadding
        inputShape = (inputShape[0] - (py0 + py1), inputShape[1] - (px0 + px1),
                      inputShape[2])

    inputSize = np.asscalar(np.prod(inputShape))
    layerShape = partitionCandidate.coreIdMap.shape

    layerType = self.__class__.__name__
    isDepthwise = 'Depthwise' in layerType or 'Pooling' in layerType

    connKwargs = partitionCandidate.connectionKwargs
    limits = self.exclusionCriteria

    # Get number of cores used by network so far. Needed for Partition
    # object.
    coreCount = self.getTotalCoreCount(partitionCandidate)

    # Flatten multiplicity map of lower layer and repeat for each channel
    # in feature map. Needed for input axons.
    preMultiplicityFlat = np.tile(np.ravel(
        partitionCandidate.multiplicityMap, 'F'), inputShape[-1])
    # Flatten multiplicity map of current layer and repeat for each channel
    # in feature map. Needed for output axons.
    multiplicityFlat = np.tile(np.ravel(
        partitionCandidate.postLayer.multiplicityMap, 'F'), layerShape[-1])

    # Todo : enable population mode in subtractive-reset. Currently only
    # support for discrete axons.
    if self.resetMode == 'soft':
        preMultiplicityFlat = 2 * np.ones_like(preMultiplicityFlat)
        multiplicityFlat = 2 * np.ones_like(multiplicityFlat)

    # Get weights. Needed for SynEntries.
    weights, biases = self.get_weights()

    # When using signed input spikes, negated weights are
    # concatenated to the input channel dimension.
    if signedInput:
        weights = np.concatenate([weights, -weights], axis=-2)
    weights = np.ravel(weights, 'F')

    # Repeat biases for every neuron in channel. Needed for
    # CompartmentGroup.
    biasesFlat = np.repeat(biases.astype(int), np.prod(layerShape[:-1]))
    biasExp = np.ones_like(biasesFlat) * \
        partitionCandidate.compartmentKwargs['biasExp']

    # Iterate over cores #
    ######################

    for relCoreId in range(partitionCandidate.numCores):

        # Get subsection of layer corresponding to current core.
        mask = partitionCandidate.coreIdMap == relCoreId
        coreShape = [max(n) - min(n) + 1 for n in np.nonzero(mask)]
        maskFlat = np.ravel(mask, 'F')
        relToAbsDestCxIdxMap = np.nonzero(maskFlat)[0]

        # kMap is the unrolled convolution operator (doubly blocked Toeplitz
        # matrix), with shape (numCxOnCore, numNeuronsInput).
        kMap = lil_matrix((np.count_nonzero(maskFlat), inputSize), dtype=int)
        kMap.rows = self.kernelIdMap.rows[maskFlat]
        kMap.data = self.kernelIdMap.data[maskFlat]

        # The neuronSize is the number of compartments per neuron.
        neuronSize = 2 if self.resetMode == 'soft' else 1

        # Interleaving #
        ################

        destinationGroups, cxBaseOffsets = \
            _getDestinationGroups(kMap, coreShape, inputShape, isDepthwise)

        numDestinationGroups = len(destinationGroups)

        if numDestinationGroups * neuronSize > limits.maxNumDestinationGroups:
            limits.numDestinationGroups += 1
            return

        # Compute interleaved size already before interleaving, so we can skip
        # costly interleaving if size too large.

        coreSizeInterleaved = _getSizeInterleaved(coreShape, destinationGroups,
                                                  neuronSize)
        # Account for compartments in subtractive-reset mode.
        srCompartments = 0
        if self.resetMode == 'soft':
            srCompartments = (coreSizeInterleaved % 4)

        if coreSizeInterleaved + srCompartments > limits.maxNumCompartments:
            limits.coreSizeInterleaved += 1
            return

        interleavedMap, permutedDestCxIdxs = _interleave(kMap, coreShape,
                                                         destinationGroups)
        assert max(permutedDestCxIdxs) <= coreSizeInterleaved

        permCxIdToRelCxId = np.array([np.where(permutedDestCxIdxs == i)[0]
                                      for i in range(coreSizeInterleaved)])

        cIdxMult = numDestinationGroups - 1

        # uniqueSourceGroups contains the unique axons of mkMap. It has shape
        # (numUniqueSourceGroups, numCxOnCore).
        # synListPtr is a list containing the indices of uniqueSourceGroups
        # that reconstruct mkMap (length: inputSize).
        uniqueSourceGroups, synListPtrs = _getUniqueSourceGroups(
            interleavedMap, cxBaseOffsets, inputShape[0])

        # Compute number of synaptic resources
        synapseEncoder = SynapseEncoder(connKwargs['numWeightBits'],
                                        limits.maxNumSynPerSynEntry,
                                        connKwargs['synapseEncoding'],
                                        connKwargs['useSharedSign'])

        # Initialize Partition #
        ########################
        chipCounter = self.incrementChipCounter(coreCount, relCoreId)

        partition = Partition(relCoreId, chipCounter, coreSizeInterleaved,
                              partitionCandidate, resetMode=self.resetMode)

        # Output Axons to post-layer #
        ##############################

        # Skip for output layer (has no output axons).
        srcIdMap = partitionCandidate.postLayer.srcIdMap
        multiplicityInterleaved = np.zeros(coreSizeInterleaved, int)
        multiplicityInterleaved[permutedDestCxIdxs] = \
            multiplicityFlat[relToAbsDestCxIdxMap]

        if len(srcIdMap):
            inverseMap = {}
            # Iterate over compartments of current core, which correspond
            # to the srcCxIds of the next postLayer.
            for relCxId, cxId in enumerate(relToAbsDestCxIdxMap):
                # If the convolution does not cover the entire range of srcIds,
                # skip that cxId (has no connections). This occurs for instance
                # with input shape (28, 28), kernel shape (5, 5), strides
                # (2, 2), and padding 'valid'.
                if cxId not in srcIdMap:
                    continue
                for inputAxonGroup, relSrcIdPost in srcIdMap[cxId]:
                    if inputAxonGroup not in inverseMap.keys():
                        inverseMap[inputAxonGroup] = [[], []]
                    interleavedCxId = permutedDestCxIdxs[relCxId]
                    inverseMap[inputAxonGroup][0].append(interleavedCxId)
                    inverseMap[inputAxonGroup][1].append(relSrcIdPost)

            # Build outputAxons based on srcIdMap of next higher layer.
            for inputAxonGroup, (cxIds, relSrcIds) in inverseMap.items():
                # Shift cxIds to soma compartment for multi-compartment neurons
                somaCxIds = np.array(cxIds)
                if self.resetMode == 'soft':
                    somaCxIds *= 2
                    somaCxIds += 1

                partition.addOutputAxonGroup(OutputAxonGroup(
                    somaCxIds, multiplicityInterleaved[cxIds],
                    np.array(relSrcIds), inputAxonGroup, partition))

            if partition.numOutputAxonCfgEntries > limits.maxNumAxons:
                limits.numOutputAxons += 1
                return
        else:
            # If output layer, create synapse groups (needed for soft reset).
            inverseMap = {relCxId: ([cxId], [0]) for relCxId, cxId
                          in enumerate(permutedDestCxIdxs)}

        # SynEntries & SynFmts #
        ########################

        synEntriesOfCore = []
        for uniqueSourceGroup in uniqueSourceGroups:

            synEntriesOfSourceGroup = _encodeSynapseGroup(
                uniqueSourceGroup, weights, cIdxMult, synapseEncoder,
                neuronSize)

            synEntriesOfCore.append(synEntriesOfSourceGroup)

        # Create synEntries and synFmts for recurrent connections.
        mappedCxIds = {}
        if self.resetMode == 'soft':
            # To ensure only one recurrent connection per neuron, duplicated
            # recurrent connections receive 0 weight.
            cxIdMap = {}
            synGrpId = len(uniqueSourceGroups)
            for _, (cxIds, relSrcIds) in inverseMap.items():
                cxIdHash = hash(tuple(cxIds))
                if cxIdHash in mappedCxIds:
                    continue
                mappedCxIds[cxIdHash] = (synGrpId, cxIds, relSrcIds)
                synGrpId += 1
                synEntriesOfRecurrentGroup = []
                for cxId, relSrcId in zip(cxIds, relSrcIds):
                    wgt = -connKwargs['weightMantSR']
                    if cxId in cxIdMap:
                        wgt = 0
                    cxIdMap[cxId] = 1
                    # wgt is set by mapper based on vThMant and scale
                    synapseEncoder.encode(np.array([cxId], int) * 2,
                                          np.array([wgt], int),
                                          0,
                                          0,
                                          np.array([1], int),
                                          softReset=True)

                    synEntriesOfRecurrentGroup.append(
                        synapseEncoder.popSynEntries())

                synEntriesOfCore.append(synEntriesOfRecurrentGroup)

        synFmts, synEntryMap = compressSynFmts(synapseEncoder.getSynFmts(),
                                               limits.maxNumSynFmt)

        if len(synFmts) > limits.maxNumSynFmt:
            limits.numSynFmts += 1
            return

        synEntriesOfCore = remapSynEntries(synEntriesOfCore, synFmts,
                                           synEntryMap)

        for synFmt in synFmts:
            partition.addSynFmt(synFmt)

        # Synapse Groups #
        ##################

        synMemOfCore = 0
        for i, synEntriesOfSourceGroup in enumerate(synEntriesOfCore):

            # Create new synapse group.
            synapseGroup = SynapseGroup(i, synEntriesOfSourceGroup)
            # The longest block of registers in synMem may not exceed 256 words
            # for one axon. If axon is shared, only need to find the maximum
            # number of words used up by any of the neurons in the population.
            if synapseGroup.maxSynMemLen > limits.maxNumSynMemWordsPerAxon:
                limits.synMemPerAxon += 1
                return

            partition.addSynapseGroup(synapseGroup)
            synMemOfCore += synapseGroup.numSynMemWords

        if synMemOfCore >= limits.maxNumSynMemWords:
            limits.numSynMemWords += 1
            return

        srcIdsOfCore = np.unique(np.concatenate(kMap.rows))

        # Input Axons from pre-layer #
        ##############################

        # Loop over all sourceGroups. synListPtr gets the unique list of
        # synaptic entries belonging to the sourceGroup. This
        # synEntriesOfSourceGroup may be shared with other sourceGroups
        # (distinguished via cxBaseOffset).
        sourceGroupSize = inputShape[0]
        for axonId in sorted(synListPtrs):
            synListPtr = synListPtrs[axonId]
            synapseGroup = partition.synapseGroups[synListPtr]

            # Get subset of srcIds that have connections in current core.
            _srcIds = np.arange(sourceGroupSize) + sourceGroupSize * axonId
            srcIds = np.intersect1d(srcIdsOfCore, _srcIds, True)
            assert isinstance(srcIds, np.ndarray)

            inputAxonGroup = InputAxonGroup(
                srcIds, preMultiplicityFlat[srcIds], synapseGroup,
                cxBaseOffsets[axonId] * neuronSize, partition)

            partition.addInputAxonGroup(inputAxonGroup)

            # Build srcIdMap
            for relSrcId, srcId in enumerate(srcIds):
                partitionCandidate.updateSrcIdMap(srcId,
                                                  (inputAxonGroup, relSrcId))

        # Input & output axons for recurrent group #
        ############################################
        if self.resetMode == 'soft':
            for (synGrpId, cxIds, relSrcIds) in mappedCxIds.values():
                synapseGroup = partition.synapseGroups[synGrpId]
                srcIds = np.array([permCxIdToRelCxId[cxId]
                                   for cxId in cxIds])
                inputAxonGroup = InputAxonGroup(
                    srcIds, multiplicityFlat[srcIds], synapseGroup,
                    0, partition)

                partition.addInputAxonGroup(inputAxonGroup)

                # Shift cxIds to soma compartment for multi-compartment neurons
                somaCxIds = np.array(cxIds) * 2
                somaCxIds += 1

                partition.addOutputAxonGroup(OutputAxonGroup(
                    somaCxIds, multiplicityInterleaved[cxIds],
                    np.array(relSrcIds), inputAxonGroup, partition))

        if partition.numOutputAxonCfgEntries > limits.maxNumAxons:
            limits.numOutputAxons += 1
            return

        if partition.numInputAxons > limits.maxNumAxons:
            limits.numInputAxons += 1
            return

        # Compartment Group #
        #####################

        biasesOfCore = biasesFlat[relToAbsDestCxIdxMap]
        biasExpOfCore = biasExp[relToAbsDestCxIdxMap]

        if self.resetMode == 'soft':
            numCx = len(permutedDestCxIdxs) * 2
            # Set biases for dendrite and soma compartments
            tempBiases = np.zeros(numCx, int)
            tempBiasExp = np.zeros(numCx, int)
            tempRelCxIds = np.zeros(numCx, int)
            tempCxIds = np.zeros(numCx, int)

            for relCxId, interleavedCxId in enumerate(permutedDestCxIdxs):
                cxId = relToAbsDestCxIdxMap[relCxId]

                # Dendrite
                tempBiases[relCxId * 2] = biasesOfCore[relCxId]
                tempBiasExp[relCxId * 2] = biasExpOfCore[relCxId]
                tempRelCxIds[relCxId * 2] = interleavedCxId * 2
                tempCxIds[relCxId * 2] = cxId * 2

                # soma
                tempBiases[relCxId * 2 + 1] = 0
                tempBiasExp[relCxId * 2 + 1] = 0
                tempRelCxIds[relCxId * 2 + 1] = interleavedCxId * 2 + 1
                tempCxIds[relCxId * 2 + 1] = cxId * 2 + 1

            permutedDestCxIdxs = tempRelCxIds
            relToAbsDestCxIdxMap = tempCxIds

            # Add dummy neurons
            outputSize = np.prod(self.output_shape[1:]).astype(int)
            maxCxId = max(permutedDestCxIdxs) + 1
            mod = maxCxId % 4
            permutedDestCxIdxs = np.concatenate(
                [permutedDestCxIdxs,
                 [maxCxId + i for i in range(mod)]]).astype(int)
            cxIds = [outputSize * neuronSize + self._dummyCxSize + i
                     for i in range(mod)]
            relToAbsDestCxIdxMap = np.concatenate([relToAbsDestCxIdxMap,
                                                   cxIds]).astype(int)
            biasesOfCore = np.concatenate([tempBiases,
                                           np.zeros(mod)]).astype(int)
            biasExpOfCore = np.concatenate([tempBiasExp,
                                            np.zeros(mod)]).astype(int)
            self._dummyCxSize += mod
            coreSizeInterleaved += mod

        biasesInterleaved = np.zeros(coreSizeInterleaved, int)
        biaseExpInterleaved = np.zeros(coreSizeInterleaved, int)
        biasesInterleaved[permutedDestCxIdxs] = biasesOfCore
        biaseExpInterleaved[permutedDestCxIdxs] = biasExpOfCore

        cxGroup = CompartmentGroup(permutedDestCxIdxs, biasesInterleaved,
                                   biaseExpInterleaved, relToAbsDestCxIdxMap)

        partition.addCompartmentGroup(cxGroup)

        partitionCandidate.addPartition(partition)

    return partitionCandidate


def _encodeSynapseGroup(synapseGroup, weights, cIdxMult, synapseEncoder,
                        neuronSize):
    """Encode single SynapseGroup into list of synEntries.

    :param list[(np.ndarray, np.ndarray)] synapseGroup: Synapses to encode.
    :param np.ndarray weights: Synaptic weights.
    :param int cIdxMult: Multiplier for compartment indices.
    :param SynapseEncoder synapseEncoder: Helper class to encode synapses.
    :param int neuronSize: The number of compartments per neuron.
    :return: List of list of SynEntry objects. The outer list has one entry for
        each neuron in the group; the inner list may contain multiple SynEntry
        objects per neuron.
    :rtype: list[list[SynEntry]]
    """

    synEntries = []

    # Loop over single source neurons.
    for synNeuron in synapseGroup:

        # synIds have cxBase subtracted, but are spaced by cIdxMult and
        # shifted by cIdxOffset.
        synVals = synNeuron[0]
        synIds = synNeuron[1]
        if cIdxMult == 0:
            wgts = getWeightsFromIds(weights, synVals)
            synapseEncoder.encode(synIds * neuronSize, wgts, 0, 0, synVals)
        else:
            cIdxOffsets = np.mod(synIds, cIdxMult + 1)
            assert not np.any(cIdxOffsets > 1)
            for cIdxOffset in [0, 1]:
                mask = cIdxOffsets == cIdxOffset
                if np.any(mask):
                    cxIds = synIds[mask]
                    cxIds = (cxIds - cIdxOffset) // (cIdxMult + 1)
                    kIds = synVals[mask]
                    wgts = getWeightsFromIds(weights, kIds)
                    synapseEncoder.encode(cxIds, wgts,
                                          cIdxOffset * neuronSize,
                                          (cIdxMult + 1) * neuronSize - 1,
                                          kIds)

        synEntries.append(synapseEncoder.popSynEntries())

    return synEntries


def validatePartitionConvlike(partitionCandidate, kernelIdMap):
    """Validate layer partition of a convolution-like layer.

    :raise AssertionError: If reconstructed kernelIdMap does not equal the
        original map.

    :param Layer partitionCandidate: The partitioned layer.
    :param lil_matrix kernelIdMap: Original kernelIdMap.
    """

    kernelIdMap2 = reconstructKMapFromPartitions(partitionCandidate.partitions,
                                                 kernelIdMap.shape)
    assert np.array_equal(kernelIdMap.data + kernelIdMap.rows,
                          kernelIdMap2.data + kernelIdMap2.rows), \
        "Could not reconstruct the kernelIdMap from the axons generated " \
        "during partitioning."


def visualizePartitionConvlike(path, name, partitionCandidate, coreIdMap,
                               coreOccupancy, isFinal, **kwargs):
    """Visualize layer partition of a convolution-like layer.

    :param str path: Where to save plots.
    :param str name: Name of layer.
    :param Layer partitionCandidate: Partitioned layer.
    :param np.ndarray coreIdMap: Tensor of layer shape. Maps neurons to their
        corresponding core id.
    :param np.ndarray coreOccupancy: The number of compartments per core.
    :param bool isFinal: Whether this layer is the output layer. If ``True``,
        certain plots are not available.
    """

    path = os.path.abspath(os.path.join(path, name))
    if not os.path.exists(path):
        os.makedirs(path)

    plot_coreIdMap(coreIdMap, path, name)

    plot_core_occupancy(coreOccupancy, path, name)

    if 'multiplicityMap' in kwargs:
        plot_multiplicity(kwargs['multiplicityMap'], path, name)

    if not isFinal:
        plot_layer_partition(partitionCandidate, path)


class NxLayer(object):
    """Base class for DNN layers on Loihi.

    :param int numWeightBits: Number of bits to use for weights. Default: 8.
    :param str synapseEncoding: Compression mode to use for synapses. Default:
        'sparse'. Other possibilities: 'runlength', 'dense1', 'dense2'.
    :param int biasExp: Bias exponent. Default: 0.
    :param int vThMant: Threshold mantissa. Default: 2 ** 9.
    :param int weightExponent: Weight exponent. Default: 0.
    :param bool useSharedSign: If ``True`` (default), inhibitory and excitatory
        connections are stored separately and use a shared sign.
    :param bool visualizePartitions: If ``True``, plot the partition layout.
        Default: ``False`` (may take several minutes for large layers.)
    :param bool validatePartitions: If ``True``, run a check that the synapse
        and axon generation is valid. Default: ``False`` (for efficiency).
    :param logging.Logger logger: Logger.
    :param bool probeSpikes: If ``True``, record spikes. Default: ``False``
        (probing slows down run time and can lead to memory issues).
    :param int threshOp: Specifies operation to apply when compartment voltage
        exceeds threshold. Default: 0 (spike and reset). If this layer uses
        softmax in the ANN, may want to set threshOp to 3 (no spike and
        saturate voltage at threshold), to avoid overflow but preserve order.
    :param str resetMode: Sets reset mode for rate-coded layers. If 'hard',
        when a neuron spikes the membrane potential is reset to zero. If
        'soft', when a neuron spikes the threshold will be subtracted from the
        membrane threshold.
    """

    def __init__(self, numWeightBits=None, synapseEncoding=None, biasExp=None,
                 vThMant=None, weightExponent=None, useSharedSign=None,
                 visualizePartitions=None, validatePartitions=None,
                 logger=None, probeSpikes=None, threshOp=None, resetMode=None,
                 **kwargs):

        # Set default parameters here instead of in method signature so we do
        # not have to copy them when deriving from the class.
        if numWeightBits is None:
            numWeightBits = 8
        if synapseEncoding is None:
            synapseEncoding = 'sparse'
        if biasExp is None:
            biasExp = 6
        if vThMant is None:
            vThMant = 2 ** 9
        if weightExponent is None:
            weightExponent = 0
        if useSharedSign is None:
            useSharedSign = True
        if visualizePartitions is None:
            visualizePartitions = False
        if validatePartitions is None:
            validatePartitions = False
        if probeSpikes is None:
            probeSpikes = False
        if threshOp is None:
            threshOp = 0
        if resetMode is None:
            resetMode = 'hard'

        assert isinstance(numWeightBits, (int, np.integer))
        assert isinstance(synapseEncoding, str)
        assert isinstance(biasExp, (int, np.integer))
        assert isinstance(vThMant, (int, np.integer))
        assert isinstance(weightExponent, (int, np.integer))
        assert resetMode in ['hard', 'soft']

        self.visualizePartitions = visualizePartitions
        self.validatePartitions = validatePartitions

        if logger is None:
            logger = get_logger("NET.DNN")
        self.logger = logger

        self._board = None
        self._cxResourceMap = None

        self.compartmentKwargs = {'biasExp': biasExp, 'vThMant': vThMant,
                                  'enableSomaTrace': probeSpikes,
                                  'threshOp': threshOp}

        self._compartmentDefaults = {'functionalState': 2,
                                     'compartmentVoltageDecay': 0,
                                     'compartmentCurrentDecay': 4096,
                                     'refractoryDelay': 1}

        self.compartmentKwargs.update(self._compartmentDefaults)

        self.connectionKwargs = {'numWeightBits': numWeightBits,
                                 'weightExponent': weightExponent,
                                 'useSharedSign': useSharedSign,
                                 'synapseEncoding': synapseEncoding}

        self._connectionDefaults = {'delay': 0,
                                    'enableDelay': 0,
                                    'enableLearning': 0,
                                    'numDelayBits': 0,
                                    'numTagBits': 0}

        self.connectionKwargs.update(self._connectionDefaults)

        # Check for softmax activation
        activation = kwargs.get('activation')

        # Add weight exponent parameter for soft-reset mode.

        # Set resetMode
        if self.__class__.__name__ not in SOFT_RESET_LAYERS:
            resetMode = 'hard'
        self._resetMode = resetMode

        if resetMode == 'soft' and activation != 'softmax':
            # The weight mantissa and exponent for the recurrent
            # connections used in subtractive-reset are computed
            # to as close as possible to the vThMant.
            wgtExp = int(np.ceil(np.log2(vThMant / 256)))

            if wgtExp > 7:
                print("WgtExp {} exceeds W_EXP_MAX={}".format(
                    wgtExp, 7))
                wgtExp = 7

                print("vThMant {} exceeds the max value {} which may "
                      "be subtracted using recurrent connections.".format(
                        vThMant, 255*2**7))

            wgtExp = np.max([0, wgtExp])
            wgtMant = int(np.rint(vThMant / 2 ** wgtExp))
            wgtMant = np.max([0, wgtMant])
            wgtMant = np.min([wgtMant, 255])
            self.connectionKwargs.update(
                {'weightExpSR': wgtExp,
                 'weightMantSR': wgtMant})

        if activation == 'softmax':
            # Set threshOp to 3 so that the compartment does not spike
            # and saturates if it reaches threshold
            self.compartmentKwargs['threshOp'] = 3
            # Increase threshold to maximum
            self.compartmentKwargs['vThMant'] = 2**17 - 1
            # Add decay to prevent saturation.
            self.compartmentKwargs['compartmentVoltageDecay'] = 2**8

            if resetMode == 'soft':
                self.connectionKwargs.update(
                    {'weightExpSR': 0,
                     'weightMantSR': 255})

        # Override if passed connection and compartment kwargs.
        # Used when loading an existing NxModel.
        if 'connectionKwargs' in kwargs:
            self.connectionKwargs.update(kwargs['connectionKwargs'])
        if 'compartmentKwargs' in kwargs:
            self.compartmentKwargs.update(kwargs['compartmentKwargs'])

        # Resource constraints
        self.exclusionCriteria = ExclusionCriteria()

        self._kernelIdMap = None
        self._pathToKernelIdMap = None

        # Misc Properties
        self._dummyCxSize = 0

        # Placeholder for zeroPadding of previous layer.
        self._zeroPadding = None

    def get_config(self):
        """Get layer configuration.

        :return: config
        :rtype: dict
        """

        config = {'compartmentKwargs': self.compartmentKwargs,
                  'connectionKwargs': self.connectionKwargs,
                  'resetMode': self.resetMode,
                  }
        return config

    @property
    def kernelIdMap(self):
        """KernelIdMap of layer."""

        # First, check whether the kernelIdMap has already been set.
        if self._kernelIdMap is not None:
            return self._kernelIdMap

        # Next, try loading kernelIdMap from temp dir.
        if self._pathToKernelIdMap is not None:
            try:
                self._kernelIdMap = load_npz(self._pathToKernelIdMap).tolil()
                self.logger.debug("Loaded kernelIdMap from %s.",
                                  self._pathToKernelIdMap)
            except IOError:
                self.logger.debug("Could not load kernelIdMap from %s.",
                                  self._pathToKernelIdMap)
        else:
            tempdir = os.path.abspath(os.path.join(os.path.dirname(
                os.path.realpath(__file__)),
                '../../..', 'temp', str(time.time())))
            os.makedirs(tempdir, exist_ok=True)
            self._pathToKernelIdMap = os.path.join(
                tempdir, 'kernelIdMap_{}.npz'.format(hash(self)))

        # Finally, compute kernelIdMap.
        if self._kernelIdMap is None:
            self.logger.debug("Computing kernelIdMap.")
            kernelIdMap = self.genKernelIdMap()
            self._kernelIdMap = kernelIdMap.tolil()

            self.logger.debug("Storing kernelIdMap in %s.",
                              self._pathToKernelIdMap)
            save_npz(self._pathToKernelIdMap, kernelIdMap)

        return self._kernelIdMap

    def deleteKernelIdMap(self):
        """Remove kernelIdMap to free memory."""

        self._kernelIdMap = None

        if self._pathToKernelIdMap is not None \
                and os.path.exists(self._pathToKernelIdMap):
            shutil.rmtree(os.path.dirname(self._pathToKernelIdMap))

    def genKernelIdMap(self):
        """Generate KernelIdMap."""

        pass

    @abstractmethod
    def compile(self, partitionCandidate):
        """Compile layer.

        :param Layer partitionCandidate: Partitioned layer object.
        """

        pass

    @abstractmethod
    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.
        """

        pass

    @abstractmethod
    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer."""

        pass

    def validatePartition(self, partitionCandidate):
        """Validate layer partition.

        :param Layer partitionCandidate: The partitioned layer.
        """

        pass

    def visualizePartition(self, path, partitionCandidate, coreIdMap,
                           coreOccupancy, **kwargs):
        """Visualize layer partition.

        :param str path: Where to save plots.
        :param Layer partitionCandidate: Partitioned layer.
        :param np.ndarray coreIdMap: Tensor of layer shape. Maps neurons to
            their corresponding core id.
        :param np.ndarray coreOccupancy: The number of compartments per core.
        """

        pass

    @property
    def maxNumCompartments(self):
        """ Return maximum number of compartments per partition"""
        return self.exclusionCriteria.maxNumCompartments

    @maxNumCompartments.setter
    def maxNumCompartments(self, n):
        """Set maxNumCompartments.

        :param int n: Maximum number of compartments.
        """
        self.exclusionCriteria.maxNumCompartments = n

    @property
    def resetMode(self):
        """Threshold reset operation of layer. 'hard' or 'soft'.
        """
        return self._resetMode

    # -------------------------------------------------------------------------
    # State interface (Consider adding specific <compartments>, <weights>
    # properties if we want to support parameters besides those from
    # compartments.
    def __getitem__(self, item):
        """Get CompartmentInterface.

        :param int item: Index.
        :return: CompartmentInterface.
        :rtype: CompartmentInterface
        """

        return CompartmentInterface(self._board, self.getCxAddr(item))

    def setBoardAndCxResourceMap(self, board, rm):
        """Set the boad and cxResourceMap.

        :param N2Board board: The board that the layer is mapped to.
        :param np.ndarray rm: The cxResourceMap mapping from compartment id
            to HW address.
        """

        self._board = board
        # TODO: Determine effect of setting sync=False.
        # sync=False required for composable DNN.
        self._board.sync = False
        self._cxResourceMap = rm

    @property
    def cxResourceMap(self):
        return self._cxResourceMap

    def getCxAddr(self, cxId):
        """Return the compartment address as a (chipId, coreId, cxId) tuple.

        :param int cxId: Global layer-wide id of compartment.

        :return: (chipId, coreId, cxId) tuple.
        :rtype: CxAddr
        """

        assert self._cxResourceMap is not None, \
            "Layer must be compiled to retrieve cxAddr."

        return CxAddr(*tuple(self._cxResourceMap[cxId, :]))

    def incrementChipCounter(self, coreCount, relCoreId):
        """Increment global chip counter.

        This counter is not equivalent to the core and chip id assigned by the
        chip / board during mapping. We need the chip counter before an actual
        chip address is assigned, because this allows checking whether axons go
        to different chips, in which case they cost more. For this check the
        actual addresses are irrelevant, so we can just increase the counter
        linearly.

        :param int coreCount: Total number of cores currently consumed by
            partitioned layers in this network.
        :param int relCoreId: Core id relative to current layer.

        :return: chipCounter
        :rtype int
        """

        return (coreCount + relCoreId) // \
            self.exclusionCriteria.maxNumCoresPerChip

    @staticmethod
    def getTotalCoreCount(partitionCandidate):
        """Get total number of cores currently consumed by partitioned layers.

        Iterates up in the layer hierarchy, starting with, but not including,
        the current partition candidate.

        :param Layer partitionCandidate: Partition candidate.
        """

        coreCount = 0
        postLayer = partitionCandidate.postLayer
        while True:
            if postLayer.postLayer is None:
                return coreCount
            coreCount += postLayer.numCores
            postLayer = postLayer.postLayer

    def disableUpdates(self):
        """
        Disables compartment updates for each core in NxLayer.

        """
        if self.cxResourceMap is None:
            return
        for chipId, coreId in np.unique(self.cxResourceMap[:, :2], axis=0):
            core = self._board.chipMap[chipId].coreMap[coreId]
            core.numUpdates[0].configure(numUpdates=0)

    def enableUpdates(self):
        """
        Enables compartment updates for each core in NxLayer.

        """

        if self.cxResourceMap is None:
            return
        for chipId, coreId in np.unique(self.cxResourceMap[:, :2], axis=0):
            inds = np.all(self.cxResourceMap[:, :2] ==
                          np.array([chipId, coreId]), axis=1)
            maxCxId = np.max(self.cxResourceMap[inds][:, 2])
            numCxGroups = int(np.ceil((maxCxId + 1) / 4))
            core = self._board.chipMap[chipId].coreMap[coreId]
            core.numUpdates[0].configure(numUpdates=numCxGroups)


class NxConv2D(NxLayer, Conv2D):
    """2D convolution layer for Loihi DNNs."""

    def __init__(self, filters, kernel_size, strides=None, padding=None,
                 numWeightBits=None, synapseEncoding=None, biasExp=None,
                 vThMant=None, weightExponent=None, useSharedSign=None,
                 visualizePartitions=None, validatePartitions=None,
                 logger=None, probeSpikes=None, **kwargs):

        NxLayer.__init__(self, numWeightBits, synapseEncoding, biasExp,
                         vThMant, weightExponent, useSharedSign,
                         visualizePartitions, validatePartitions, logger,
                         probeSpikes, **kwargs)

        # Have to copy Keras default arguments here.
        if strides is None:
            strides = 1
        if padding is None:
            padding = 'valid'

        kwargs = removeNxKwargs(kwargs)

        Conv2D.__init__(self, filters, kernel_size, strides=strides,
                        padding=padding, **kwargs)

        if self.dilation_rate != (1, 1):
            raise NotImplementedError

        # Padding tuple
        self._padding = kwargs.get('_padding', None)
        self._zeroPadding = kwargs.get('_zeroPadding', None)

    def get_config(self):
        config = {'_padding': self.padding,
                  '_zeroPadding': self._zeroPadding}
        baseConfig = Conv2D.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def build(self, input_shape):
        Conv2D.build(self, input_shape)

        self._padding = _getPadding(input_shape.as_list()[1:], self.padding,
                                    self.kernel_size, self.strides,
                                    self.dilation_rate)

    def genKernelIdMap(self):
        inputShape = self.input_shape[1:]
        # When using signed spikes the number of channels in the input
        # is doubled.
        if len(self._inbound_nodes[:1]):
            layer = self._inbound_nodes[0].inbound_layers
            # layer may be wrapped in a list of lenght 1:
            if isinstance(layer, (list, tuple)):
                layer = layer[0]
            if hasattr(layer, 'signed'):
                if layer.signed:
                    inputShape = inputShape[:-1] + (2 * inputShape[-1],)
        return _genKernelIdMap(inputShape, self.output_shape[1:],
                               self._padding, self.strides, self.kernel_size,
                               self.dilation_rate,
                               zeroPadding=self.zeroPadding)

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Multiplicity map, which has the same shape as the input to
            current layer except that the channel axis is removed. Each entry
            specifies to how many cores that neuron sends its spikes.
        :rtype: np.ndarray
        """

        multiplicityMap = _getMultiplicityMapConvlike(
            coreIdMap, self.input_shape[1:], self.kernel_size, self.strides,
            self._padding, self.dilation_rate, self.zeroPadding)

        # If we partition a normal convolution layer along the channel
        # dimension, each input neuron has to be duplicated for every new core.
        # This is not necessary in a DepthwiseConv layer.
        multiplicityMap *= len(set(coreIdMap[0, 0, :]))

        return multiplicityMap

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 3). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x3 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments, order=[2, 0, 1])

    def compile(self, partitionCandidate):
        return compileConvlike(self, partitionCandidate)

    def validatePartition(self, partitionCandidate):
        """Validate layer partition.

        :raise AssertionError: If reconstructed kernelIdMap does not equal the
            original map.

        :param Layer partitionCandidate: The partitioned layer.
        """

        if self.validatePartitions:
            validatePartitionConvlike(partitionCandidate, self.kernelIdMap)

    def visualizePartition(self, path, partitionCandidate, coreIdMap,
                           coreOccupancy, **kwargs):
        """Visualize layer partition.

        :param str path: Where to save plots.
        :param Layer partitionCandidate: Partitioned layer.
        :param np.ndarray coreIdMap: Tensor of layer shape. Maps neurons to
            their corresponding core id.
        :param np.ndarray coreOccupancy: The number of compartments per core.
        """

        if self.visualizePartitions:
            isFinal = partitionCandidate.postLayer.postLayer is None
            visualizePartitionConvlike(path, self.name, partitionCandidate,
                                       coreIdMap, coreOccupancy, isFinal,
                                       **kwargs)

    @property
    def zeroPadding(self):
        """Zero padding of previous layer. Replaces ZeroPadding Layers."""
        return self._zeroPadding

    @zeroPadding.setter
    def zeroPadding(self, zeroPadding):
        """Set zeroPadding and add to padding.
         :param tuple (int) zeroPadding: Tuple of ints with padding values."""
        py0, py1, px0, px1 = zeroPadding
        p0, p1, p2, p3 = self._padding
        self._padding = (p0 + py0, p1 + py1, p2 + px0, p3 + px1)
        self._zeroPadding = zeroPadding


class NxConv1D(NxLayer, Conv1D):
    """1D convolution layer for Loihi DNNs."""

    def __init__(self, filters, kernel_size, strides=None, padding=None,
                 numWeightBits=None, synapseEncoding=None, biasExp=None,
                 vThMant=None, weightExponent=None, useSharedSign=None,
                 visualizePartitions=None, validatePartitions=None,
                 logger=None, probeSpikes=None, **kwargs):

        NxLayer.__init__(self, numWeightBits, synapseEncoding, biasExp,
                         vThMant, weightExponent, useSharedSign,
                         visualizePartitions, validatePartitions, logger,
                         probeSpikes, **kwargs)

        # Have to copy Keras default arguments here.
        if strides is None:
            strides = 1
        if padding is None:
            padding = 'valid'

        kwargs = removeNxKwargs(kwargs)

        Conv1D.__init__(self, filters, kernel_size, strides=strides,
                        padding=padding, **kwargs)

        # Padding tuple
        self._padding2D = None

        # To be able to reuse some of the partitioning functions of Conv2D
        # layers, we temporarily add a dummy axis to certain Conv1D attributes.
        # Externally, the Conv1D layer still looks and behaves as 1D.
        self._input_shape3D = None
        self._strides2D = None
        self._kernel_shape2D = None
        self._dilation_rate2D = None

    def get_config(self):
        config = {}
        baseConfig = Conv1D.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def build(self, input_shape):
        Conv1D.build(self, input_shape)

        self._input_shape3D = (int(input_shape[1]), 1, int(input_shape[2]))
        self._kernel_shape2D = (self.kernel_size[0], 1)
        self._strides2D = (self.strides[0], 1)
        self._dilation_rate2D = (self.dilation_rate[0], 1)
        self._padding2D = _getPadding(self._input_shape3D, self.padding,
                                      self._kernel_shape2D, self._strides2D,
                                      self._dilation_rate2D)

    def genKernelIdMap(self):
        output_shape3D = (self.output_shape[1], 1, self.output_shape[2])
        return _genKernelIdMap(self._input_shape3D, output_shape3D,
                               self._padding2D, self._strides2D,
                               self._kernel_shape2D,
                               self._dilation_rate2D)

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Multiplicity map, which has the same shape as the input to
            current layer except that the channel axis is removed. Each entry
            specifies to how many cores that neuron sends its spikes.
        :rtype: np.ndarray
        """

        coreIdMap3D = np.expand_dims(coreIdMap, 1)
        multiplicityMap = _getMultiplicityMapConvlike(
            coreIdMap3D, self._input_shape3D, self._kernel_shape2D,
            self._strides2D, self._padding2D, self._dilation_rate2D)

        # If we partition a normal convolution layer along the channel
        # dimension, each input neuron has to be duplicated for every new core.
        # This is not necessary in a DepthwiseConv layer.
        multiplicityMap *= len(set(coreIdMap[0, :]))

        return multiplicityMap[:, 0]

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 2). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x2 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments)

    def compile(self, partitionCandidate):
        return compileConvlike(self, partitionCandidate)

    def validatePartition(self, partitionCandidate):
        """Validate layer partition.

        :raise AssertionError: If reconstructed kernelIdMap does not equal the
            original map.

        :param Layer partitionCandidate: The partitioned layer.
        """

        if self.validatePartitions:
            validatePartitionConvlike(partitionCandidate, self.kernelIdMap)

    def visualizePartition(self, path, partitionCandidate, coreIdMap,
                           coreOccupancy, **kwargs):
        """Visualize layer partition.

        :param str path: Where to save plots.
        :param Layer partitionCandidate: Partitioned layer.
        :param np.ndarray coreIdMap: Tensor of layer shape. Maps neurons to
            their corresponding core id.
        :param np.ndarray coreOccupancy: The number of compartments per core.
        """

        if self.visualizePartitions:
            isFinal = partitionCandidate.postLayer.postLayer is None
            visualizePartitionConvlike(path, self.name, partitionCandidate,
                                       coreIdMap, coreOccupancy, isFinal,
                                       **kwargs)


class NxDepthwiseConv2D(NxLayer, DepthwiseConv2D):
    """Depthwise convolution layer for Loihi DNNs."""

    def __init__(self, kernel_size, strides=None, padding=None,
                 numWeightBits=None, synapseEncoding=None, biasExp=None,
                 vThMant=None, weightExponent=None, useSharedSign=None,
                 visualizePartitions=None, validatePartitions=None,
                 logger=None, probeSpikes=None, **kwargs):

        NxLayer.__init__(self, numWeightBits, synapseEncoding, biasExp,
                         vThMant, weightExponent, useSharedSign,
                         visualizePartitions, validatePartitions, logger,
                         probeSpikes, **kwargs)

        # Have to copy Keras default arguments here.
        if strides is None:
            strides = 1
        if padding is None:
            padding = 'valid'

        kwargs = removeNxKwargs(kwargs)

        DepthwiseConv2D.__init__(self, kernel_size, strides, padding, **kwargs)

        if self.dilation_rate != (1, 1):
            raise NotImplementedError

        # Padding tuple
        self._padding = kwargs.get('_padding', None)
        self._zeroPadding = kwargs.get('_zeroPadding', None)

    def get_config(self):
        config = {'_padding': self.padding,
                  '_zeroPadding': self._zeroPadding}
        baseConfig = DepthwiseConv2D.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    @property
    def zeroPadding(self):
        """Zero padding of previous layer. Replaces ZeroPadding Layers."""
        return self._zeroPadding

    @zeroPadding.setter
    def zeroPadding(self, zeroPadding):
        """Set zeroPadding and add to padding.
         :param tuple (int) zeroPadding: Tuple of ints with padding values."""
        py0, py1, px0, px1 = zeroPadding
        p0, p1, p2, p3 = self._padding
        self._padding = (p0 + py0, p1 + py1, p2 + px0, p3 + px1)
        self._zeroPadding = zeroPadding

    def build(self, input_shape):
        DepthwiseConv2D.build(self, input_shape)

        self._padding = _getPadding(input_shape.as_list()[1:], self.padding,
                                    self.kernel_size, self.strides,
                                    self.dilation_rate)

    def genKernelIdMap(self):
        return _genKernelIdMap(self.input_shape[1:], self.output_shape[1:],
                               self._padding, self.strides, self.kernel_size,
                               self.dilation_rate, True,
                               zeroPadding=self.zeroPadding)

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Multiplicity map, which has the same shape as the input to
            current layer except that the channel axis is removed. Each entry
            specifies to how many cores that neuron sends its spikes.
        :rtype: np.ndarray
        """

        return _getMultiplicityMapConvlike(coreIdMap, self.input_shape[1:],
                                           self.kernel_size, self.strides,
                                           self._padding, self.dilation_rate,
                                           zeroPadding=self.zeroPadding)

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 3). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x3 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments)

    def compile(self, partitionCandidate):
        return compileConvlike(self, partitionCandidate)

    def validatePartition(self, partitionCandidate):
        """Validate layer partition.

        :raise AssertionError: If reconstructed kernelIdMap does not equal the
            original map.

        :param Layer partitionCandidate: The partitioned layer.
        """

        if self.validatePartitions:
            validatePartitionConvlike(partitionCandidate, self.kernelIdMap)

    def visualizePartition(self, path, partitionCandidate, coreIdMap,
                           coreOccupancy, **kwargs):
        """Visualize layer partition.

        :param str path: Where to save plots.
        :param Layer partitionCandidate: Partitioned layer.
        :param np.ndarray coreIdMap: Tensor of layer shape. Maps neurons to
            their corresponding core id.
        :param np.ndarray coreOccupancy: The number of compartments per core.
        """

        if self.visualizePartitions:
            isFinal = partitionCandidate.postLayer.postLayer is None
            visualizePartitionConvlike(path, self.name, partitionCandidate,
                                       coreIdMap, coreOccupancy, isFinal,
                                       **kwargs)


class NxAveragePooling2D(NxLayer, AveragePooling2D):
    """Average pooling layer for Loihi DNNs."""

    def __init__(self, pool_size=None, strides=None, padding=None,
                 numWeightBits=None, synapseEncoding=None, biasExp=None,
                 vThMant=None, weightExponent=None, useSharedSign=None,
                 visualizePartitions=None, validatePartitions=None,
                 logger=None, probeSpikes=None, **kwargs):

        NxLayer.__init__(self, numWeightBits, synapseEncoding, biasExp,
                         vThMant, weightExponent, useSharedSign,
                         visualizePartitions, validatePartitions, logger,
                         probeSpikes, **kwargs)

        # Have to copy Keras default arguments here.
        if pool_size is None:
            pool_size = (2, 2)
        if padding is None:
            padding = 'valid'

        kwargs = removeNxKwargs(kwargs)

        AveragePooling2D.__init__(self, pool_size, strides, padding, **kwargs)

        # Padding tuple
        self._padding = None

    def get_config(self):
        config = {}
        baseConfig = AveragePooling2D.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def build(self, input_shape):
        input_shape_list = input_shape.as_list()
        self._padding = _getPadding(input_shape_list[1:], self.padding,
                                    self.pool_size, self.strides, (1, 1))

        output_shape = self.compute_output_shape(input_shape_list)

        # Add weights and biases.
        inChannels = input_shape_list[-1]
        outChannels = output_shape[-1]
        weightShape = self.pool_size + (inChannels, outChannels)
        self.add_weight('W', weightShape, initializer='ones', trainable=False)
        self.add_weight('b', (outChannels,), initializer='zeros',
                        trainable=False)

        AveragePooling2D.build(self, input_shape)

    def genKernelIdMap(self):
        return _genKernelIdMap(self.input_shape[1:], self.output_shape[1:],
                               self._padding, self.strides, self.pool_size,
                               (1, 1), True)

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Multiplicity map, which has the same shape as the input to
            current layer except that the channel axis is removed. Each entry
            specifies to how many cores that neuron sends its spikes.
        :rtype: np.ndarray
        """

        return _getMultiplicityMapConvlike(coreIdMap, self.input_shape[1:],
                                           self.pool_size, self.strides,
                                           self._padding, (1, 1))

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 3). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x3 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments)

    def compile(self, partitionCandidate):
        return compileConvlike(self, partitionCandidate)

    def validatePartition(self, partitionCandidate):
        """Validate layer partition.

        :raise AssertionError: If reconstructed kernelIdMap does not equal the
            original map.

        :param Layer partitionCandidate: The partitioned layer.
        """

        if self.validatePartitions:
            validatePartitionConvlike(partitionCandidate, self.kernelIdMap)

    def visualizePartition(self, path, partitionCandidate, coreIdMap,
                           coreOccupancy, **kwargs):
        """Visualize layer partition.

        :param str path: Where to save plots.
        :param Layer partitionCandidate: Partitioned layer.
        :param np.ndarray coreIdMap: Tensor of layer shape. Maps neurons to
            their corresponding core id.
        :param np.ndarray coreOccupancy: The number of compartments per core.
        """

        if self.visualizePartitions:
            isFinal = partitionCandidate.postLayer.postLayer is None
            visualizePartitionConvlike(path, self.name, partitionCandidate,
                                       coreIdMap, coreOccupancy, isFinal,
                                       **kwargs)


class NxDense(NxLayer, Dense):
    """Fully-connected layer for Loihi DNNs."""

    def __init__(self, units, numWeightBits=None, synapseEncoding=None,
                 biasExp=None, vThMant=None, weightExponent=None,
                 useSharedSign=None, visualizePartitions=None,
                 validatePartitions=None, logger=None, probeSpikes=None,
                 **kwargs):

        NxLayer.__init__(self, numWeightBits, synapseEncoding, biasExp,
                         vThMant, weightExponent, useSharedSign,
                         visualizePartitions, validatePartitions, logger,
                         probeSpikes, **kwargs)

        kwargs = removeNxKwargs(kwargs)

        Dense.__init__(self, units, **kwargs)

    def get_config(self):
        config = {}
        baseConfig = Dense.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 1). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x1 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments)

    def compile(self, partitionCandidate):
        """Compile partition candidate.

        Tries to create compartment, synapse and axon groups for each partition
        in the layer.

        Returns ``None`` if the partition candidate exceed some resource limit.

        :param Layer partitionCandidate: The layer partition candidate.

        :return: ``partitionCandidate``.
        :rtype: Layer | None
        """

        # Input variables #
        ###################

        inputSize = self.input_shape[1]

        limits = self.exclusionCriteria

        connKwargs = partitionCandidate.connectionKwargs

        # Get number of cores used by network so far. Needed for Partition
        # object.
        coreCount = self.getTotalCoreCount(partitionCandidate)

        # Flatten multiplicity map of lower layer and repeat for each channel
        # in feature map. Needed for input axons.
        preMultiplicityFlat = np.ravel(partitionCandidate.multiplicityMap, 'F')

        # Flatten multiplicity map of current layer and repeat for each channel
        # in feature map. Needed for output axons.
        multiplicityFlat = \
            np.ravel(partitionCandidate.postLayer.multiplicityMap, 'F')

        # Todo : enable population mode in subtractive-reset. Currently only
        # support for discrete axons.
        if self.resetMode == 'soft':
            preMultiplicityFlat = 2 * np.ones_like(preMultiplicityFlat)
            multiplicityFlat = 2 * np.ones_like(multiplicityFlat)

        # Get weights and biases. Needed for SynEntries and CompartmentGroup.
        weights, biases = self.get_weights()

        if len(self._inbound_nodes):
            prevLayer = self._inbound_nodes[0].inbound_layers
            # prevLayer may be wrapped in a list of lenght 1:
            if isinstance(prevLayer, (list, tuple)):
                prevLayer = prevLayer[0]
            if 'Flatten' in prevLayer.__class__.__name__:
                shape = prevLayer.input_shape[1:]
                idxs = np.arange(int(np.prod(shape)))
                permutation = np.ravel(np.reshape(idxs, shape, 'C'), 'F')
                weights = weights[permutation]

        biases = biases.astype(int)
        biasExp = np.ones_like(biases) * \
            partitionCandidate.compartmentKwargs['biasExp']

        neuronSize = 2 if self.resetMode == 'soft' else 1
        # somaOffset = 1 if self.resetMode == 'soft' else 0

        # Iterate over cores #
        ######################

        for relCoreId in range(partitionCandidate.numCores):

            # Get subsection of layer corresponding to current core.
            mask = partitionCandidate.coreIdMap == relCoreId
            maskFlat = np.ravel(mask, 'F')
            cxIds = np.nonzero(maskFlat)[0]
            numNeurons = len(cxIds)
            relCxIds = np.arange(numNeurons)

            if len(relCxIds) * neuronSize > limits.maxNumCompartments:
                limits.maxNumCompartments += 1
                return

            # Compute number of synaptic resources
            synapseEncoder = SynapseEncoder(connKwargs['numWeightBits'],
                                            limits.maxNumSynPerSynEntry,
                                            'dense1',
                                            connKwargs['useSharedSign'])

            # SynEntries & SynFmts #
            ########################

            synEntriesOfCore = []
            for i in range(inputSize):

                kernelIds = cxIds + 1
                wgts = weights[i, cxIds]
                synapseEncoder.encode(relCxIds * neuronSize, wgts, 0, 0,
                                      kernelIds)

                synEntriesOfSourceGroup = [synapseEncoder.popSynEntries()]

                synEntriesOfCore.append(synEntriesOfSourceGroup)

            # synEntries and synFmts for recurrent connections
            if self.resetMode == 'soft':

                wgt = np.array([-connKwargs['weightMantSR']])
                for relCxId in relCxIds:
                    synEntriesOfRecurrentGroup = []
                    synapseEncoder.encode(np.array([relCxId]) * neuronSize,
                                          wgt,
                                          0,
                                          0,
                                          np.array([1], int),
                                          softReset=True)

                    synEntriesOfRecurrentGroup.append(
                        synapseEncoder.popSynEntries())

                    synEntriesOfCore.append(synEntriesOfRecurrentGroup)

            synFmts, synEntryMap = compressSynFmts(synapseEncoder.getSynFmts(),
                                                   limits.maxNumSynFmt)

            if len(synFmts) > limits.maxNumSynFmt:
                limits.numSynFmts += 1
                return

            synEntriesOfCore = remapSynEntries(synEntriesOfCore, synFmts,
                                               synEntryMap)

            chipCounter = self.incrementChipCounter(coreCount, relCoreId)

            partition = Partition(relCoreId, chipCounter, numNeurons,
                                  partitionCandidate, resetMode=self.resetMode)

            for synFmt in synFmts:
                partition.addSynFmt(synFmt)

            # Synapse Groups #
            ##################

            synMemOfCore = 0
            for i, synEntriesOfSourceGroup in enumerate(synEntriesOfCore):

                # Create new synapse group.
                synapseGroup = SynapseGroup(i, synEntriesOfSourceGroup)
                synMemOfSourceGroup = synapseGroup.numSynMemWords

                # The longest block of registers in synMem may not exceed 256
                # words for one axon. If axon is shared, only need to find the
                # maximum number of words used up by any of the neurons in the
                # population.
                if synapseGroup.maxSynMemLen > limits.maxNumSynMemWordsPerAxon:
                    limits.synMemPerAxon += 1
                    return

                partition.addSynapseGroup(synapseGroup)
                synMemOfCore += synMemOfSourceGroup

            if synMemOfCore >= limits.maxNumSynMemWords:
                limits.numSynMemWords += 1
                return

            # Input Axons #
            ###############
            recurrentSize = len(relCxIds) if self.resetMode == 'soft' else 0
            inputAxonMultiplicity = np.concatenate(
                [preMultiplicityFlat, np.ones(recurrentSize) * 2])
            for axonId, synapseGroup in enumerate(partition.synapseGroups):
                srcNodeIds = np.array([axonId % inputSize], int)
                inputAxonGroup = InputAxonGroup(
                    srcNodeIds, inputAxonMultiplicity[axonId: axonId + 1],
                    synapseGroup, 0, partition)

                partition.addInputAxonGroup(inputAxonGroup)

                # Build srcIdMap for input layer.
                if axonId < inputSize:
                    partitionCandidate.updateSrcIdMap(axonId,
                                                      (inputAxonGroup, 0))

                if partition.numInputAxons > limits.maxNumAxons:
                    limits.numInputAxons += 1
                    return

            # Output Axons #
            ################

            # Skip for output layer (has no output axons).
            if len(partitionCandidate.postLayer.srcIdMap):

                for postPartition in partitionCandidate.postLayer.partitions:

                    for relCxId, inputAxonGroup in \
                            zip(relCxIds, postPartition.inputAxonGroups):
                        cxId = cxIds[relCxId]
                        outputId = relCxId
                        if self.resetMode == 'soft':
                            outputId = relCxId * neuronSize + 1
                        outputAxonGroup = OutputAxonGroup(
                            np.array([outputId], int),
                            multiplicityFlat[cxId: cxId + 1],
                            np.array([0], int), inputAxonGroup, partition)

                        partition.addOutputAxonGroup(outputAxonGroup)

                if partition.numOutputAxonCfgEntries > limits.maxNumAxons:
                    limits.numOutputAxons += 1
                    return

            # Output Axon for Recurrent Connections #
            #########################################
            if self.resetMode == 'soft':
                for relCxId, inputAxonGroup in \
                        zip(relCxIds, partition.inputAxonGroups[inputSize:]):
                    outputAxonGroup = OutputAxonGroup(
                            np.array([relCxId * neuronSize + 1], int),
                            np.array([2], int),
                            np.array([0], int), inputAxonGroup, partition)

                    partition.addOutputAxonGroup(outputAxonGroup)

                if partition.numOutputAxonCfgEntries > limits.maxNumAxons:
                    limits.numOutputAxons += 1
                    return

            # Compartment Group #
            #####################
            if self.resetMode == 'soft':
                numCx = len(relCxIds) * 2
                # Set biases for dendrite and soma compartments
                tempBiases = np.zeros(numCx, int)
                tempBiasExp = np.zeros(numCx, int)
                tempRelCxIds = np.zeros(numCx, int)
                tempCxIds = np.zeros(numCx, int)

                for relCxId, cxId in enumerate(cxIds):
                    # Dendrite
                    tempBiases[relCxId * 2] = biases[cxId]
                    tempBiasExp[relCxId * 2] = biasExp[cxId]
                    tempRelCxIds[relCxId * 2] = relCxId * 2
                    tempCxIds[relCxId * 2] = cxId * 2

                    # soma
                    tempBiases[relCxId * 2 + 1] = 0
                    tempBiasExp[relCxId * 2 + 1] = 0
                    tempRelCxIds[relCxId * 2 + 1] = relCxId * 2 + 1
                    tempCxIds[relCxId * 2 + 1] = cxId * 2 + 1

                relCxIds = tempRelCxIds
                cxIds = tempCxIds

                # Add dummy neurons
                outputSize = self.output_shape[1]
                maxCxId = len(relCxIds)
                mod = maxCxId % 4
                relCxIds = np.concatenate(
                    [relCxIds,
                     [maxCxId + i for i in range(mod)]]).astype(int)
                newCxIds = [outputSize * neuronSize
                            + self._dummyCxSize + i for i in range(mod)]
                cxIds = np.concatenate([cxIds,
                                        newCxIds]).astype(int)
                biasesOfCore = np.concatenate(
                    [tempBiases, np.zeros(mod)]).astype(int)
                biasExpOfCore = np.concatenate(
                    [tempBiasExp, np.zeros(mod)]).astype(int)
                self._dummyCxSize += mod
            else:
                biasesOfCore = biases[cxIds]
                biasExpOfCore = biasExp[cxIds]

            cxGroup = CompartmentGroup(
                relCxIds, biasesOfCore, biasExpOfCore, cxIds)
            partition.addCompartmentGroup(cxGroup)

            partitionCandidate.addPartition(partition)

        return partitionCandidate

    def getMultiplicityMap(self, coreIdMap):

        # In a fully-connected layer, the multiplicity equals the number of
        # cores that the layer is distributed across.
        return np.ones(self.input_shape[1:], int) * (np.max(coreIdMap) + 1)


class NxInputLayer(NxLayer, InputLayer):
    """Input layer for Loihi DNNs."""

    def __init__(self, input_shape=None, batch_size=None, biasExp=None,
                 vThMant=None, visualizePartitions=None, logger=None,
                 signed=False, **kwargs):

        NxLayer.__init__(self, biasExp=biasExp, vThMant=vThMant,
                         visualizePartitions=visualizePartitions,
                         logger=logger, **kwargs)

        # Set param for signed input spikes.
        self._signed = signed

        kwargs = removeNxKwargs(kwargs)

        InputLayer.__init__(self, input_shape, batch_size, **kwargs)

    def get_config(self):
        config = {'signed': self._signed}
        baseConfig = InputLayer.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, ndim_input). The first axis represents
            possible partition configurations of the layer. Each candidate
            consists of a 2 x ndim matrix: [numCoresPerAxis, coreShape]. The
            keys of the dictionary are the number of cores associated with a
            particular array of candidates.
        :rtype: dict
        """
        outputShape = fix_input_layer_shape(self.output_shape)[1:]

        # When using signed spikes the number of channels in the output
        # is doubled.
        if self.signed:
            outputShape = outputShape[:-1] + (2 * outputShape[-1],)
        return getPartitionCandidates(outputShape,
                                      self.maxNumCompartments)

    def compile(self, partitionCandidate):
        """Compile partition candidate.

        Tries to create compartment group and axon groups for each partition
        in the layer.

        Returns ``None`` if the partition candidate exceed some resource limit.

        :param Layer partitionCandidate: The proposed layer partition.
        :return: partitionCandidate.
        :rtype: Layer
        """

        # Compute shape for signed input.
        outputShape = fix_input_layer_shape(self.output_shape)[1:]

        # When using signed spikes the number of channels in the output
        # is doubled.
        if self.signed:
            outputShape = outputShape[:-1] + (2 * outputShape[-1],)

        # Get number of cores used by network so far. Needed for Partition
        # object.
        coreCount = self.getTotalCoreCount(partitionCandidate)

        # Flatten multiplicity map of current layer and repeat for each channel
        # in feature map. Needed for output axons.
        multiplicityFlat = np.tile(np.ravel(
            partitionCandidate.postLayer.multiplicityMap, 'F'),
            outputShape[-1])

        # Todo : Enable population mode for subtractive reset.
        if self.resetMode == 'soft':
            multiplicityFlat = 2 * np.ones_like(multiplicityFlat)

        # Iterate over cores #
        ######################
        connKwargs = partitionCandidate.connectionKwargs
        limits = self.exclusionCriteria

        # Get neuronSize to check for additional compartments in 'soft' mode.
        neuronSize = 2 if self.resetMode == 'soft' else 1

        for relCoreId in range(partitionCandidate.numCores):

            # Get subsection of layer corresponding to current core.
            mask = partitionCandidate.coreIdMap == relCoreId
            maskFlat = np.ravel(mask, 'F')
            relToAbsDestCxIdxMap = np.nonzero(maskFlat)[0]

            # Check core numCx
            numCx = len(relToAbsDestCxIdxMap) * neuronSize
            if numCx > limits.maxNumCompartments:
                limits.coreSizeInterleaved += 1
                return

            # Initialize Partition #
            ########################
            chipCounter = self.incrementChipCounter(coreCount, relCoreId)

            partition = Partition(relCoreId, chipCounter, -1,
                                  partitionCandidate,
                                  resetMode=self.resetMode)

            # Output Axons #
            ################

            srcIdMap = partitionCandidate.postLayer.srcIdMap
            inverseMap = {}
            # Iterate over compartments of current core, which correspond to
            # the srcCxIds of the next postLayer.
            for relCxId, cxId in enumerate(relToAbsDestCxIdxMap):
                # If the convolution does not cover the entire range of
                # srcIds, skip that cxId (has no connections).
                # This occurs for instance with input shape (28, 28),
                # kernel shape (5, 5), strides (2, 2), and padding 'valid'.
                if cxId not in srcIdMap:
                    continue
                for inputAxonGroup, relSrcIdPost in srcIdMap[cxId]:
                    if inputAxonGroup not in inverseMap.keys():
                        inverseMap[inputAxonGroup] = [[], []]
                    inverseMap[inputAxonGroup][0].append(relCxId)
                    inverseMap[inputAxonGroup][1].append(relSrcIdPost)

            # Build outputAxons based on srcIdMap of next higher layer.
            multiplicityOfCore = multiplicityFlat[relToAbsDestCxIdxMap]
            for inputAxonGroup, (cxIds, relSrcIds) in inverseMap.items():
                # Shift cxIds to soma compartment for multi-compartment neurons
                somaRelCxIds = np.array(cxIds)
                if self.resetMode == 'soft':
                    somaRelCxIds *= 2
                    somaRelCxIds += 1

                partition.addOutputAxonGroup(OutputAxonGroup(
                    np.array(somaRelCxIds), multiplicityOfCore[cxIds],
                    np.array(relSrcIds), inputAxonGroup, partition))

            if partition.numOutputAxonCfgEntries > limits.maxNumAxons:
                limits.numOutputAxons += 1
                return

            # synEntries and synFmts for recurrent connections
            if self.resetMode == 'soft':
                if not len(srcIdMap):
                    numCx = len(relToAbsDestCxIdxMap)
                    inverseMap = {relCxId: ([relCxId], [0])
                                  for relCxId in range(numCx)}

                # Compute number of synaptic resources
                synapseEncoder = SynapseEncoder(connKwargs['numWeightBits'],
                                                limits.maxNumSynPerSynEntry,
                                                connKwargs['synapseEncoding'],
                                                connKwargs['useSharedSign'])

                synEntriesOfCore = []
                mappedCxIds = {}
                synGrpId = 0

                # Check inverseMap to ensure all compartments receive at least
                # one inhibitory spike.
                cxIdMap = {}
                for _, (cxIds, relSrcIds) in inverseMap.items():
                    cxIdHash = hash(tuple(cxIds))
                    if cxIdHash in mappedCxIds:
                        continue
                    mappedCxIds[cxIdHash] = (synGrpId, cxIds, relSrcIds)
                    synGrpId += 1
                    synEntriesOfRecurrentGroup = []
                    for cxId, relSrcId in zip(cxIds, relSrcIds):
                        wgt = -connKwargs['weightMantSR']
                        if cxId in cxIdMap:
                            wgt = 0
                        cxIdMap[cxId] = 1
                        # wgt is set by mapper to match vTh.
                        synapseEncoder.encode(np.array([cxId], int) * 2,
                                              np.array([wgt], int),
                                              0,
                                              0,
                                              np.array([1], int),
                                              softReset=True)

                        synEntriesOfRecurrentGroup.append(
                            synapseEncoder.popSynEntries())

                    synEntriesOfCore.append(synEntriesOfRecurrentGroup)

                # For some configurations a neuron may receive input but is not
                # connected to the next layer. A recurrent synapse is added to
                # ensure the v_mem does not overflow.
                synEntriesOfRecurrentGroup = []
                numCx = len(relToAbsDestCxIdxMap)
                for cxId in range(numCx):
                    if cxId not in cxIdMap:
                        wgt = -connKwargs['weightMantSR']
                        synapseEncoder.encode(np.array([cxId], int) * 2,
                                              np.array([wgt], int),
                                              0,
                                              0,
                                              np.array([1], int),
                                              softReset=True)

                        synEntriesOfRecurrentGroup.append(
                            synapseEncoder.popSynEntries())

                if len(synEntriesOfRecurrentGroup):
                    synEntriesOfCore.append(synEntriesOfRecurrentGroup)

                synFmts, synEntryMap = compressSynFmts(
                    synapseEncoder.getSynFmts(), limits.maxNumSynFmt)

                if len(synFmts) > limits.maxNumSynFmt:
                    limits.numSynFmts += 1
                    return

                synEntriesOfCore = remapSynEntries(synEntriesOfCore, synFmts,
                                                   synEntryMap)

                for synFmt in synFmts:
                    partition.addSynFmt(synFmt)

                # Synapse Groups #
                ##################

                synMemOfCore = 0
                for i, synEntriesOfSourceGroup in enumerate(synEntriesOfCore):

                    # Create new synapse group.
                    synapseGroup = SynapseGroup(i, synEntriesOfSourceGroup)
                    # The longest block of registers in synMem may not exceed
                    # 256 words for one axon. If axon is shared, only need to
                    # find the maximum number of words used up by any of the
                    # neurons in the population.
                    if (synapseGroup.maxSynMemLen >
                            limits.maxNumSynMemWordsPerAxon):
                        limits.synMemPerAxon += 1
                        return

                    partition.addSynapseGroup(synapseGroup)
                    synMemOfCore += synapseGroup.numSynMemWords

                if synMemOfCore >= limits.maxNumSynMemWords:
                    limits.numSynMemWords += 1
                    return

                # Input & output axons for recurrent group #
                ############################################
                if self.resetMode == 'soft':
                    for (synGrpId, cxIds, relSrcIds) in mappedCxIds.values():
                        synapseGroup = partition.synapseGroups[synGrpId]
                        srcIds = relToAbsDestCxIdxMap[cxIds]
                        inputAxonGroup = InputAxonGroup(
                            srcIds, multiplicityFlat[srcIds], synapseGroup,
                            0, partition)

                        partition.addInputAxonGroup(inputAxonGroup)

                        # Shift cxIds to soma compartment for
                        # multi-compartment neurons
                        somaCxIds = np.array(cxIds) * 2
                        somaCxIds += 1

                        partition.addOutputAxonGroup(OutputAxonGroup(
                            somaCxIds, multiplicityFlat[srcIds],
                            np.array(relSrcIds), inputAxonGroup, partition))

                if partition.numOutputAxonCfgEntries > limits.maxNumAxons:
                    limits.numOutputAxons += 1
                    return

                if partition.numInputAxons > limits.maxNumAxons:
                    limits.numInputAxons += 1
                    return

            # Compartment Group #
            #####################
            numCx = len(relToAbsDestCxIdxMap)
            if self.resetMode == 'soft':
                tempCxIdxMap = np.zeros(numCx * 2, int)
                relCxIds = np.arange(numCx) * 2
                tempCxIdxMap[relCxIds] = relToAbsDestCxIdxMap * 2
                tempCxIdxMap[relCxIds + 1] = relToAbsDestCxIdxMap * 2 + 1
                relToAbsDestCxIdxMap = tempCxIdxMap
                numCx *= 2
                cxIds = np.arange(numCx)

                # Add dummy neurons
                outputSize = np.prod(outputShape).astype(int)
                mod = numCx % 4
                cxIds = np.concatenate([cxIds, [numCx + i for i
                                                in range(mod)]]).astype(int)

                relCxIds = [outputSize * 2 + self._dummyCxSize + i
                            for i in range(mod)]
                relToAbsDestCxIdxMap = np.concatenate([relToAbsDestCxIdxMap,
                                                       relCxIds]).astype(int)

                biasMant = np.zeros(numCx + mod, int)
                biasExp = np.ones_like(biasMant) * \
                    partitionCandidate.compartmentKwargs['biasExp']
                self._dummyCxSize += mod
            else:
                cxIds = np.arange(numCx)

                # BiasMant will be set later.
                biasMant = np.zeros(numCx, int)
                biasExp = np.ones_like(biasMant) * \
                    partitionCandidate.compartmentKwargs['biasExp']

            cxGroup = CompartmentGroup(cxIds, biasMant, biasExp,
                                       relToAbsDestCxIdxMap)

            partition.addCompartmentGroup(cxGroup)

            partitionCandidate.addPartition(partition)

        return partitionCandidate

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        Not needed in input layer; returns empty array.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Multiplicity map, which has the same shape as the input to
            current layer except that the channel axis is removed. Each entry
            specifies to how many cores that neuron sends its spikes.
        :rtype: np.ndarray
        """

        return np.array([])

    def visualizePartition(self, path, partitionCandidate, coreIdMap,
                           coreOccupancy, **kwargs):
        """Visualize layer partition.

        :param str path: Where to save plots.
        :param Layer partitionCandidate: Partitioned layer.
        :param np.ndarray coreIdMap: Tensor of layer shape. Maps neurons to
            their corresponding core id.
        :param np.ndarray coreOccupancy: The number of compartments per core.
        """

        if self.visualizePartitions:
            path = os.path.abspath(os.path.join(path, self.name))
            if not os.path.exists(path):
                os.makedirs(path)

            plot_coreIdMap(coreIdMap, path, self.name)

            plot_core_occupancy(coreOccupancy, path, self.name)

            if len(self._outbound_nodes) > 0:
                plot_layer_partition(partitionCandidate, path)

    @property
    def signed(self):
        """Return True if signed spikes are used in input.
        """
        return self._signed


class NxFlatten(NxLayer, Flatten):
    """Flatten layer for Loihi DNNs."""

    def __init__(self, data_format=None, **kwargs):

        NxLayer.__init__(self, **kwargs)

        kwargs = removeNxKwargs(kwargs)

        Flatten.__init__(self, data_format, **kwargs)

    def get_config(self):
        config = {}
        baseConfig = Flatten.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def compile(self, partitionCandidate):
        return partitionCandidate.postLayer

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Not needed in flatten layer; returns empty array.
        :rtype: np.ndarray
        """

        return np.array([])

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        The flatten layer is not actually ported to Loihi, so the partition
        candidates proposed here will never be used. The flatten compile
        method just propagates the post layer partition through.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 1). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x1 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments)


class NxZeroPadding2D(NxLayer, ZeroPadding2D):
    """Padding layer for Loihi DNNs."""

    def __init__(self, padding=None, data_format=None, **kwargs):

        NxLayer.__init__(self, **kwargs)

        kwargs = removeNxKwargs(kwargs)

        if padding is None:
            padding = (1, 1)

        ZeroPadding2D.__init__(self, padding, data_format=data_format,
                               **kwargs)

    def get_config(self):
        config = {}
        baseConfig = ZeroPadding2D.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def compile(self, partitionCandidate):
        return partitionCandidate.postLayer

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Not needed in zero-padding layer; returns empty array.
        :rtype: np.ndarray
        """

        return np.array([])

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        The padding layer is not actually ported to Loihi, so the partition
        candidates proposed here will never be used. The compile method just
        propagates the post layer partition through.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 1). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x1 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments)


class NxReshape(NxLayer, Reshape):
    """Reshape layer for Loihi DNNs."""

    def __init__(self, target_shape, **kwargs):

        NxLayer.__init__(self, **kwargs)

        kwargs = removeNxKwargs(kwargs)

        Reshape.__init__(self, target_shape, **kwargs)

    def get_config(self):
        config = {}
        baseConfig = Reshape.get_config(self)
        baseConfig2 = NxLayer.get_config(self)
        config.update(baseConfig)
        config.update(baseConfig2)
        return config

    def compile(self, partitionCandidate):
        return partitionCandidate.postLayer

    def getMultiplicityMap(self, coreIdMap):
        """Generate multiplicity map.

        :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each
            entry specifies the core id to which that neuron belongs.

        :return: Not needed in reshape layer; returns empty array.
        :rtype: np.ndarray
        """

        return np.array([])

    def getPartitionCandidates(self):
        """Get possible partition configurations for a layer.

        The reshape layer is not actually ported to Loihi, so the partition
        candidates proposed here will never be used. The compile method just
        propagates the post layer partition through.

        :return: Dictionary of partition candidates. Each value is a 3D array
            of shape (numCandidates, 2, 1). The first axis represents possible
            partition configurations of the layer. Each candidate consists of
            a 2x1 matrix: [numCoresPerAxis, coreShape]. The keys of the
            dictionary are the number of cores associated with a particular
            array of candidates.
        :rtype: dict
        """

        return getPartitionCandidates(self.output_shape[1:],
                                      self.maxNumCompartments)


class NxModel(Model):
    """A DNN model class for Loihi."""

    def __init__(self, *args, **kwargs):

        self.numCandidatesToCompute = kwargs.pop('numCandidatesToCompute', 1)
        self.logdir = kwargs.pop('logdir', None)
        self.logger = kwargs.pop('logger', None)
        self.verbose = kwargs.pop('verbose', False)
        self.saveOutput = kwargs.pop('saveOutput', False)
        self.storeAllCandidates = kwargs.pop('storeAllCandidates', False)

        self.board = None
        self.tmpdir = None
        self.pathMappables = None
        self.pathPartitions = None
        self.partitionOptimizer = None
        self._maxNumCoresPerChip = kwargs.pop('maxNumCoresPerChip', 128)
        assert 128 >= self._maxNumCoresPerChip > 0

        self._isInitialized = False

        super(NxModel, self).__init__(*args, **kwargs)

    def initialize(self):
        """Initialize NxModel paths and attributes.

        It is necessary to do this outside of the NxModel constructor to enable
        loading the model from disk later. The Keras load_model function, which
        is used for loading, does not pass any arguments to the NxModel
        constructor. So after loading we need to call this function to
        overwrite the default settings.
        """

        if self.logdir is None:
            self.tmpdir = os.path.abspath(os.path.join(os.path.dirname(
                os.path.realpath(__file__)),
                '../../..', 'temp', str(time.time())))
            os.makedirs(self.tmpdir, exist_ok=True)
            self.logdir = self.tmpdir

        modelDumpDir = os.path.join(self.logdir, 'model_dumps')
        self.pathMappables = os.path.join(modelDumpDir, 'mappables')
        self.pathPartitions = os.path.join(modelDumpDir, 'partitions')
        for path in [self.pathMappables, self.pathPartitions]:
            if not os.path.exists(path):
                os.makedirs(path)

        if self.logger is None:
            self.logger = get_logger("NET.DNN")

        self.logger.addHandler(logging.FileHandler(
            os.path.join(self.logdir, 'compiler.log')))

        self.partitionOptimizer = PartitionOptimizer(
            self.numCandidatesToCompute, self.logger, logdir=self.logdir,
            storeAllCandidates=self.storeAllCandidates)

        self._isInitialized = True

    def get_config(self):

        # TODO: Find a way to store config params for NxModel class.
        # get_config cannot be used by Model subclasses for saving
        # config params.
        config = {'numCandidatesToCompute': self.numCandidatesToCompute,
                  'logdir': self.logdir,
                  'verbose': self.verbose,
                  'saveOutput': self.saveOutput,
                  'storeAllCandidates': self.storeAllCandidates,
                  '_maxNumCoresPerChip': self._maxNumCoresPerChip}
        baseConfig = super(NxModel, self).get_config()
        config.update(baseConfig)
        return config

    @logMemTime('Partition')
    def partition(self):
        """Find optimal partition and store it on disk."""

        if not self._isInitialized:
            self.initialize()

        self.partitionOptimizer.initialize(self.layers[-1].output_shape[1:])

        coreCount = 0
        for i, layer in enumerate(reversed(self.layers)):
            layer.exclusionCriteria.maxNumCoresPerChip = \
                self._maxNumCoresPerChip

            self.logger.info("Finding best partition for %s.", layer.name)

            # Compute the optimal layer partitioning based on the possible
            # partitionings of the next higher layer.
            self.partitionOptimizer.run(layer)

            # Get mappable only for the case that we want to dump it to disk.
            mappableLayer = self.partitionOptimizer.getOptimalPartition()

            # Some layers are "virtual" in the sense that they have no physical
            # representation on Loihi in terms of compartments etc, e.g. 
            # reshape or padding layers. In these cases, the mappableLayer is
            # just a reference to the postLayer and need not be stored again.
            if mappableLayer.id != layer.name:
                self.logger.info("Skipped (virtual layer).")
                continue

            # Save partitioned layer so we can load it later for mapping.
            self.partitionOptimizer.savePartitionConfig(self.pathPartitions,
                                                        mappableLayer)
            serializeLayer(mappableLayer, self.pathMappables)

            # Print partition statistics.
            if self.verbose:
                layer.exclusionCriteria.print()

            numCores = mappableLayer.numCores
            self.logger.info("Layer %s was distributed across %s core%s.",
                             mappableLayer.id, numCores, getS(numCores))
            coreCount += numCores

        # Clean-up.
        self.partitionOptimizer.clearTemp()

        # If it wasn't private, could replace 128 by
        # self.layers[-1].exclusionCriteria._maxNumCoresPerChip
        chipCount = int(np.ceil(coreCount / 128))
        self.logger.info("Total: %s core%s (%s chip%s).", coreCount,
                         getS(coreCount), chipCount, getS(chipCount))

        # Save output plots.
        if self.saveOutput:
            self.partitionOptimizer.saveOptimalPartitionCostTerms(self.logdir)
            self.partitionOptimizer.saveCandidateCosts(self.logdir)
            self.saveExclusionCriteriaHitCount(self.logdir)

            plot_core_utilization(self.partitionOptimizer.getLayers(),
                                  self.logdir)
            plot_exclusion_criteria_hit_count(self.logdir)
            visualize_partitions(self.logdir)
            plot_cost_graph(self.logdir)
            plot_cost_terms(self.logdir)
            plot_cx_syn(self.logdir, self)

    def compileModel(self, board=None):
        """Compile the ``NxModel``.

        Attemps to load intermediate representations from disk. Continues
        pipeline at the latest possible stage.

        :param N2Board | None board: Board where the model is mapped to.
        """

        if not self._isInitialized:
            self.initialize()

        self.board = N2Board(0, numCoresPerChip=self._maxNumCoresPerChip) \
            if board is None else board

        hasPartitionConfig, hasMappable = self.canLoad()

        # Worst case: Nothing has been stored; start optimization from scratch.
        if not hasPartitionConfig and not hasMappable:
            self.partition()

        # If possible, skip optimization step by loading a previously found
        # partition configuration.
        elif not hasMappable:
            self.partitionFromSavedConfig()

        # The steps above ensure that now there is a mappable stored on disk.
        mapper = self.map()

        # Clean-up.
        self.clearTemp()

        return mapper

    @logMemTime('Run')
    def run(self, numSteps):
        """Run model for given number of steps.

        :param int numSteps: Number of time steps to run model.
        """

        assert self.board is not None, \
            "Model must be compiled before running."

        self.board.run(numSteps)

    def disconnect(self):
        """Disconnect from board."""

        assert self.board is not None, \
            "Model must be compiled before disconnecting."

        self.board.disconnect()

        self.clearTemp()

    def clearTemp(self):
        """Remove temporary files, if any."""

        if self.tmpdir is not None and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

        for layer in self.layers:
            layer.deleteKernelIdMap()

    def canLoad(self):
        """Return flags that indicate which intermediate results are available.

        :return: Tuple of boolean flags.
        :rtype: tuple[bool]
        """

        hasMappable = (os.path.exists(self.pathMappables) and
                       len(os.listdir(self.pathMappables)))
        hasPartitionConfig = (os.path.exists(self.pathPartitions) and
                              len(os.listdir(self.pathPartitions)))
        return hasPartitionConfig, hasMappable

    @logMemTime('Map')
    def map(self):
        """Load partitioned and compiled layers and map them to an nx board."""

        self.logger.info("Loading and mapping pre-compiled layers from %s.",
                         self.pathMappables)

        mapper = DnnMapper(self.board)

        for compilableLayer in reversed(self.layers):
            name = compilableLayer.name

            try:
                mappableLayer = deserializeLayer(self.pathMappables,
                                                 name + '.pickle')
                # Ensure that mappables are updated with latest changes, e.g.
                # threshold or weight exponents.
                mappableLayer.compartmentKwargs.update(
                    compilableLayer.compartmentKwargs)
                mappableLayer.connectionKwargs.update(
                    compilableLayer.connectionKwargs)
            except FileNotFoundError:
                continue

            self.logger.info("Mapping layer %s.", name)
            mapper.map(mappableLayer)

            # This enables accessing neuron fields directly, e.g. for probes.
            compilableLayer.setBoardAndCxResourceMap(
                self.board, mappableLayer.genCxResourceMap())

        return mapper

    @logMemTime('partitionFromSavedConfig')
    def partitionFromSavedConfig(self):
        """Compile partitioned layers from stored partition config.

        The stored candidates essentially consist of the coreIdMap which
        assigns a core to each neuron in the layer.

        This function calls the compile method on each of the Layer objects.
        """

        mappablePostLayer = getDummyLayer(self.layers[-1].output_shape[1:])

        for compilableLayer in reversed(self.layers):

            filepath = os.path.join(self.pathPartitions,
                                    compilableLayer.name + '.npz')

            if not os.path.exists(filepath):
                continue

            self.logger.info("Compiling stored partition candidate %s.",
                             compilableLayer.name)
            partitionConfig = np.load(filepath)
            coreIdMap = partitionConfig['coreIdMap']
            multiplicityMap = partitionConfig['multiplicityMap']

            mappableLayer = Layer(compilableLayer.name,
                                  compilableLayer.__class__.__name__,
                                  compilableLayer.compartmentKwargs,
                                  compilableLayer.connectionKwargs, coreIdMap,
                                  multiplicityMap, mappablePostLayer)

            # Populate mappableLayer with synEntries, axons etc.
            mappableLayer = compilableLayer.compile(mappableLayer)

            # Costly kernelIdMap is not used any more and can be removed.
            compilableLayer.deleteKernelIdMap()

            # Save layer to disk.
            serializeLayer(mappableLayer, self.pathMappables)

            # Update postLayer for next iteration.
            mappablePostLayer = mappableLayer

    def saveExclusionCriteriaHitCount(self, path):
        """Save hit count of exclusion criteria.

        Saves the number of times that a partition candicate was rejected
        because it hit a certain exclusion criterion.

        :param str path: Where to save partition.
        """

        self.logger.info("Saving hit count of partition exclusion criteria to "
                         "%s.", path)

        out = {}
        for layer in self.layers:
            if not hasattr(layer, 'exclusionCriteria'):
                continue
            for key, value in layer.exclusionCriteria.asdict().items():
                if key not in out:
                    out[key] = []
                out[key].append(value)

        np.savez_compressed(os.path.join(path, 'exclusion_criteria_hit_count'),
                            **out)


def loadNxModel(filepath, customObjects=None, doCompile=None, **kwargs):
    """Load NxModel from disk.

    :param str filepath: One of the following:
        - Path to the saved model, or
        - h5py.File or h5py.Group object from which to load the model.
    :param dict customObjects: Optional dictionary mapping names (strings) to
        custom classes or functions to be considered during deserialization.
    :param bool doCompile: whether to doCompile the model after loading.

    :return: NxModel.
    :rtype: NxModel
    """

    if customObjects is None:
        customObjects = {}

    if doCompile is None:
        doCompile = True

    nxCustomObjects = {
        'NxModel': NxModel, 'NxDense': NxDense, 'NxFlatten': NxFlatten,
        'NxZeroPadding2D': NxZeroPadding2D, 'NxInputLayer': NxInputLayer,
        'NxAveragePooling2D': NxAveragePooling2D, 'NxReshape': NxReshape,
        'NxDepthwiseConv2D': NxDepthwiseConv2D, 'NxConv2D': NxConv2D,
        'NxConv1D': NxConv1D}

    nxCustomObjects.update(customObjects)

    model = load_model(filepath, nxCustomObjects, doCompile)

    # Overwrite default attributes set by Keras load_model function.
    for attr in ['verbose', 'numCandidatesToCompute', 'logdir', 'logger',
                 'saveOutput']:
        if attr in kwargs:
            setattr(model, attr, kwargs[attr])

    if 'storeAllCandidates' in kwargs:
        model.storeAllCandidates = kwargs['storeAllCandidates']

    model.initialize()

    return model
