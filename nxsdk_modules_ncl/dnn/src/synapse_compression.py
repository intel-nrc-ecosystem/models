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
This module contains functions for encoding synapses in synEntries, and
compressing the corresponding synFmts.
"""

import math
import numpy as np
from scipy.sparse import lil_matrix

from nxsdk_modules_ncl.dnn.src.data_structures import SynFmt, SynEntry, Compression


class SynapseEncoder:
    """Synapse encoder.

    :param int numWeightBits: Number of bit used for weights.
    :param int maxNumSynPerSynEntry: The maximum number of synapses per
        synEntry.
    :param str compression: Compression mode of synapses.
    :param bool useSharedSign: How to deal with inhibitory and excitatory
        connections. If ``True``, negative and positive weights are separated
        and their sign is shared. Otherwise, the sign of each weight is stored.
    :param int | None numDelayBits: Number of bits used for delays.
    """

    def __init__(self, numWeightBits, maxNumSynPerSynEntry,
                 compression, useSharedSign, numDelayBits=None):

        self.numWeightBits = numWeightBits
        assert (numDelayBits in [3, 4, 5, 6]) | (numDelayBits is None)
        self.numDelayBits = numDelayBits
        self._maxNumSynPerSynEntry = maxNumSynPerSynEntry
        self._maxNumSkipBits = 5
        self._synFmts = []
        self._synEntries = []
        self.compression = compression
        self.useSharedSign = useSharedSign

    def popSynEntries(self):
        """Return and remove ``SynEntry`` list for current ``SynapseGroup``.

        :return: synEntries.
        :rtype: list[SynEntry]
        """

        synEntries = self.getSynEntries()

        self.resetSynEntries()

        return synEntries

    def getSynEntries(self):
        """Get list of ``SynEntry`` objects for current ``SynapseGroup``.

        :return: synEntries.
        :rtype: list[SynEntry]
        """

        return self._synEntries

    def getSynFmts(self):
        """Get list of ``SynFmt`` objects for current ``SynapseGroup``.

        :return: synFmts.
        :rtype: list[SynFmt]
        """

        return self._synFmts

    def resetSynEntries(self):
        """Reset synEntries.

        Needed when ``SynapseEncoder`` iterates over ``SynapseGroup`` objects
        of current partition.
        """

        self._synEntries = []

    @staticmethod
    def _getNumIdxBits(val):
        """Compute number of required compartment index bits.

        Maps to legal values: 0, 6, 7, 8, 9, 10.

        :param int val: Largest compartment index of ``SynEntry``.
        :return: Number of bits required by compartment indices.
        :rtype: int
        """

        numIdxBits = int(math.ceil(math.log2(val + 1)))
        assert numIdxBits <= 10, "numIdxBits too large."
        return max(6, numIdxBits)

    def _getNumSkipBits(self, val):
        """Compute number of required skip bits.

        Maps to legal values: 2, 3, 4, 5.

        :param int val: Largest skip index of ``SynEntry``.
        :return: Number of bits required by skip indices.
        :rtype: int
        """

        numSkipBits = int(math.ceil(math.log2(val + 1)))
        assert numSkipBits <= self._maxNumSkipBits, "numSkipBits too large."
        return max(2, numSkipBits)

    def _encodeSynSparse(self, synIds, weights, cIdxOffset, cIdxMult, signMode,
                         kernelIds=None, delays=None, softReset=False):
        """Encode synapses using SPARSE compression.

        :param np.ndarray synIds: Synapse indices.
        :param np.ndarray weights: Weight values.
        :param int cIdxOffset: cIdxOffset.
        :param int cIdxMult: cIdxMult.
        :param int signMode: Defines how inhibitory and excitatory weights are
            treated. 1: Signs are not shared. 2: Excitatory connections with
            shared sign. 3: Inhibitory connections with shared sign.
        :param np.ndarray kernelIds: Global indices of synapse weights. Only
            needed for plotting and reconstructing kernelIdMap to validate
            partition.
        :param np.ndarray delays: Delay values
        :param bool softReset: True if synapse is used in soft-reset mode.

        :return: List of synaptic entries.
        :rtype: list[SynEntry]
        """

        kIds = None
        dlys = None
        numSyn = len(synIds)
        numSynEntries = int(math.ceil(numSyn / self._maxNumSynPerSynEntry))
        for i in range(numSynEntries):
            # Compute id range of up to 60 synapses of current synEntry
            cIdStart = i * self._maxNumSynPerSynEntry
            cIdEnd = min(cIdStart + self._maxNumSynPerSynEntry, numSyn)

            # Get synaptic configuration
            idx = synIds[cIdStart:cIdEnd]
            if kernelIds is not None:
                kIds = kernelIds[cIdStart:cIdEnd]
            wgts = weights[cIdStart:cIdEnd]
            if delays is not None:
                dlys = delays[cIdStart:cIdEnd]
            numIdxBits = self._getNumIdxBits(np.max(idx))

            # Generate synapses and synEntries
            synFmt = SynFmt(len(self._synFmts), cIdxOffset, cIdxMult,
                            numIdxBits, 0, self.numWeightBits,
                            Compression.SPARSE, signMode, self.numDelayBits,
                            softReset)
            self._synFmts.append(synFmt)

            synEntry = SynEntry(0, idx, wgts, synFmt, kIds, dlys)
            self._synEntries.append(synEntry)

    def _encodeSynRunLength(self, synIds, weights, cIdxOffset, cIdxMult,
                            signMode, kernelIds=None, delays=None,
                            softReset=False):
        """Encode synapses using RUNLENGTH compression.

        Creates a new ``SynEntry`` whenever it detects a gap >= 2**5 (limit on
        ``skipIdxBits``).

        .. note:: Assumes that first skip index is zero!

        :param np.ndarray synIds: Synapse indices.
        :param np.ndarray kernelIds: Weight indices.
        :param int cIdxOffset: cIdxOffset.
        :param int cIdxMult: cIdxMult.
        :param int signMode: Defines how inhibitory and excitatory weights are
            treated. 1: Signs are not shared. 2: Excitatory connections with
            shared sign. 3: Inhibitory connections with shared sign.
        :param np.ndarray kernelIds: Global indices of synapse weights. Only
            needed for plotting and reconstructing kernelIdMap to validate
            partition.
        :param np.ndarray delays: Delay values
        :param bool softReset: True if synapse is used in soft-reset mode.

        :return: List of synaptic entries.
        :rtype: list[SynEntry]
        """

        kIds = None
        dlys = None
        numSyn = len(synIds)
        cIdStart = 0
        # Find indices of gaps in the connection vector, in descending order.
        gapIds = list(np.flatnonzero(
            np.diff(synIds) >= 2 ** self._maxNumSkipBits))[::-1]
        while cIdStart < numSyn:
            # Compute id range of up to 60 synapses of current synEntry or
            # until next compartment index increment >= 2**5.
            cIdEnd = min(cIdStart + self._maxNumSynPerSynEntry, numSyn)
            if len(gapIds) and gapIds[-1] < cIdEnd:
                cIdEnd = gapIds.pop() + 1

            # Get synaptic configuration
            idx = synIds[cIdStart:cIdEnd]
            skipIdx = np.insert(np.diff(idx), 0, 0)
            prefixOffset = idx[0]
            if kernelIds is not None:
                kIds = kernelIds[cIdStart:cIdEnd]
            wgts = weights[cIdStart:cIdEnd]
            if delays is not None:
                dlys = delays[cIdStart:cIdEnd]
            numIdxBits = self._getNumIdxBits(prefixOffset)
            numSkipBits = self._getNumSkipBits(np.max(skipIdx))

            # Generate synapses and synEntries
            synFmt = SynFmt(len(self._synFmts), cIdxOffset, cIdxMult,
                            numIdxBits, numSkipBits, self.numWeightBits,
                            Compression.RUNLENGTH, signMode, self.numDelayBits,
                            softReset)
            self._synFmts.append(synFmt)

            synEntry = SynEntry(prefixOffset, skipIdx, wgts, synFmt, kIds,
                                dlys)
            self._synEntries.append(synEntry)

            cIdStart = cIdEnd

    def _encodeSynDense1(self, synIds, weights, cIdxOffset, cIdxMult, signMode,
                         kernelIds=None, delays=None, softReset=False):
        """Encode synapses using DENSE compression.

        This encoder creates a new synaptic entry whenever a compartment
        index increment greater than 1 is detected. That means no dummy
        connections with zero weight are created.

        :param np.ndarray synIds: Synapse indices.
        :param np.ndarray kernelIds: Weight indices.
        :param int cIdxOffset: cIdxOffset.
        :param int cIdxMult: cIdxMult.
        :param int signMode: Defines how inhibitory and excitatory weights are
            treated. 1: Signs are not shared. 2: Excitatory connections with
            shared sign. 3: Inhibitory connections with shared sign.
        :param np.ndarray kernelIds: Global indices of synapse weights. Only
            needed for plotting and reconstructing kernelIdMap to validate
            partition.
        :param np.ndarray delays: Delay values
        :param bool softReset: True if synapse is used in soft-reset mode.

        :return: List of synaptic entries.
        :rtype: list[SynEntry]
        """

        kIds = None
        dlys = None
        numSyn = len(synIds)
        cIdStart = 0
        # Find indices of gaps in the connection vector, in descending order.
        gapIds = list(np.flatnonzero(np.diff(synIds) > 1))[::-1]
        while cIdStart < numSyn:
            # Compute id range of up to 60 synapses of current synEntry or
            # until next non-1 compartment index increment
            cIdEnd = min(cIdStart + self._maxNumSynPerSynEntry, numSyn)
            if len(gapIds) and gapIds[-1] < cIdEnd:
                cIdEnd = gapIds.pop() + 1

            # Get synaptic configuration
            idx = synIds[cIdStart:cIdEnd]
            prefixOffset = idx[0]
            if kernelIds is not None:
                kIds = kernelIds[cIdStart:cIdEnd]
            wgts = weights[cIdStart:cIdEnd]
            if delays is not None:
                dlys = delays[cIdStart:cIdEnd]
            numIdxBits = self._getNumIdxBits(prefixOffset)

            # Generate synapses and synEntries
            synFmt = SynFmt(len(self._synFmts), cIdxOffset, cIdxMult,
                            numIdxBits, 0, self.numWeightBits,
                            Compression.DENSE, signMode, self.numDelayBits,
                            softReset)
            self._synFmts.append(synFmt)

            synEntry = SynEntry(prefixOffset, idx - prefixOffset, wgts, synFmt,
                                kIds, dlys)
            self._synEntries.append(synEntry)

            cIdStart = cIdEnd

    def _encodeSynDense2(self, synIds, weights, cIdxOffset, cIdxMult, signMode,
                         kernelIds=None, delays=None, softReset=False):
        """Encode synapses using DENSE compression.

        This encoder does not create synaptic entries whenever the
        compartment index increments by more than 1. Instead it inserts dummy
        connections with weight 0.

        :param np.ndarray synIds: Synapse indices.
        :param np.ndarray weights: Weight values.
        :param int cIdxOffset: cIdxOffset.
        :param int cIdxMult: cIdxMult.
        :param int signMode: Defines how inhibitory and excitatory weights are
            treated. 1: Signs are not shared. 2: Excitatory connections with
            shared sign. 3: Inhibitory connections with shared sign.
        :param np.ndarray kernelIds: Global indices of synapse weights. Only
            needed for plotting and reconstructing kernelIdMap to validate
            partition.
        :param np.ndarray delays: Delay values
         :param bool softReset: True if synapse is used in soft-reset mode.

        :return: List of synaptic entries.
        :rtype: list[SynEntry]
        """

        cIdsUnprocessed = np.copy(synIds)
        wgtsUnprocessed = np.copy(weights)
        dlysUnprocessed = None if delays is None else np.copy(delays)
        kIdsUnprocessed = None if kernelIds is None else np.copy(kernelIds)
        kIds = None
        dlys = None

        while len(cIdsUnprocessed):
            # Extract compartment index range of at most size maxNumSynPerEntry
            cIdStart = cIdsUnprocessed[0]
            mask = cIdsUnprocessed < (cIdStart + self._maxNumSynPerSynEntry)
            notMask = np.logical_not(mask)
            cIds = cIdsUnprocessed[mask]
            cIdEnd = np.max(cIds) + 1

            # Get synaptic configuration
            idx = np.arange(cIdEnd - cIdStart)
            prefixOffset = cIdStart

            if kernelIds is not None:
                kIds = np.zeros(idx.shape, int)
                kIds[cIds - cIdStart] = kIdsUnprocessed[mask]
                kIdsUnprocessed = kIdsUnprocessed[notMask]

            wgts = np.zeros(idx.shape, int)
            wgts[cIds - cIdStart] = wgtsUnprocessed[mask]

            if delays is not None:
                dlys = np.zeros(idx.shape, int)
                dlys[cIds - cIdStart] = dlysUnprocessed[mask]
                dlysUnprocessed = dlysUnprocessed[notMask]

            numIdxBits = self._getNumIdxBits(prefixOffset)

            # Discard processed synaptic configuration
            cIdsUnprocessed = cIdsUnprocessed[notMask]
            wgtsUnprocessed = wgtsUnprocessed[notMask]

            # Generate synapses and synEntries
            synFmt = SynFmt(len(self._synFmts), cIdxOffset, cIdxMult,
                            numIdxBits, 0, self.numWeightBits,
                            Compression.DENSE, signMode, self.numDelayBits,
                            softReset)
            self._synFmts.append(synFmt)

            synEntry = SynEntry(prefixOffset, idx, wgts, synFmt, kIds, dlys)
            self._synEntries.append(synEntry)

    def encode(self, synIds, weights, cIdxOffset, cIdxMult, kernelIds=None,
               delays=None, softReset=False):
        """Encodes synapses in a given compression and sign mode.

        :param np.ndarray synIds: Synapse indices.
        :param np.ndarray weights: Weight values.
        :param int cIdxOffset: cIdxOffset.
        :param int cIdxMult: cIdxMult.
        :param np.ndarray kernelIds: Global indices of synapse weights. Only
            needed for plotting and reconstructing kernelIdMap to validate
            partition.
        :param np.ndarray delays: Delay values.
         :param bool softReset: True if synapse is used in soft-reset mode.
        """

        # Quite costly at this point. Is already covered outside.
        # assert np.all(weights < 256) and np.all(-256 <= weights), \
        #     "Weights must be within [-256, 256)."
        # assert 0 <= cIdxOffset < 16, \
        #     "cIdxOffset must be within [0, 15]."
        # assert 0 <= cIdxMult < 16, \
        #     "cIdxMult must be within [0, 15]."

        assert isinstance(softReset, bool)
        assert self.compression in {'sparse', 'runlength', 'dense1', 'dense2'}
        if self.compression == 'sparse':
            encodeFct = self._encodeSynSparse
        elif self.compression == 'runlength':
            encodeFct = self._encodeSynRunLength
        elif self.compression == 'dense1':
            encodeFct = self._encodeSynDense1
        elif self.compression == 'dense2':
            encodeFct = self._encodeSynDense2
        else:
            encodeFct = self._getOptimalEncoding()

        if self.useSharedSign:
            idxPos = weights >= 0
            idxNeg = np.logical_not(idxPos)
            kernelIdsPos = None if kernelIds is None else kernelIds[idxPos]
            kernelIdsNeg = None if kernelIds is None else kernelIds[idxNeg]
            encodeFct(synIds[idxPos], weights[idxPos], cIdxOffset, cIdxMult, 2,
                      kernelIdsPos, delays, softReset)
            encodeFct(synIds[idxNeg], weights[idxNeg], cIdxOffset, cIdxMult, 3,
                      kernelIdsNeg, delays, softReset)
        else:
            encodeFct(synIds, weights, cIdxOffset, cIdxMult, 1, kernelIds,
                      delays, softReset)

    def _getOptimalEncoding(self):
        """Find optimal compression mode for encoding a given set of synapses.

        Apply all four encoding functions (for pos/neg weights separately
        if ``useSharedSign`` is ``True``). Repeat ``N`` times (by setting
        self.compression = None again, but remembering the cost of each
        encoding). Then pick the most efficient encoding and fix for the rest
        of the partition.
        """

        # Todo: Implement.
        return self._encodeSynSparse


def reconstructKMapFromPartitions(partitions, shape):
    """Re-generate the original kernelMap from generated axons.

    This is useful to validate the synaptic compression.

    :param list[Partition] partitions: Partitions of layer.
    :param np.ndarray shape: Shape of kernelMap
        (numOutputNeurons, numInputNeurons).
    :return: Reconstructed kernelMap.
    :rtype: lil_matrix
    """

    kMapFull = lil_matrix(shape, dtype=int)

    for partition in partitions:
        kMapInterleaved = np.zeros((partition.sizeInterleaved, shape[1]), int)
        for inputAxonGroup in partition.inputAxonGroups:
            synapseGroup = inputAxonGroup.synGroup
            for srcId, synEntries in zip(inputAxonGroup.srcNodeIds,
                                         synapseGroup.synEntries):
                for synEntry in synEntries:
                    # Get kernelIds and cxIds for non-zero kernelIds (to
                    # avoid that kernelId=0 fields overwrite others)
                    kIds = synEntry.kernelIds
                    cxIds = inputAxonGroup.cxBase + synEntry.getCxIds()
                    cxIds = cxIds[kIds != 0]
                    kIds = kIds[kIds != 0]
                    # Insert non-zero kernelIds
                    kMapInterleaved[cxIds, srcId] = kIds
        kMapCore = kMapInterleaved[partition.compartmentGroup.cxIds]
        kMapFull[partition.compartmentGroup.relToAbsDestCxIdxMap] = kMapCore

    return kMapFull


def compressSynFmts(synFmts, maxNumSynFmt):
    """Compress ``synFmts`` by pruning and merging.

    Finds the unique set of synFmts and merges the remaining synFmts until the
    number of different synFmts is below ``maxNumSynFmts``.

    :param list[SynFmt] synFmts: List of all synFmt objects.
    :param int maxNumSynFmt: Maximum number of synFmts supported by core.
    :returns: tuple(synFmts, synEntryToFmtMap)
        synFmts is a list of merged synFmt objects.
        synEntryToFmtMap is a mapping vector from global synEntry
        index to set of merged synFmts.
    """

    # Extract synFmt properties into array for processing
    synFmtProperties = np.stack([synFmt.asArray() for synFmt in synFmts])

    # Extract unique synFmts properties and mapping from synEntries to synFmts
    synFmtProperties, synEntryToFmtMap = np.unique(synFmtProperties,
                                                   return_inverse=True, axis=0)
    numSynFmts = synFmtProperties.shape[0]

    # Compute pairwise distances between rows of synFmtProperties
    # (cIdx{Offset/Mult} and compression receive high weight because they are
    # not allowed to differ). It's conceivable to merge different compressions
    # but that requires re-encoding of synEntries.
    distances = np.infty * np.ones((numSynFmts, numSynFmts))
    for i in range(numSynFmts):
        for j in range(i):
            distances[i, j] = _computeDistance(synFmtProperties[i],
                                               synFmtProperties[j])

    # Merge closest rows of synFmtProperties iteratively until number of
    # distinct synFmts is small enough.
    while numSynFmts > maxNumSynFmt:

        if np.all(np.isinf(distances)):
            print(numSynFmts)
            break

        # Find index of shortest distance in distances.
        idx = np.unravel_index([np.argmin(distances)], distances.shape)
        idx0, idx1 = sorted([idx[0][0], idx[1][0]])

        # Overwrite the first of the two closest rows with the merged row.
        # The merged row has the same cIdx{Offset/Mult} and compression but
        # uses their max of numIdxBits, numSkipBits and numWgtBits.
        synFmtProperties[idx0, 2:5] = \
            np.max(synFmtProperties[[idx0, idx1], 2:5], axis=0)

        # Delete the second of the two closest rows in synFmtProperties and in
        # distances.
        synFmtProperties = np.delete(synFmtProperties, idx1, axis=0)
        distances = np.delete(np.delete(distances, idx1, axis=0), idx1, axis=1)

        # Recompute distances of all other rows to merged row.
        for j in range(idx0):
            distances[idx0, j] = _computeDistance(synFmtProperties[idx0],
                                                  synFmtProperties[j])
        for j in range(idx0 + 1, distances.shape[1]):
            distances[j, idx0] = _computeDistance(synFmtProperties[j],
                                                  synFmtProperties[idx0])

        # Remap synEntryToFmtMap from redundant row to merged row.
        synEntryToFmtMap[synEntryToFmtMap == idx1] = idx0

        # Reduce all row indices above the redundant row by one.
        synEntryToFmtMap[synEntryToFmtMap > idx1] -= 1

        numSynFmts -= 1

    synFmts = []
    for i in range(numSynFmts):
        args = synFmtProperties[i, :-4]
        # Convert compression arg from int to Enum.
        kwargs = {'compression': Compression(synFmtProperties[i, -4]),
                  'signMode': synFmtProperties[i, -3],
                  'numDlyBits': synFmtProperties[i, -2],
                  'softReset': synFmtProperties[i, -1]}
        synFmts.append(SynFmt(i, *args, **kwargs))

    return synFmts, synEntryToFmtMap


def _computeNumBitDistance(numBits1, numBits2):
    """Define pairwise distance metric between vectors of numBits.

    Assigns a smaller distance to vectors that are long compared to short
    vectors despite similar delta between them. This is favorable because if we
    merge two formats with numBits 6 and 7 the relative increase in bits is
    smaller than if we merge such with numBits 1 and 2.

    :param np.ndarray numBits1: First vector of numBits.
    :param np.ndarray numBits2: Second vector of numBits.
    :return: Distance measure.
    :rtype: float
    """

    numBitsDelta = numBits1 - numBits2
    numBitsSum = numBits1 + numBits2
    return numBitsDelta.dot(numBitsDelta) / numBitsSum.dot(numBitsSum)


def _computeDistance(a, b):
    """Compute distance between two vectors of ``SynFmt`` properties.

    Returns infinity if the two synFmts differ in offset, multiplier,
    compression or signMode. Otherwise returns the distance of the numBit
    properties.

    :param np.ndarray a: First vector of synFmt propoerties.
    :param np.ndarray b: Second vector of synFmt propoerties.
    :return: Distance measure.
    :rtype: float
    """

    if not np.array_equal(a[[0, 1, 5, 6]], b[[0, 1, 5, 6]]):
        return np.infty
    else:
        return _computeNumBitDistance(a[[2, 3, 4]], b[[2, 3, 4]])


def remapSynEntries(synEntriesOfCore, synFmts, synEntryToFmtMap):
    """Remap synEntries after corresponding synFmts have been compressed.

    Remaps the ``synFmtId`` pointers in each ``SynEntry`` to the merged set of
    synFmts according to synEntryToFmtMap.

    :param list[list[list[synEntry]]] synEntriesOfCore: List of synEntires for
        all synGroups per core, for all source nodes per synGroup and for all
        synEntries per source node.
    :param list[SynFmt] synFmts: List of all synFmts (before merging).
    :param np.ndarray synEntryToFmtMap: Mapping vector from global synEntry id
        to merged synFmtIds.

    :return: List of remapped synEntries.
    :rtype: list[list[list[SynEntry]]]
    """

    i = 0
    for synEntriesOfGroup in synEntriesOfCore:
        for synEntriesOfNeuron in synEntriesOfGroup:
            for synEntry in synEntriesOfNeuron:
                synEntry.synFmt = synFmts[synEntryToFmtMap[i]]
                i += 1

    return synEntriesOfCore
