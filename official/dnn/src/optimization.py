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

"""Functionality to find the optimal partitioning of a DNN layer."""

import os
import time
from collections import namedtuple, OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from nxsdk_modules.dnn.src.data_structures import Layer
from nxsdk_modules.dnn.src.utils import getCoreOccupancy, \
    getCoreIdMapFromCoreShape, getS

if TYPE_CHECKING:
    import logging
    from nxsdk_modules.dnn.src.dnn_layers import NxLayer

CostTerms = namedtuple('CostTerms', ['coreCost', 'inputAxonCost',
                                     'outputAxonCost', 'synCost',
                                     'postLayerCost'])


class ExclusionCriteria:
    """Set of criteria that define the limits of Loihi hardware."""

    __slots__ = ['maxNumSynPerSynEntry', 'maxNumCompartments',
                 'maxNumAxons', 'maxNumSynMemWords', 'maxNumSynFmt',
                 'maxNumDestinationGroups', 'maxNumSynMemWordsPerAxon',
                 'maxNumCoresPerChip', 'numDestinationGroups',
                 'coreSizeInterleaved', 'numSynFmts', 'synMemPerAxon',
                 'numSynMemWords', 'numInputAxons', 'numOutputAxons',
                 '_counterAttr']

    def __init__(self):

        self.maxNumSynPerSynEntry = 60
        self.maxNumCompartments = 1024
        self.maxNumAxons = 4096
        self.maxNumSynMemWords = 16384
        # ToDo: Raise maxNumSynFmt to 15 once we have a proper synapse encoder.
        self.maxNumSynFmt = 7
        self.maxNumDestinationGroups = 16  # Due to cxBase bug.
        self.maxNumSynMemWordsPerAxon = 256
        self.maxNumCoresPerChip = 128

        self.numDestinationGroups = 0
        self.coreSizeInterleaved = 0
        self.numSynFmts = 0
        self.synMemPerAxon = 0
        self.numSynMemWords = 0
        self.numInputAxons = 0
        self.numOutputAxons = 0

        self._counterAttr = [a for a in self.__slots__
                             if 'maxNum' not in a and '_' not in a]

    def toList(self):
        """Transform class attributes into list.

        :return: List of exclusion criteria.
        :rtype: list[int]
        """

        return [getattr(self, attr) for attr in self._counterAttr]

    def asdict(self):
        """Transform class attributes into dictionary.

        :return: Dictionary of exclusion criteria. Ordered according to the
            time each criterion is applied.
        :rtype: OrderedDict
        """

        return OrderedDict([(key, getattr(self, key))
                            for key in self._counterAttr])

    def print(self):
        """Print exclusion criteria."""

        print("Excluded the following partition candidates:")
        for attr in self._counterAttr:
            print("\t{}: {}".format(attr, getattr(self, attr)))

    @property
    def numCandidates(self):
        return np.sum(self.toList())


class PartitionOptimizer:
    """Determine optimal partitioning of a DNN layer.

    :param int numCandidatesToCompute: Number of partitions to compare.
    :param logging.Logger | None logger: Logging object.
    :param str logdir: Where to save figures.
    :param bool storeAllCandidates: Whether to keep all partition candidates in
        memory. This flag needs to be set to ``True`` if user wants to call the
        ``saveCanddiateCosts`` method.
    """

    def __init__(self, numCandidatesToCompute, logger, logdir=None,
                 storeAllCandidates=None):

        self.numCandidatesToCompute = numCandidatesToCompute
        self.logger = logger
        self._optimalPartitions = None
        self._allCandidates = []
        self._storeAllCandidates = storeAllCandidates

        if logdir is None:
            logdir = os.path.join(os.path.expanduser('~'),
                                  'dnn_partitioner_plots',
                                  time.strftime('%Y%m%d-%H%M%S'))
        self.logdir = logdir

    @staticmethod
    def savePartitionConfig(path, layer):
        """Save partition configuration of a layer.

        This method saves the coreIdMap and the multiplicityMap, which can be
        used to partition a layer.

        :param str path: Where to save partition.
        :param Layer layer: Partitioned layer to save.
        """

        np.savez_compressed(os.path.join(path, layer.id), **layer.asDict())

    def savePartitionConfigs(self, path):
        """Save partition configurations for all layers of a network.

        :param str path: Where to save layers.
        """

        self.logger.info("Saving model partitions to %s.", path)

        for layer in self.getLayers():
            self.savePartitionConfig(path, layer)

    def getOptimalPartition(self):
        """Get optimal ``Layer`` partition.

        :return: Optimal partition.
        :rtype: Layer
        """

        # Sorted at construction.
        return self._optimalPartitions[0]

    def saveOptimalPartitionCostTerms(self, path):
        """Save cost terms of optimal partition.

        :param str path: Where to save cost terms.
        """

        self.logger.info("Saving partition cost terms to %s.", path)

        cost_terms = {}
        layers = self.getLayers()
        for layer in layers:
            for key, value in getCostTerms(layer)._asdict().items():
                if key not in cost_terms:
                    cost_terms[key] = []
                cost_terms[key].append(value)
        np.savez_compressed(os.path.join(path, 'cost_terms'), **cost_terms)

    def saveCandidateCosts(self, path):
        """Save total cost of all partition candidates of each layer.

        :param str path: Where to save cost.
        """

        if not len(self._allCandidates):
            self.logger.warning(
                "Saving candidate cost failed: Candidates were not kept for "
                "memory reasons. Need to set 'storeAllCandidates=True' before "
                "partitioning.")
            return

        self.logger.info("Saving partition candidate costs to %s.", path)

        allCosts = []
        for candidate in self._allCandidates:
            allCosts.append([layer.cost for layer in
                             self.getLayers(candidate)])

        np.savez_compressed(os.path.join(path, 'candidate_costs'),
                            all_costs=allCosts)

    def getLayers(self, startLayer=None):
        """Helper function to extract all partitioned layers.

        Each layer stores a pointer to its parent layer, which we use to
        reconstruct the network hierarchy.

        :param Layer | None startLayer: If provided, use this layer as starting
            point to traverse network hierarchy. If not provided, choose
            optimally partitioned layer.
        :return: List of partitioned layers.
        :rtype list[Layer]
        """

        layers = []

        # Start with bottom layer.
        postLayer = self.getOptimalPartition() if startLayer is None \
            else startLayer
        while True:
            layers.append(postLayer)
            postLayer = postLayer.postLayer
            if postLayer is None:
                # Remove last layer in this list, which is a dummy layer.
                return layers[:-1]

    def initialize(self, modelOutputShape):
        """Initialize ``PartitionOptimizer`` with a dummy ``Layer`` partition.

        This is necessary because the optimization of layer L requires the
        partitioning of layer L+1, and we iterate over the layers of the
        network starting with the output layer.

        :param np.ndarray modelOutputShape: Shape of output layer.
        """

        self._optimalPartitions = [getDummyLayer(modelOutputShape[:-1])]

    def run(self, layer):
        """Partition layer.

        Computes total resource requirements for inputAxons, synapses,
        compartments and outputAxons given the partitions of the subsequent
        layer.

        Checks different ways of partitioning this layer across cores and
        computes cost of each partitioning.

        :param NxLayer | KerasLayer layer: The layer to partition.
        """

        assert self._optimalPartitions is not None, \
            "Need to call PartitionOptimizer.initialize() before running."

        # Propose a set of possible partitions for this layer, purely based on
        # its shape, not taking into account the post-layer partition.
        candidateDict = layer.getPartitionCandidates()

        # For each of the selected partition candidates of the post-layer,
        # choose again as many for the current layer.
        candidates = []
        for postLayerPartition in self._optimalPartitions:
            # Todo: The iterations in this loop are independent - parallelize!
            candidates += self.selectCandidates(candidateDict, layer,
                                                postLayerPartition)

        # Update the set of optimal partition candidates.
        costs = [computeTotalCost(partitionCandidate)
                 for partitionCandidate in candidates]
        candidates = np.array(candidates)[np.argsort(costs)]
        if self._storeAllCandidates:
            self._allCandidates = candidates
        self._optimalPartitions = candidates[:self.numCandidatesToCompute]

        # kernelIdMap of this layer is not needed anymore (can be many GB).
        layer.deleteKernelIdMap()

    def clearTemp(self):
        """Remove temporary data used during optimization."""

        candidates = self._allCandidates if self._storeAllCandidates \
            else self._optimalPartitions
        for candidate in candidates:
            for layer in self.getLayers(candidate):
                layer.clearTemp()

    def selectCandidates(self, candidateDict, layer, postLayerPartition):
        """From a set of candidates, choose a subset of valid partitions.

        :param dict candidateDict: Set of possible partition candidates.
        :param NxLayer | KerasLayer layer: The layer to partition.
        :param Layaer postLayerPartition: The next higher layer, which has been
            partitioned already.

        :return:
        :rtype: list[Layer]
        """

        # Iterate through set of partition candidates for this layer,
        # and validate candidate based on the partition of the subsequent
        # layer.
        candidates = []
        numToFind = self.numCandidatesToCompute
        numFound = 0
        for numCores in sorted(candidateDict):
            for numCoresPerAxis, coreShape in candidateDict[numCores]:

                partitionCandidate = tryCreatePartition(
                    numCoresPerAxis, coreShape, postLayerPartition, layer,
                    self.logdir)

                if partitionCandidate is not None:
                    candidates.append(partitionCandidate)

                    numFound = len(candidates)
                    if numFound == numToFind:
                        print('\n')
                        break

            if numFound == numToFind:
                break

        if numFound == 0:
            layer.exclusionCriteria.print()
            raise RuntimeError("No valid partition found.")
        if numFound < numToFind:
            self.logger.debug(
                "Found %s partition candidate%s, not the requested %s.",
                numFound, getS(numFound), numToFind)

        return candidates


def getDummyLayer(shape):
    """Create a dummy layer, typically as postLayer of the output layer.

    :param list | tuple | np.ndarray shape: Shape of layer.

    :return: Dummy layer.
    :rtype: Layer
    """

    return Layer('DummyPartitionFinalLayer', '', {}, {}, np.array([]),
                 np.ones(shape, int))


def tryCreatePartition(numCoresPerAxis, coreShape, postLayerPartition, layer,
                       logdir):
    """Try creating a partition of the layer.

    Fails if proposed partition exceeds one of the Loihi limits.

    :param np.ndarray | list | tuple numCoresPerAxis: Number of cores along
        each layer dimension.
    :param np.ndarray | list | tuple coreShape: The shape of the largest
        core.
    :param Layer postLayerPartition: The subsequent partitioned layer.
    :param KerasLayer | NxConv2D layer: The layer to partition.
    :param str logdir: Where to save plots.

    :return: Valid partition candidate.
    :rtype: Layer
    """
    outputShape = layer.output_shape[1:]

    # When using signed spikes the number of channels in the output
    # is doubled.
    if hasattr(layer, 'signed'):
        if layer.signed:
            outputShape = outputShape[:-1] + (2 * outputShape[-1],)
    coreIdMap = getCoreIdMapFromCoreShape(coreShape, outputShape,
                                          numCoresPerAxis)

    coreOccupancy = getCoreOccupancy(coreIdMap, numCoresPerAxis)

    if np.any(coreOccupancy > layer.maxNumCompartments):
        return

    multiplicityMap = layer.getMultiplicityMap(coreIdMap)

    partitionCandidate = Layer(layer.name, layer.__class__.__name__,
                               layer.compartmentKwargs, layer.connectionKwargs,
                               coreIdMap, multiplicityMap, postLayerPartition)

    # Pass coreOccupancy to partitionCandidate here only to be able to plot it
    # later when the partition has been stored to disk.
    partitionCandidate.coreOccupancy = coreOccupancy

    partitionCandidate = layer.compile(partitionCandidate)

    if partitionCandidate is None:
        print('.', end='', flush=True)
        return

    layer.validatePartition(partitionCandidate)

    layer.visualizePartition(logdir, partitionCandidate, coreIdMap,
                             coreOccupancy, multiplicityMap=multiplicityMap)

    print('x', end='', flush=True)

    return partitionCandidate


def getCostTerms(partitionedLayer):
    """Get cost terms of partitioned layer.

    Each cost term has been normalized with respect to the capacity of one
    core, to allow adding the cost terms up.

    :param Layer partitionedLayer: Partitioned layer.
    :return: Cost terms of partitioned layer.
    :rtype: CostTerms
    """

    # Skip dummy partition of final layer (exists only to provide a
    # multiplicityMap to the final layer).
    if partitionedLayer.postLayer is None:
        return

    # The cost of each layer accumulates the cost of the post layer partition.
    # This way, the cost of the currently lowest layer in the partitioning
    # loop represents the total cost of this partitioning path.
    postLayerCost = computeTotalCost(partitionedLayer.postLayer)

    return CostTerms(partitionedLayer.coreCost,
                     partitionedLayer.inputAxonCost,
                     partitionedLayer.outputAxonCost,
                     partitionedLayer.synapseCost,
                     postLayerCost)


def computeTotalCost(partitionedLayer, weights=None):
    """Get total cost of partitioned layer.

    :param Layer partitionedLayer: Partitioned layer.
    :param np.ndarray weights: Weight coefficients of cost terms.
    :return: Total cost of partitioned layer.
    :rtype: float
    """

    costTerms = getCostTerms(partitionedLayer)

    if costTerms is None:
        return 0

    costTerms = np.array(list(costTerms._asdict().values()))

    if weights is None:
        weights = np.ones(len(costTerms))
    elif np.isscalar(weights):
        weights = weights * np.ones(len(costTerms))

    assert len(weights) == len(costTerms)

    cost = np.dot(costTerms, weights)

    return cost
