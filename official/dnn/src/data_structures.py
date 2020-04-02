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

"""
Data structures for CNN partitioner.
"""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum

import numpy as np


class Layer:
    """Container for partitions in a layer and other high level information.

    :param int | str layerId: The layer id, as a label or digit.
    :param str layerType: The layer type, e.g. "Conv2D" or "Dense".
        Obtained by calling __class__.__name__ on a Keras layer.
    :param dict compartmentKwargs: Loihi compartment parameters, typical keys:
        ``vThMant``, ``biasExp``.
    :param dict connectionKwargs: Loihi connection parameters, typical keys:
        ``numWeightBits``, ``weightExponent``.
    :param np.ndarray coreIdMap: Integer tensor of same shape as layer. Each
        element indicates which core the neuron belongs to.
    :param np.ndarray multiplicityMap: Integer tensor of same shape as layer,
        except that the channel dimension is removed. Each element indicates
        to how many destination cores the neuron needs to send its spikes.
    :param Layer | None postLayer: The post-synaptic layer. Not applicable in
        output layer.
    """

    def __init__(self, layerId, layerType, compartmentKwargs, connectionKwargs,
                 coreIdMap, multiplicityMap, postLayer=None):

        assert isinstance(layerId, (int, np.integer, str))
        assert isinstance(layerType, str)
        assert isinstance(compartmentKwargs, dict)
        assert isinstance(connectionKwargs, dict)
        assert isinstance(coreIdMap, np.ndarray)
        assert isinstance(multiplicityMap, np.ndarray)
        if postLayer is not None:
            assert isinstance(postLayer, Layer)

        self.id = layerId
        self.type = layerType
        self.compartmentKwargs = compartmentKwargs
        self.connectionKwargs = connectionKwargs
        self.coreIdMap = coreIdMap
        self.multiplicityMap = multiplicityMap
        self.postLayer = postLayer

        self.coreOccupancy = None
        self._srcIdMap = {}
        self._isMapped = False
        self._partitions = []
        # Multiplier for complex layers
        multiplier = 2 if 'Complex' in layerType else 1
        self._numCores = 1 if coreIdMap.size == 0 \
            else multiplier * (np.max(coreIdMap) + 1)

        self._numSyn = 0
        self._numSynEntries = 0
        self._numSynMemWords = 0
        self._numInputAxons = 0
        self._numOutputAxons = 0
        self._numOutputAxonCfgEntries = 0

        self._inputAxonCost = 0
        self._outputAxonCost = 0
        self._synapseCost = 0

    def genCxResourceMap(self):
        """Generate a compartment resource map.

        Maps from global layer-wide compartment id to its
        ``(chipId, coreId, cxId)`` address.

        :raises AssertionError: Layer must be mapped before cxResourceMap can
            be generated.

        :return: cxResourceMap
        :rtype: np.ndarray
        """

        assert self._isMapped, \
            "Layer must be mapped before cxResourceMap can be generated."

        # Initialize cxResourceMap
        numCx = 0
        for p in self.partitions:
            numCx += p.compartmentGroup.numCompartments

        cxResourceMap = np.zeros((numCx, 3), int)

        # Populate cxResourceMap
        for p in self.partitions:
            # Get global layer-wide compartment ids
            cxGrp = p.compartmentGroup
            globalCxIds = cxGrp.relToAbsDestCxIdxMap

            cxResourceMap[globalCxIds, 0] = p.chipId
            cxResourceMap[globalCxIds, 1] = p.coreId
            cxResourceMap[globalCxIds, 2] = cxGrp.cxIds

        return cxResourceMap

    def addPartition(self, partition):
        """Add partition to layer, and update cost properties.

        :param Partition partition: Partition.
        """

        self._partitions.append(partition)

        self._numSyn += partition.numSyn
        self._numSynEntries += partition.numSynEntries
        self._numSynMemWords += partition.numSynMemWords
        self._numInputAxons += partition.numInputAxons
        self._numOutputAxons += partition.numOutputAxons
        self._numOutputAxonCfgEntries += partition.numOutputAxonCfgEntries

        self._inputAxonCost += partition.inputAxonCost
        self._outputAxonCost += partition.outputAxonCost
        self._synapseCost += partition.synapseCost

    def updateSrcIdMap(self, key, value):
        """Update source id map.

        :param int key: Global source id.
        :param tuple[InputAxonGroup, int] value: A tuple containing the input
            axon group and the source id relative to that axon.
        """

        if key not in self._srcIdMap.keys():
            self._srcIdMap[key] = []
        self._srcIdMap[key].append(value)

    @property
    def partitions(self):
        """List of layer partitions.

        :return: Layer partitions.
        :rtype: list[Partition]
        """

        return self._partitions

    @property
    def srcIdMap(self):
        """Source id map.

        This is a helper container that is built by layer ``L`` and used to
        construct output axons in layer ``L-1``.

        :return: Source id map. Dictionary mapping from global source ids to a
            tuple containing the input axon group and the source id relative
            to that axon.
        :rtype: dict[int, tuple[InputAxonGroup, int]]
        """

        return self._srcIdMap

    def clearTemp(self):
        """Clean up temporary data."""

        self._srcIdMap = None

    @property
    def numCores(self):
        """Number of cores used by layer.

        :return: Number of cores used by layer.
        :rtype: int
        """

        return self._numCores

    @property
    def numSyn(self):
        """Number of synapses in layer.

        :return: Number of synapses in layer.
        :rtype: int
        """

        return self._numSyn

    @property
    def numSynEntries(self):
        """Number of synEntries in layer.

        :return: Number of synEntries in layer.
        :rtype: int
        """

        return self._numSynEntries

    @property
    def numSynMemWords(self):
        """Number of synMemWords used by layer.

        :return: Number of synMemWords used by layer.
        :rtype: int
        """

        return self._numSynMemWords

    @property
    def numInputAxons(self):
        """Number of input axons in layer.

        :return: Number of input axons in layer.
        :rtype: int
        """

        return self._numInputAxons

    @property
    def numOutputAxons(self):
        """Number of output axons in layer.

        :return: Number of output axons in layer.
        :rtype: int
        """

        return self._numOutputAxons

    @property
    def numOutputAxonCfgEntries(self):
        """Number of output axon config entries in layer.

        :return: Number of output axon config entries in layer.
        :rtype: int
        """

        return self._numOutputAxonCfgEntries

    @property
    def inputAxonCost(self):
        """The total input axon cost of this layer.

        :return: Axon cost.
        :rtype: float
        """

        return self._inputAxonCost

    @property
    def outputAxonCost(self):
        """The total output axon cost of this layer.

        :return: Axon cost.
        :rtype: float
        """

        return self._outputAxonCost

    @property
    def synapseCost(self):
        """The total synapse cost of this layer.

        :return: Synapse cost.
        :rtype: float
        """

        return self._synapseCost

    @property
    def coreCost(self):
        """The total core cost of this layer.

        :return: Core cost.
        :rtype: int
        """

        return self._numCores

    @property
    def cost(self):
        """The total cost of partitioning this layer.

        :return: Partitioning cost of layer.
        :rtype: float
        """

        return (self.inputAxonCost + self.outputAxonCost +
                self.synapseCost + self.coreCost)

    def setMapped(self):
        """Set flag that this layer is mapped."""

        self._isMapped = True

    def asDict(self):
        """Return certain attributes of ``Layer`` as dict.

        :return: Selection of ``Layer`` attributes as dictionary.
        :rtype: dict
        """

        return {'id': self.id, 'multiplicityMap': self.multiplicityMap,
                'coreIdMap': self.coreIdMap,
                'coreOccupancy': self.coreOccupancy}


def serializeLayer(layer, path):
    """Save layer as pickle file.

    :param Layer layer: Layer to serialize.
    :param str path: Where to save output file.
    """

    postLayer = layer.postLayer

    # PostLayer will be None if the last layers of the network are "virtual"
    # layers like Flatten or Reshape. We do not want to serialize those.
    if postLayer is None:
        return

    # Temporarily overwrite pointer to parent layer to avoid redundant storage.
    layer.postLayer = postLayer.id

    with open(os.path.join(path, layer.id + '.pickle'), 'wb') as f:
        pickle.dump(layer, f)

    layer.postLayer = postLayer


def deserializeLayer(path, filename):
    """Load layer from pickle file.

    :param str path: Directory to saved file.
    :param str filename: Name of file.

    :return: Deserialized layer.
    :rtype: Layer
    """

    with open(os.path.join(path, filename), 'rb') as f:
        return pickle.load(f)


def saveMappableLayers(layers, path):
    """Store each partitioned and compiled layer as pickle file on disk.

    :param list[Layer] layers: List of Layer objects.
    :param str path: Where to save partition.
    """

    path = os.path.join(path, 'compiled_partitions')
    if not os.path.exists(path):
        os.makedirs(path)

    # Save each individual layer as pickle file.
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(serializeLayer, layer, path)
                   for layer in layers]
        for future in futures:
            future.result()


def loadMappableLayers(path):
    """Load compiled partitions from disk.

    The partitions are stored in the subfolder ``<path>/compiled_partitions``.
    The method expects one pickle file for each layer, and skips over any
    non-pickle files in that folder.

    :raises FileNotFoundError if the directory does not exist or contains no
    pickle files.

    :param str path: Path to stored partition files.
    :return: List of compiled ``Layer`` objects.
    :rtype: list[Layer]
    """

    path = os.path.join(path, 'compiled_partitions')

    if not os.path.exists(path):
        raise FileNotFoundError

    filenames = [f for f in os.listdir(path)
                 if os.path.splitext(f)[1] == '.pickle']

    # First, collect all compiled layers from disk.
    layers = {}
    layerNames = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(deserializeLayer, path, filename)
                   for filename in filenames]
        for future in futures:
            layer = future.result()
            layers[layer.id] = layer
            layerNames.append(layer.id)

    if len(layers) == 0:
        raise FileNotFoundError

    # Then, stitch them back together by replacing each layer's pointer to its
    # parent by the actual parent layer. (When saving the partition candidate,
    # we set layer.postLayer = layer.postLayer.id, to avoid redundant storage.)
    for layer in layers.values():
        # Skip output layer, whose postLayer is actually still the dummy
        # partition, not an id.
        if isinstance(layer.postLayer, Layer):
            continue
        postLayerName = layer.postLayer
        layer.postLayer = layers[postLayerName]
        layerNames.remove(postLayerName)

    # Finally, return ordered list of layers.
    outLayers = []
    # Start with bottom layer. This is the only layer that was not called as a
    # postLayer in the loop above.
    assert len(layerNames) == 1
    postLayer = layers[layerNames[0]]
    while True:
        outLayers.append(postLayer)
        postLayer = postLayer.postLayer
        if postLayer is None:
            # Remove last layer in this list, which is a dummy layer.
            return outLayers[:-1]


class Partition:
    """Container for compartments, connections and axons in a given partition.

    :param int partitionId: Id of partition.
    :param int chipCounter: Linearly increasing counter of chips used. Not
        identical to chipId assigned by board during mapping.
    :param int sizeInterleaved: The upper limit of the range of compartment
        indices in this partition after the compartments have been interleaved.
        May be larger than the number of compartments in the partition, because
        we space compartments by cIdxMult during interleaving, and not all of
        the inserted spaces may be filled.
    :param Layer parentLayer: The layer container this partition belongs to.
    :param bool isInhibitory: Flag to identify inhibitory partitions in complex
        layers.
    :param str resetMode: Sets reset mode for rate-coded layers. If 'hard',
        when a neuron spikes the membrane potential is reset to zero. If 'soft',
        when a neuron spikes the threshold will be subtracted from the
        membrane threshold.
    """

    def __init__(self, partitionId, chipCounter, sizeInterleaved, parentLayer,
                 inInhibitory=False, resetMode='hard'):

        assert isinstance(parentLayer, Layer)

        self.id = partitionId
        self.sizeInterleaved = sizeInterleaved
        self._layer = parentLayer

        self._inputAxonGroups = []
        self._outputAxonGroups = []
        self._synapseGroups = []
        self._compartmentGroup = None
        self._synFmts = []

        self._numSyn = 0
        self._numSynEntries = 0
        self._numSynMemWords = 0
        self._numInputAxons = 0
        self._numOutputAxons = 0
        self._numOutputAxonCfgEntries = 0

        self._inputAxonCost = 0
        self._outputAxonCost = 0
        self._synapseCost = 0
        self._cost = 0

        self._chipCounter = chipCounter
        self.chipId = None
        self.coreId = None

        self._isInhibitory = inInhibitory
        self._resetMode = resetMode

    # -------------------------------------------------------------------------
    # Getter
    @property
    def inputAxonGroups(self):
        """The input axon groups of this partition.

        :return: Input axon groups of partition.
        :rtype: list[InputAxonGroup]
        """

        return self._inputAxonGroups

    @property
    def synapseGroups(self):
        """The synapse groups of this partition.

        :return: Synapse groups of partition.
        :rtype: list[SynapseGroup]
        """

        return self._synapseGroups

    @property
    def compartmentGroup(self):
        """The Compartment group of this partition.

        :return: Compartment group of partition.
        :rtype: CompartmentGroup
        """

        return self._compartmentGroup

    @property
    def outputAxonGroups(self):
        """The Output axon groups of this partition.

        :return: Output axon groups of partition.
        :rtype: list[OutputAxonGroup]
        """

        return self._outputAxonGroups

    @property
    def synFmts(self):
        """The synFmts of this partition.

        :return: SynFmts of partition.
        :rtype: list[SynFmt]
        """

        return self._synFmts

    # -------------------------------------------------------------------------
    # add* interface
    def addInputAxonGroup(self, inputAxonGroup):
        """Add input axon group to partition, and update cost properties.

        :param InputAxonGroup inputAxonGroup: The input axon group to add.
        """

        assert isinstance(inputAxonGroup, InputAxonGroup)

        self._inputAxonCost += inputAxonGroup.cost
        self._numInputAxons += inputAxonGroup.numAxons

        self._inputAxonGroups.append(inputAxonGroup)

    def addSynapseGroup(self, synapseGroup):
        """Add synapse group to partition, and update cost properties.

        :param SynapseGroup synapseGroup: The synapse group to add.
        """

        assert isinstance(synapseGroup, SynapseGroup)

        self._synapseCost += synapseGroup.cost
        self._numSyn += synapseGroup.numSyn
        self._numSynEntries += synapseGroup.numSynEntries
        self._numSynMemWords += synapseGroup.numSynMemWords

        self._synapseGroups.append(synapseGroup)

    def addCompartmentGroup(self, compartmentGroup):
        """Add compartment group to partition.

        :param CompartmentGroup compartmentGroup: The compartment group to add.
        """

        assert isinstance(compartmentGroup, CompartmentGroup)
        self._compartmentGroup = compartmentGroup

    def addOutputAxonGroup(self, outputAxonGroup):
        """Add output axon group to partition, and update cost properties.

        :param OutputAxonGroup outputAxonGroup: The output axon group to add.
        """

        assert isinstance(outputAxonGroup, OutputAxonGroup)

        self._outputAxonCost += outputAxonGroup.cost
        self._numOutputAxons += outputAxonGroup.numAxons
        self._numOutputAxonCfgEntries += outputAxonGroup.numAxonCfgEntries

        self._outputAxonGroups.append(outputAxonGroup)

    def addSynFmt(self, synFmt):
        """Add synFmt to partition.

        :param SynFmt synFmt: The synFmt to add.
        """

        assert isinstance(synFmt, SynFmt)
        self._synFmts.append(synFmt)

    # -------------------------------------------------------------------------
    # Cost
    @property
    def numSyn(self):
        """Number of synapses in partition.

        :return: Number of synapses in partition.
        :rtype: int
        """

        return self._numSyn

    @property
    def numSynEntries(self):
        """Number of synEntries in partition.

        :return: Number of synEntries in partition.
        :rtype: int
        """

        return self._numSynEntries

    @property
    def numSynMemWords(self):
        """Number of synMemWords in partition.

        :return: Number of synMemWords in partition.
        :rtype: int
        """

        return self._numSynMemWords

    @property
    def numInputAxons(self):
        """Number of input axons in partition.

        :return: Number of input axons in partition.
        :rtype: int
        """

        return self._numInputAxons

    @property
    def numOutputAxons(self):
        """Number of output axons in partition.

        :return: Number of output axons in partition.
        :rtype: int
        """

        return self._numOutputAxons

    @property
    def numOutputAxonCfgEntries(self):
        """Number of output axon config entries in partition.

        :return: Number of output axon config entries in partition.
        :rtype: int
        """

        return self._numOutputAxonCfgEntries

    @property
    def inputAxonCost(self):
        """The total input axon cost of this partition.

        :return: Axon cost.
        :rtype: float
        """

        return self._inputAxonCost

    @property
    def outputAxonCost(self):
        """The total output axon cost of this partition.

        :return: Axon cost.
        :rtype: float
        """

        return self._outputAxonCost

    @property
    def synapseCost(self):
        """The total synapse cost of this partition.

        :return: Synapse cost.
        :rtype: float
        """

        return self._synapseCost

    @property
    def cost(self):
        """The total cost of this partition.

        :return: Partition cost.
        :rtype: float
        """

        return self.inputAxonCost + self.outputAxonCost + self.synapseCost

    # -------------------------------------------------------------------------
    # Misc
    @property
    def layer(self):
        """Return parent layer of partition.

        :return: Parent layer of partition.
        :rtype: Layer
        """

        return self._layer

    @property
    def chipCounter(self):
        """Return the chip counter of current partition.

        This counter is not identical to the chip id assigned by the board
        during mapping. We need the chip counter before an actual
        chip address is assigned, because this allows checking whether axons go
        to different chips, in which case they cost more.

        :return: Chip counter.
        :rtype int
        """

        return self._chipCounter

    @property
    def isInhibitory(self):
        """Flag to tell the mapper whether this partition is inhibitory.

        :return: "Inhibitory" flag.
        :rtype bool
        """

        return self._isInhibitory

    @property
    def resetMode(self):
        """Return the partition resetMode.

        Determines whether to reset voltage to zero or by subtraction.

        :return str resetMode
        :rtype str
        """

        return self._resetMode


class SynapseGroup:
    """Shared synapse group for a population of neurons.

    :param int groupId: Id of synapse group.
    :param list[list[SynEntry]] synEntries: List of lists of synEntries. Each
        sub-list contains the synEntries of a particular neuron in population.
    """

    __slots__ = ['id', '_synEntries', '_maxNumBitsPerWord', '_numSyn',
                 '_numSynEntries', '_numSynMemWords', '_maxNumWords',
                 '_maxNumSynMemWords', '_cost']

    def __init__(self, groupId, synEntries):
        self._maxNumSynMemWords = 16384
        self._maxNumBitsPerWord = 64

        self.id = groupId
        self._synEntries = synEntries

        self._numSyn = None
        self._numSynEntries = None
        self._numSynMemWords = None
        self._maxNumWords = None
        self._cost = None

        self._updateCost()

    def _updateCost(self):
        """Update the cost of this synapse group."""

        self._numSyn = 0
        self._numSynEntries = 0
        numBitsOfNeurons = []
        for synEntriesOfNeuron in self._synEntries:
            self._numSynEntries += len(synEntriesOfNeuron)
            numBitsOfNeuron = 0
            for synEntry in synEntriesOfNeuron:
                numBitsOfNeuron += synEntry.numBits
                self._numSyn += synEntry.numSyn
            numBitsOfNeurons.append(numBitsOfNeuron)

        # Keep track of largest number of synMemWords of the neurons within
        # source group as the number of words needs to be the same for all
        # neurons of the source group.
        numBitsOfNeurons = np.asarray(numBitsOfNeurons, int)
        numSynMemWords = np.ceil(numBitsOfNeurons / self._maxNumBitsPerWord)
        remainder = numBitsOfNeurons % self._maxNumBitsPerWord
        incNumSynMemWords = np.logical_or(remainder == 0, remainder >= 59)
        self._maxNumWords = int(np.max(numSynMemWords + incNumSynMemWords))

        numNeurons = len(self._synEntries)
        self._numSynMemWords = self._maxNumWords * numNeurons

        self._cost = self._numSynMemWords / self._maxNumSynMemWords

    @property
    def synEntries(self):
        """List of synaptic entries of this synapse group.

        :return: List of synaptic entries.
        :rtype: list[list[SynEntry]]
        """

        return self._synEntries

    @synEntries.setter
    def synEntries(self, synEntries):
        """Set synEntries of this synapse groups, and update cost properties.

        :param list[SynEntry] synEntries: Synaptic entries.
        """

        self._synEntries = synEntries
        self._updateCost()

    @property
    def numSyn(self):
        """Number of synapses in group.

        :return: Number of synapses in group.
        :rtype: int
        """

        return self._numSyn

    @property
    def numSynEntries(self):
        """Number of synEntries in group.

        :return: Number of synEntries in group.
        :rtype: int
        """

        return self._numSynEntries

    @property
    def numSynMemWords(self):
        """Number of synMemWords in group.

        :return: Number of synMemWords in group.
        :rtype: int
        """

        return self._numSynMemWords

    @property
    def maxSynMemLen(self):
        """Maximum number of synMemWords in group.

        :return: Maximum number of synMemWords.
        :rtype: int
        """

        return self._maxNumWords

    @property
    def cost(self):
        """The cost of this synapse group.

        :return: Cost of synapse group.
        :rtype: float
        """

        return self._cost


class CompartmentGroup:
    """Compartment group.

    :param np.ndarray cxIds: Compartment indices.
    :param np.ndarray biasMant: Bias mantissa for each compartment.
    :param np.ndarray biasExp: Bias exponent for each compartment.
    :param np.ndarray relToAbsDestCxIdxMap: Vector containing the absolute
        destination compartment indices. Allows mapping the indices relative
        to current partition to the global indices in layer.
    """

    __slots__ = ['cxIds', 'biasMant', 'biasExp', 'relToAbsDestCxIdxMap']

    def __init__(self, cxIds, biasMant, biasExp, relToAbsDestCxIdxMap):

        assert isinstance(cxIds, np.ndarray)
        assert isinstance(biasMant, np.ndarray)
        assert isinstance(biasExp, np.ndarray)
        assert isinstance(relToAbsDestCxIdxMap, np.ndarray)

        assert np.issubdtype(cxIds.dtype, np.integer)
        assert np.issubdtype(biasMant.dtype, np.integer)
        assert np.issubdtype(relToAbsDestCxIdxMap.dtype, np.integer)

        self.cxIds = cxIds
        self.biasMant = biasMant
        self.biasExp = biasExp
        # Maps the contiguous compartment indices of this partition to their
        # original place in the full kernelIdMatrix.
        self.relToAbsDestCxIdxMap = relToAbsDestCxIdxMap

    @property
    def numCompartments(self):
        """Number of compartments in compartment group.

        :return: Number of compartments in compartment group.
        :rtype: int
        """

        return len(self.cxIds)


class Compression(IntEnum):
    """Enumeration for the three compression modes of Loihi synapses."""

    SPARSE = 0
    RUNLENGTH = 1
    DENSE = 3


class SynFmt:
    """Synaptic format.

    :param int synFmtId: Id of synaptic format.
    :param int cIdxOffset: Offset of compartment indices.
    :param int cIdxMult: Multiplier of compartment indices.
    :param int numIdxBits: Number of index bits.
    :param int numSkipBits: Number of skip bits.
    :param int numWgtBits: Number of weight bits.
    :param int numDlyBits: Number of delay bits.
    :param int | Compression compression: Which compression mode is used to
        encode synapses.
    :param int signMode: Defines how inhibitory and excitatory weights are
        treated. 1: Signs are not shared. 2: Excitatory connections with
        shared sign. 3: Inhibitory connections with shared sign.
    :param bool softReset: True if synapse is used in soft-reset mode.
    """

    __slots__ = ['id', 'cIdxOffset', 'cIdxMult', 'numIdxBits', 'numSkipBits',
                 'numWgtBits', 'compression', 'signMode', 'numDlyBits',
                 'softReset']

    def __init__(self, synFmtId, cIdxOffset, cIdxMult, numIdxBits, numSkipBits,
                 numWgtBits, compression, signMode, numDlyBits=None,
                 softReset=False):

        assert isinstance(synFmtId, (int, np.integer))
        assert isinstance(cIdxOffset, (int, np.integer))
        assert isinstance(cIdxMult, (int, np.integer))
        assert isinstance(numIdxBits, (int, np.integer))
        assert isinstance(numSkipBits, (int, np.integer))
        assert isinstance(numWgtBits, (int, np.integer))
        assert isinstance(compression, (int, np.integer, Compression))
        assert isinstance(signMode, (int, np.integer))
        assert isinstance(numDlyBits, (int, np.integer, type(None)))

        self.id = synFmtId
        self.cIdxOffset = cIdxOffset
        self.cIdxMult = cIdxMult
        self.numIdxBits = numIdxBits
        self.numSkipBits = numSkipBits
        self.numWgtBits = numWgtBits
        self.compression = compression
        self.signMode = signMode
        self.numDlyBits = numDlyBits if numDlyBits is not None else 0
        self.softReset = softReset

    def asArray(self):
        """Transform synFmt into ndarray.

        :return: Numpy array containing the fields of this synFmt.
        :rtype: np.ndarray
        """

        return np.array([self.cIdxOffset, self.cIdxMult, self.numIdxBits,
                         self.numSkipBits, self.numWgtBits,
                         int(self.compression), self.signMode,
                         self.numDlyBits, self.softReset])

    def print(self, indentation=0):
        """Print fields of this synFmt.

        :param int indentation: How many spaces to insert before each line.
        """

        indent = " " * indentation
        print(indent + "compression={}".format(self.compression))
        print(indent + "signMode={}".format(self.signMode))
        print(indent + "cIdxMult={}".format(self.cIdxMult))
        print(indent + "cIdxOffset={}".format(self.cIdxOffset))
        print(indent + "numIdxBits={}".format(self.numIdxBits))
        print(indent + "numSkipBits={}".format(self.numSkipBits))
        print(indent + "numWgtBits={}".format(self.numWgtBits))
        print(indent + "numDlyBits={}".format(self.numDlyBits))
        print(indent + "softReset={}]".format(self.softReset))


class SynEntry:
    """Synaptic entry.

    :param int prefixOffset: Prefix offset.
    :param np.ndarray idxs: Indices of synapses.
    :param np.ndarray weights: Synapse weights.
    :param SynFmt synFmt: The synaptic format corresponding to this synEntry.
    :param np.ndarray kernelIds: Global indices of synapse weights. Only needed
        for plotting and reconstructing kernelIdMap to validate partition.
    :param np.ndarray delays: Synapse delays.
    """

    __slots__ = ['prefixOffset', '_idxs', 'kernelIds', 'weights', '_synFmt',
                 '_numSyn', '_numSynBits', '_numPrefixBits', '_numBits',
                 '_cost', 'delays']

    def __init__(self, prefixOffset, idxs, weights, synFmt, kernelIds=None,
                 delays=None):

        assert isinstance(prefixOffset, (int, np.integer))
        assert isinstance(idxs, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert isinstance(synFmt, SynFmt)
        assert isinstance(delays, (type(None), np.ndarray))

        if kernelIds is not None:
            assert isinstance(kernelIds, np.ndarray)

        self.prefixOffset = prefixOffset
        self._idxs = idxs
        self.weights = weights
        self.delays = delays
        self._synFmt = synFmt
        self.kernelIds = kernelIds

        self._numSyn = None
        self._numSynBits = None
        self._numPrefixBits = None
        self._numBits = None
        self._cost = None

        self._updateCost()

    def _updateCost(self):
        """Update cost of this synEntry."""

        self._numSyn = len(self._idxs)

        self._numSynBits = self._numSyn * (self._synFmt.numWgtBits +
                                           self._synFmt.numDlyBits)

        if self._synFmt.compression == Compression.SPARSE:
            self._numSynBits += self._numSyn * self._synFmt.numIdxBits
        elif self._synFmt.compression == Compression.RUNLENGTH:
            self._numSynBits += self._numSyn * self._synFmt.numSkipBits

        self._numPrefixBits = 4 + 6  # synMemFmtId + numSyn
        if self._synFmt.compression is not Compression.SPARSE:
            self._numPrefixBits += self._synFmt.numIdxBits

        self._numBits = self._numPrefixBits + self._numSynBits

        self._cost = self._numBits

    @property
    def numSyn(self):
        """Number of synapses in this synEntry.

        :return: Number of synapses.
        :rtype: int
        """

        return self._numSyn

    @property
    def cost(self):
        """Cost of this synEntry.

        :return: The cost of this synEntry.
        :rtype: float
        """

        return self._cost

    @property
    def numSynBits(self):
        """Number of synapse bits used by this synEntry.

        :return: The number of synapse bits of this synEntry.
        :rtype: int
        """

        return self._numSynBits

    @property
    def numPrefixBits(self):
        """Number of prefix bits of this synEntry.

        :return: The number of prefix bits of this synEntry.
        :rtype: int
        """

        return self._numPrefixBits

    @property
    def numBits(self):
        """Number of bits of this synEntry.

        :return: The number of bits of this synEntry.
        :rtype: int
        """

        return self._numBits

    def getCxIds(self):
        """Get the compartment indices of this synEntry.

        :return: Compartment indices.
        :rtype: np.ndarray
        """

        if self._synFmt.compression is not Compression.RUNLENGTH:
            # For SPARSE and DENSE, idx specifies indices directly
            cxIds = np.copy(self.idxs)
        else:
            # For RUNLENGTH, idx specifies differences
            cxIds = np.cumsum(self.idxs)

        # For RUNLENGTH and DENSE, offset by prefixOffset. Do this before
        # scaling with cIdxMult, because prefixOffset was obtained after
        # dividing cxIds by cIdxMult.
        if self._synFmt.compression is not Compression.SPARSE:
            cxIds += self.prefixOffset

        # Scale by cIdxOffset and cIdxMult
        cxIds = self._synFmt.cIdxOffset + cxIds * (self._synFmt.cIdxMult + 1)

        return cxIds

    @property
    def idxs(self):
        """Destination compartment indices of this synEntry.

        :return: Index vector.
        :rtype: np.ndarray
        """

        return self._idxs

    @idxs.setter
    def idxs(self, idxs):
        """Set indices of this synEntry, and udpate cost properties.

        :param np.ndarray idxs: The destination compartment indices of this
            synEntry.
        """

        self._idxs = idxs
        self._updateCost()

    @property
    def synFmt(self):
        """SynFmt of this synEntry.

        :return: The synFmt corresponding to this synEntry.
        :rtype: SynFmt
        """

        return self._synFmt

    @synFmt.setter
    def synFmt(self, synFmt):
        """Set synFmt, and update cost properties of this synEntry.

        :param SynFmt synFmt: The synFmt corresponding to this synEntry.
        """

        self._synFmt = synFmt
        self._updateCost()

    @property
    def synFmtId(self):
        """Id of the synFmt corresponding to this synEntry.

        :return: The id of the synFmt corresponding to this synEntry.
        :rtype: int
        """

        return self._synFmt.id

    def print(self, indentation=0):
        """Print fields of this synEntry.

        :param int indentation: How many spaces to insert before each line.
        """

        indent = " "*indentation
        print(indent+"SynEntry:")
        print(indent+"  numSyn={}".format(self.numSyn))
        print(indent+"  numPrefixBits={}".format(self.numPrefixBits))
        print(indent+"  numSynBits={}".format(self.numSynBits))
        print(indent+"  idx={}".format(self.idxs))
        print(indent+"  kernelIds={}".format(self.kernelIds))


class InputAxonGroup:
    """Input axon group.

    :param np.ndarray srcNodeIds: Vector of source node ids of this axon group.
    :param np.ndarray multiplicity: Integer vector where each element indicates
        to how many destination cores the source node needs to send its spikes.
    :param SynapseGroup synGroup: The synapse group corresponding to this axon
        group.
    :param int cxBase: The cxBase of this axon group.
    :param Partition parentPartition: The parent partition of this axon group.
    """

    def __init__(self, srcNodeIds, multiplicity, synGroup, cxBase,
                 parentPartition):

        assert isinstance(srcNodeIds, np.ndarray)
        assert isinstance(multiplicity, np.ndarray)
        assert isinstance(synGroup, SynapseGroup)
        assert isinstance(cxBase, (int, np.integer))
        assert isinstance(parentPartition, Partition)

        self._id = hash(self)

        self._maxNumAxonCfgEntries = 4096

        self.srcNodeIds = srcNodeIds
        self._multiplicity = multiplicity
        self.synGroup = synGroup
        self.cxBase = cxBase
        self._partition = parentPartition

        self._numNodes = None
        self._numAxons = None
        self._cost = None

        self._updateCost()

    @property
    def id(self):
        """Unique identifyer of this axon group.

        Need this to be able to map from output axons in layer L to input axons
        in layer L+1. Each output axon directly points to its corresponding
        input axon, but after serializing and loading the layer, this
        reference does not point to the same object in memory any more.
        """

        return self._id

    @property
    def partition(self):
        """The parent partition of this axon group.

        :return: Parent partition.
        :rtype: Partition
        """

        return self._partition

    @property
    def chipCounter(self):
        """Chip counter. Not identical to chipId assigned by board.

        :return: Linearly increasing chip counter.
        :rtype: int
        """

        return self._partition.chipCounter

    @property
    def numNodes(self):
        """Number of nodes in this axon group.

        :return: The number of nodes in this axon group.
        :rtype: int
        """

        return self._numNodes

    def _updateCost(self):
        """Update the cost of this axon group."""

        self._numNodes = len(self.multiplicity)

        numDiscrete = np.sum(self.multiplicity > 1)
        hasShared = np.any(self.multiplicity == 1)
        self._numAxons = numDiscrete + hasShared

        self._cost = self._numAxons / self._maxNumAxonCfgEntries

    @property
    def multiplicity(self):
        """The number of cores that this axon projects to.

        :return: Multiplicity.
        :rtype: np.ndarray
        """

        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity):
        """Set multiplicity attribute of axon, and update axon cost properties.

        :param np.ndarray multiplicity: Multiplicity vector. Each element
            specifies to how many cores this axon projects.
        """

        self._multiplicity = multiplicity
        self._updateCost()

    @property
    def numAxons(self):
        """Number of axons in this group.

        :return: The number of axons in this group.
        :rtype: int
        """

        return self._numAxons

    @property
    def cost(self):
        """Cost of this input axon group, measured by number of synMap entries.

        :return: Cost of axon group.
        :rtype: int
        """

        return self._cost


class OutputAxonGroup:
    """Output axon group.

    :param np.ndarray cxIds: Vector of compartment indices of this axon group.
    :param np.ndarray multiplicity: Integer vector where each element indicates
        to how many destination cores the source node needs to send its spikes.
    :param np.ndarray relSrcIds: Vector of relative source indices.
    :param InputAxonGroup inAxGrp: Input axon group corresponding to this
        output axon group.
    :param Partition parentPartition: The parent partition of this axon group.
    """

    def __init__(self, cxIds, multiplicity, relSrcIds, inAxGrp,
                 parentPartition):
        assert isinstance(cxIds, np.ndarray)
        assert isinstance(multiplicity, np.ndarray)
        assert isinstance(relSrcIds, np.ndarray)
        assert isinstance(inAxGrp, InputAxonGroup)
        assert isinstance(parentPartition, Partition)

        self._maxNumAxonCfgEntries = 4096

        self.cxIds = cxIds
        self._multiplicity = multiplicity
        self.relSrcIds = relSrcIds
        self.inAxGrpId = inAxGrp.id
        self._postChipCounter = inAxGrp.chipCounter
        self._partition = parentPartition

        self._numNodes = None
        self._numAxons = None
        self._numAxonCfgEntries = None
        self._cost = None

        self._updateCost()

    @property
    def partition(self):
        """The parent partition of this axon group.

        :return: Parent partition.
        :rtype: Partition
        """

        return self._partition

    @property
    def chipCounter(self):
        """Chip counter. Not identical to chipId assigned by board.

        :return: Linearly increasing chip counter.
        :rtype: int
        """

        return self._partition.chipCounter

    @property
    def numNodes(self):
        """Number of nodes in this axon group.

        :return: The number of nodes in this axon group.
        :rtype: int
        """

        return self._numNodes

    @property
    def multiplicity(self):
        """The number of cores that this axon projects to.

        :return: Multiplicity.
        :rtype: np.ndarray
        """

        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity):
        """Set multiplicity attribute of axon, and update axon cost properties.

        :param np.ndarray multiplicity: Multiplicity vector. Each element
            specifies to how many cores this axon projects.
        """

        self._multiplicity = multiplicity
        self._updateCost()

    def _updateCost(self):
        """Update the cost of this axon group."""
        assert len(self.multiplicity) > 0

        self._numNodes = len(self.multiplicity)

        discreteCost = 1
        sharedCost = 2
        if self.chipCounter != self._postChipCounter:
            discreteCost += 1
            sharedCost += 1
        numDiscrete = np.sum(self.multiplicity > 1)
        hasShared = np.any(self.multiplicity == 1)
        self._numAxons = numDiscrete + hasShared
        self._numAxonCfgEntries = \
            numDiscrete * discreteCost + hasShared * sharedCost
        self._cost = self._numAxonCfgEntries / self._maxNumAxonCfgEntries

    @property
    def numAxonCfgEntries(self):
        """Number of axon config entries.

        :return: The number of axon config entries.
        :rtype: int
        """

        return self._numAxonCfgEntries

    @property
    def numAxons(self):
        """Number of axons in this group.

        :return: The number of axons in this group.
        :rtype: int
        """

        return self._numAxons

    @property
    def cost(self):
        """Cost of this output axon group.

        :return: Cost of axon group.
        :rtype: int
        """

        return self._cost
