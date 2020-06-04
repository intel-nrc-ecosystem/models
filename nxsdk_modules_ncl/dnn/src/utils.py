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

"""Utility functions for DNN partitioner."""
import os
import time
from collections import OrderedDict
from functools import wraps

import numpy as np
from scipy.sparse import coo_matrix

# Computes the length of elements in a numpy array.
getLen = np.vectorize(lambda x: len(x))


def normalizeImageDim(arr):
    """Add or remove dimensions from array to make it 2D.

    :param np.ndarray arr: Array to transform.

    :return: 2D array.
    :rtype: np.ndarray
    """

    if arr.ndim == 1:
        return np.expand_dims(arr, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, 0]
    raise NotImplementedError("Invalid image dimensions.")


def to_integer(weights, biases, bitwidth, normalize=True):
    """Convert weights and biases to integers.

    :param np.ndarray weights: 2D or 4D weight tensor.
    :param np.ndarray biases: 1D bias vector.
    :param int bitwidth: Number of bits for integer conversion.
    :param bool normalize: Whether to normalize weights and biases by the
        common maximum before quantizing.

    :return: The quantized weights and biases.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    max_val = np.max(np.abs(np.concatenate([weights, biases], None))) \
        if normalize else 1
    a_min = -2**bitwidth
    a_max = - a_min - 1
    weights = np.clip(weights / max_val * a_max, a_min, a_max).astype(int)
    biases = np.clip(biases / max_val * a_max, a_min, a_max).astype(int)
    return weights, biases


def getCoreOccupancy(coreIdMap, numCoresPerAxis):
    """Get the number of compartments per core for a given layer.

    :param np.ndarray coreIdMap: Tensor of same shape as layer, where each
        entry indicates the core id of the corresponding neuron.
    :param np.ndarray | list | tuple numCoresPerAxis: The number of cores along
        each axis of layer.

    :return: The number of compartments per core.
    :rtype: np.ndarray
    """

    coreOccupancy = []
    numCores = np.max(coreIdMap) + 1
    for coreId in range(numCores):
        coreOccupancy.append(np.count_nonzero(coreIdMap == coreId))

    return np.reshape(coreOccupancy, numCoresPerAxis)


def getCoreIdMapFromCoreShape(coreShape, layerShape, numCoresPerAxis):
    """Get the coreIdMap from the core shape.

    Assumes layer is 3D (but any axis may have size 1).

    :param np.ndarray | list | tuple coreShape: The shape of the largest core.
    :param np.ndarray | list | tuple layerShape: The shape of the layer.
    :param np.ndarray | list | tuple numCoresPerAxis: The number of cores along
        each axis of layer.

    :return: tuple[coreIds, coreCounter]
        where
        np.ndarray coreIds is the coreIdMap and
        int coreCounter is the number of cores of this layer.
    """

    # Allocate coreIdMap, which may be larger than the actual layer in case
    # ``coreShape`` does not evenly divide ``layerShape``. We remove the excess
    # rows and columns later.
    coreIds = -np.ones(np.multiply(coreShape, numCoresPerAxis), int)
    for coreCounter, core in enumerate(np.ndindex(tuple(numCoresPerAxis))):
        ind = [slice(coreShape[ax] * d, coreShape[ax] * (d + 1))
               for ax, d in enumerate(core)]
        coreIds[tuple(ind)] = coreCounter

    # Remove rows and columns in case ``coreShape`` does not evenly divide
    # ``shape``.
    if not np.array_equal(layerShape, coreIds.shape):
        ind = [slice(d) for d in layerShape]
        coreIds = coreIds[tuple(ind)]

    return coreIds


def getInversePermutation(permutation):
    """Invert a given permutation vector.

    :param list | tuple | np.ndarray permutation: Permutation vector to invert.
    :return: Inverted permutation vector.
    :rtype: list
    """

    inverse = [0] * len(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i

    return inverse


def getWeightsFromIds(weights, kernelIds):
    """Get the weights corresponding to the provided indices.

    .. note:: The kernel indices are assumed to start with 1 instead of 0.
        This is because we use sparse matrices to represent kernel ids,
        which would remove zero-based ids. This method subtracts 1 from the
        kernel indices for proper indexing of the weight matrix.

    :param np.ndarray weights: Array of weight values.
    :param np.ndarray kernelIds: Array of indices.
    :return: Array of weights.
    :rtype: np.ndarray
    """

    return weights[kernelIds - 1]


def _getPadding(inputShape, padding, kernelShape, strides, dilation,
                inbound_nodes=None):
    """Get the padding to be applied on top, bottom, left and right.

    :param np.ndarray | tuple | list inputShape: Shape of layer input (not
        including the batch size).
    :param str padding: The type of padding ('same', 'valid').
    :param np.ndarray | tuple | list kernelShape: Shape of kernel (height,
        width).
    :param np.ndarray | tuple | list strides: Strides along height and width
        dimension.
    :param tuple dilation: Dilation rate.
    :param list inbound_nodes: The layers feeding into this layer.
    """

    py0 = py1 = px0 = px1 = 0

    # First, check if the previous layer was a ZeroPadding layer.
    if inbound_nodes is not None:
        if len(inbound_nodes) == 1:
            inbound = inbound_nodes[0]
        else:
            raise NotImplementedError
        if inbound.__class__.__name__ == 'ZeroPadding2D':
            ((py0, py1), (px0, px1)) = inbound.padding

    if padding == 'valid':
        pass
    elif padding == 'same':
        height, width = inputShape[:-1]
        ky, kx = kernelShape
        sy, sx = strides
        dy, dx = dilation
        qy = (sy * (np.ceil(height / sy) - 1) + dy * ky - height) / 2
        py0 = py1 = int(qy)
        # No need to pad if kernel size is 1.
        if qy % 1 and ky > 1:
            py1 += 1
        qx = (sx * (np.ceil(width / sx) - 1) + dx * kx - width) / 2
        px0 = px1 = int(qx)
        if qx % 1 and kx > 1:
            px1 += 1
    elif padding == 'causal':
        py0 = dilation[0] * (kernelShape[0] - 1)
    else:
        raise NotImplementedError

    return py0, py1, px0, px1


def _getMultiplicityMapConvlike(coreIdMap, inputShape, kernelShape, strides,
                                padding, dilation, zeroPadding=None):
    """Generate multiplicity map for a depthwise-separable Conv2D layer.

    :param np.ndarray coreIdMap: Tensor of same shape as the layer. Each entry
        specifies the core id to which that neuron belongs.
    :param np.ndarray | tuple | list inputShape: Shape of layer input (not
        including batch size).
    :param np.ndarray | tuple | list kernelShape: Shape of kernel (height,
        width).
    :param np.ndarray | tuple | list strides: Strides along height and width
        dimension.
    :param np.ndarray | tuple | list padding: Zero-padding applied to top,
        bottom, left and right of input.
    :param tuple dilation: Dilation rate.
    :param tuple zeroPadding: Zero padding values from previous padding layer.

    :return: Multiplicity map, which has the same shape as the input to
        current layer except that the channel axis is removed. Each entry
        specifies to how many cores that neuron sends its spikes.
    :rtype: np.ndarray
    """

    # Subtract zero padding of previous layer from inputShape. Used if previous
    # layer
    # was ZeroPadding.
    if zeroPadding is not None:
        py0, py1, px0, px1 = zeroPadding
        inputShape = (inputShape[0] - (py0 + py1), inputShape[1] - (px0 + px1),
                      inputShape[2])

    height, width = inputShape[:-1]
    ky, kx = kernelShape
    sy, sx = strides
    py0, py1, px0, px1 = padding
    dy, dx = dilation

    # Add zero-padding
    height += py0 + py1
    width += px0 + px1

    destinationCoreIdMap = np.empty((height, width), object)
    for i in range(destinationCoreIdMap.shape[0]):
        for j in range(destinationCoreIdMap.shape[1]):
            destinationCoreIdMap[i, j] = set()

    # Loop over width of previous layer.
    for j in np.arange(0, width - dx * kx + 1, sx):
        jj = j // sx  # Width idx of this layer.
        # Loop over height of previous layer.
        for i in np.arange(0, height - dy * ky + 1, sy):
            ii = i // sy  # Height idx of this layer.
            coreId = coreIdMap[ii, jj, 0]
            # Add the coreId to every location in the input feature map
            # that connects to (ii, jj) via the kernel. Only need to
            # consider first channel here; others are factored in below.
            for m in range(ky):
                for n in range(kx):
                    destinationCoreIdMap[i + dy * m, j + dx * n].add(coreId)

    multiplicityMap = getLen(destinationCoreIdMap)

    # Remove zero-padding.
    multiplicityMap = multiplicityMap[py0: height - py1, px0: width - px1]

    return multiplicityMap


def _genKernelIdMap(inputShape, outputShape, padding, strides, kernelShape,
                    dilation, isDepthwise=False, zeroPadding=None):
    """Generate a KernelIdMap of the layer.

    :param np.ndarray | tuple | list inputShape: Shape of layer input (not
        including batch size).
    :param np.ndarray | tuple | list outputShape: Layer shape (not including
        batch size).
    :param np.ndarray | tuple | list padding: Zero-padding applied to top,
        bottom, left and right of input.
    :param np.ndarray | tuple | list strides: Strides along height and width
        dimension.
    :param np.ndarray | tuple | list kernelShape: Shape of kernel (height,
        width).
    :param tuple dilation: Dilation rate.
    :param bool isDepthwise: Whether the layer operates on each channel in the
        input feature map separately (e.g. pooling or depth-wise separable
        convolution).
    :param tuple zeroPadding: Zero padding values from previous padding layer.

    :return: kernelIdMap.
    :rtype: coo_matrix
    """

    # Subtract zero padding of previous layer from inputShape. Used if previous
    # layer was ZeroPadding.
    if zeroPadding is not None:
        py0, py1, px0, px1 = zeroPadding
        inputShape = (inputShape[0] - (py0 + py1), inputShape[1] - (px0 + px1),
                      inputShape[2])

    inputSize = np.asscalar(np.prod(inputShape))
    outputSize = np.asscalar(np.prod(outputShape))
    numStrides = np.asscalar(np.prod(outputShape[:-1]))
    outIds = np.arange(numStrides)
    inputIdMap = np.reshape(np.arange(inputSize), inputShape, 'F')
    if isDepthwise:
        inputChannels = 1
        inputShift = inputShape[0] * inputShape[1]
    else:
        inputChannels = inputShape[-1]
        inputShift = 0

    # Add zero-padding.
    doPad = np.any(padding)
    if doPad:
        py0, py1, px0, px1 = padding
        inputIdMap = np.pad(inputIdMap, ((py0, py1), (px0, px1), (0, 0)),
                            'constant', constant_values=-1)

    dy, dx = dilation

    # Get indices of input neurons where conv kernel will be applied.
    outIdsY, outIdsX, _ = np.unravel_index(outIds, outputShape, 'F')
    inIdsY = outIdsY * strides[0]
    inIdsX = outIdsX * strides[1]

    # Generate a flat dummy kernel. Need to offset by 1 because lil_matrix does
    # not store zeros.
    kernelSize = np.asscalar(np.prod(kernelShape)) * inputChannels
    kIds = np.arange(kernelSize) + 1

    inputIds = []
    kernelIds = []
    outputIds = []
    for outId, inIdY, inIdX in zip(outIds, inIdsY, inIdsX):
        inIds = inputIdMap[slice(inIdY, inIdY + dy * kernelShape[0], dy),
                           slice(inIdX, inIdX + dx * kernelShape[1], dx),
                           slice(inputChannels)]
        inIds = np.ravel(inIds, 'F')

        # Remove zero-padding.
        if doPad:
            paddingMask = inIds > -1
            inIds = inIds[paddingMask]
            _kIds = kIds[paddingMask]
        else:
            _kIds = kIds

        inputIds.append(inIds)
        kernelIds.append(_kIds)
        outputIds.append([outId] * len(_kIds))

    inputIds = np.concatenate(inputIds)
    kernelIds = np.concatenate(kernelIds)
    outputIds = np.concatenate(outputIds)

    # Insert kernel into all channels of feature map.
    data = []
    rows = []
    cols = []
    for cId in range(outputShape[-1]):
        # Increment kernel ids by the kernelSize for the next channel.
        data.append(kernelIds + cId * kernelSize)
        cols.append(inputIds + cId * inputShift)
        rows.append(outputIds + cId * numStrides)

    data = np.concatenate(data)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    return coo_matrix((data, (rows, cols)), (outputSize, inputSize), int)


def _interleave(kMap, coreShape, destinationGroups):
    """Interleave kernelIdMap to account for cxBase bug.

    :param lil_matrix kMap: kernelIdMap.
    :param np.ndarray | list[int] coreShape: Shape of current core.
    :param list destinationGroups: Sizes of non-overlapping destination groups.

    :return: (interleavedMap, permutedDestCxIdxs) tuple.
    """

    numDestGroups = len(destinationGroups)

    permutedDestCxIdxs = np.concatenate(
        [cxBaseOffset + numDestGroups * np.arange(groupSize)
         for cxBaseOffset, groupSize in enumerate(destinationGroups)])

    sizeInterleaved = np.max(destinationGroups) * numDestGroups

    permutedDestCxIdxs = np.concatenate(
        [permutedDestCxIdxs + j * sizeInterleaved
         for j in range(coreShape[-1])])

    # The interleavedMap we construct here is transposed wrt kMap. This is to
    # make subsequent slicing operations more efficient.
    newShape = (kMap.shape[1], sizeInterleaved * coreShape[-1])
    data = np.concatenate(kMap.data)
    rowIdxs = np.concatenate(kMap.rows)
    colIdxs = np.repeat(permutedDestCxIdxs, [len(d) for d in kMap.data])
    interleavedMap = coo_matrix((data, (rowIdxs, colIdxs)), newShape)

    return interleavedMap.tolil(), permutedDestCxIdxs


def _getDestinationGroups(kMap, coreShape, inputShape, isDepthwise):
    """Get non-overlapping destination groups and corresponding cxBaseOffsets.

    :param lil_matrix kMap: kernelIdMap of current core.
    :param np.ndarray | list[int] coreShape: Shape of current core.
    :param np.ndarray | tuple | list inputShape: Shape of input to this layer
        (not including batch size).
    :param bool isDepthwise: Whether the layer is depthwise-separable, i.e.
        applies convolutions across each input channel separately.

    :return: (groups, cxBaseOffsets) tuple.
    """

    sizeSrcGroup = inputShape[0]
    numChannels = inputShape[-1]
    numSrcGroups = inputShape[1] if len(inputShape) == 3 else 1

    kMap = kMap.tocoo()

    # It suffices to compute the destination groups on the part of the kMap
    # that belongs to the first input and output channel of the layer.
    # But in depthwise-separable convolution layers, the kernelIdMap is
    # diagonal, so kMap (which is a subsection of the kernelIdMap) may be all
    # zero in the range of the first input channel. Thus we need to find the
    # start index of the unique channel that is non-zero.
    s = sizeSrcGroup * numSrcGroups
    col0 = (np.min(kMap.col) // s) * s if isDepthwise else 0
    srcMask = np.logical_and(col0 <= kMap.col, kMap.col < col0 + s)
    col = kMap.col[srcMask]
    row = kMap.row[srcMask]
    row0 = np.min(kMap.row) if isDepthwise else 0
    destMask = np.logical_and(row0 <= row,
                              row < row0 + np.prod(coreShape[:-1]))
    col = col[destMask]
    row = row[destMask]

    prevGroupEnd = 0
    cxBaseOffsets = []
    cxBaseOffset = -1
    groups = []
    for groupId in range(numSrcGroups):

        # Get slice corresponding to one source group.
        groupMask = np.logical_and(col0 + groupId * sizeSrcGroup <= col,
                                   col < col0 + (groupId + 1) * sizeSrcGroup)
        cxIdRange = row[groupMask]

        # If source group has any synapses, get range of destination ids.
        if len(cxIdRange):
            # row indices of coo_matrix are ordered, so we can just take the
            # first and last.
            groupOffset = cxIdRange[0]
            endId = cxIdRange[-1]

            if groupOffset == prevGroupEnd:
                # Start new non-overlapping destination group.
                groups.append(endId - groupOffset + 1)
                prevGroupEnd = endId + 1
                cxBaseOffset += 1
            elif groupId + 1 == numSrcGroups and endId >= prevGroupEnd:
                # The last group may be smaller.
                groups.append(endId - prevGroupEnd + 1)

        cxBaseOffsets.append(cxBaseOffset)

    # Repeat for all input channels.
    cxBaseOffsets = np.array(cxBaseOffsets * numChannels)

    return groups, cxBaseOffsets


def _getSizeInterleaved(coreShape, destinationGroups, neuronSize):
    """Get the size of the partition after interleaving.

    This number is equivalent to the range of compartment indices, but may be
    larger than the actual number of compartments.

    :param np.ndarray | list[int] coreShape: Shape of current core.
    :param list destinationGroups: Sizes of non-overlapping destination groups.
    :param int neuronSize: The number of compartments per neuron.

    :return: Size of interleaved partition.
    :rtype: int
    """

    return np.max(destinationGroups) * len(destinationGroups) * coreShape[-1] \
        * neuronSize


def _getUniqueSourceGroups(interleavedMap, cxBaseOffsets, sizeSrcGroup):
    """Get unique source groups.

    :param lil_matrix interleavedMap: The interleaved kernelIdMap. Rows
        correspond to input neurons, columns to output neurons.
    :param np.ndarray cxBaseOffsets: Vector of cxBaseOffsets of each source
        group.
    :param int sizeSrcGroup: Number of neurons belonging to each group.

    :return: (uniqueGroups, synListPtrs) tuple
    """

    rows = interleavedMap.rows
    data = interleavedMap.data

    groups = []
    groupsFlat = OrderedDict()
    prevLength = None
    axis = 0
    for groupId, cxBaseOffset in enumerate(cxBaseOffsets):
        group = []
        for relSrcId in range(sizeSrcGroup):
            # Get column corresponding to one neuron.
            absSrcId = relSrcId + groupId * sizeSrcGroup
            row = rows[absSrcId]
            if len(row):
                kernelIds = data[absSrcId]
                cxIds = [r - cxBaseOffset for r in row]

                groupsFlat.setdefault(groupId, [])
                groupsFlat[groupId] += kernelIds + cxIds

                group.append([np.array(kernelIds), np.array(cxIds)])

        if len(group):
            groups.append(group)
            length = len(groupsFlat[groupId])
            if prevLength is None:
                prevLength = length
            if prevLength != length:
                axis = None

    _, ids, invIds = np.unique(list(groupsFlat.values()), axis=axis,
                               return_index=True, return_inverse=True)

    uniqueGroups = [groups[i] for i in ids]

    synListPtrs = {}
    i = 0
    for groupId, group in groupsFlat.items():
        if len(group):
            synListPtrs[groupId] = invIds[i]
            i += 1

    return uniqueGroups, synListPtrs


def extendShape(shape):
    """Add dummy dimensions to make ``shape`` 3D.

    :raises NotImplementedError: If ``shape`` is neither 1, 2 or 3D.

    :param np.ndarray | list | tuple shape: Shape.

    :return: 3D shape.
    :rtype: tuple
    """

    if len(shape) == 1:
        return shape[0], 1, 1
    if len(shape) == 2:
        return shape[0], 1, shape[1]
    if len(shape) == 3:
        return shape

    raise NotImplementedError


def getPartitionCandidates(layerShape, maxNumCompartments=1024, order=None):
    """Get possible partition configurations for a layer.

    :param list | tuple | np.ndarray layerShape: The shape of the layer.
    :param int maxNumCompartments: The maximum number of compartments per core.
    :param list | tuple | np.ndarray order: The order in which we loop over the
        axes of the layer. The ``order`` parameter affects the ordering of
        candidates in the output dictionary. Allows to implicitly pre-sort the
        partition list according to a heuristic on the cost. For instance, in
        standard convolution layers, output channel partitioning is the most
        expensive because it requires duplicating the whole input layer. In
        this case, we want to iterate over the channel dimension last:
        ``order=[2, 0, 1]``. In depthwise separable convolutions, neurons do
        not fan-out to more than one channel, so separating channels is
        cheapest: ``order=[0, 1, 2]``. Splicing along the height is always
        more expensive than along width because we stride in Fortran style.
    :return: Dictionary of partition candidates. Each value is a 3D array of
        shape (numCandidates, 2, 3). The first axis represents possible
        partition configurations of the layer. Each candidate consists of a
        2x3 matrix: [numCoresPerAxis, coreShape]. The keys of the dictionary
        are the number of cores associated with a particular array of
        candidates.
    :rtype: dict
    """

    layerShape = np.array(layerShape)
    order = list(np.arange(len(layerShape))) if order is None else list(order)
    invOrder = getInversePermutation(order)

    numCompartments = np.prod(layerShape)
    minNumCores = int(np.ceil(numCompartments / maxNumCompartments))
    candidates = {}
    for numCoresPerAxis in np.ndindex(*layerShape[order]):

        # Add 1 to get at least one core per axis. If order was permuted
        # earlier, revert now.
        numCoresPerAxis = 1 + np.array(numCoresPerAxis)[invOrder]

        # Check that candidate is using enough cores.
        numCores = np.prod(numCoresPerAxis)
        if numCores < minNumCores:
            continue

        # Get largest core shape by dividing the layer shape by the number of
        # partitions per axis. Round up to avoid getting an additional, small
        # residual core.
        coreShape = np.ceil(layerShape / numCoresPerAxis).astype(int)

        # Compute the number of cores per axis based on the core shape. Due to
        # rounding, this number may differ from the original one, in which case
        # we discard the candidate. Example:
        #   layerShape = [35, 31, 1]
        #   numCoresPerAxis = [1, 9, 1]
        #   coreShape = [35, 4, 1]
        #   numCoresPerAxis2 = [1, 8, 1] --> discards superfluous core.
        # The next valid numCoresPerAxis will be [1, 11, 1].
        numCoresPerAxis2 = np.ceil(layerShape / coreShape)
        if np.any(numCoresPerAxis != numCoresPerAxis2):
            continue

        # If needed, create new list in dict for candidates with this number of
        # cores.
        if numCores not in candidates:
            candidates[numCores] = []

        # Add valid candidate to list.
        candidates[numCores].append([numCoresPerAxis, coreShape])

    return candidates


def getS(n):
    """Return string 's' if ``n`` > 1 and '' otherwise.

    :param int n: Number.
    :return: String to append to a word to indicate if it is plural.
    :rtype: str
    """

    return 's' if n > 1 else ''


def importPlt():
    """Import matplotlib.pyplot after setting backend.

    Uses 'Agg' backend if no display is found. Otherwise default backend is
    kept.

    :return: Reference to matplotlib.pyplot
    """

    if 'DISPLAY' not in os.environ:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def getMemoryUsage():
    """Return the memory usage in MB."""

    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / 2 ** 20


def logMemTime(name):
    def logMemTimeDecorator(function):
        @wraps(function)
        def wrapper(self=None, *args, **kwargs):
            m0 = getMemoryUsage()
            t0 = time.time()

            out = function(self, *args, **kwargs)

            t1 = time.time()
            m1 = getMemoryUsage()

            self.logger.info("%s time: %s s, memory: %s MB",
                             name, t1 - t0, m1 - m0)

            return out

        return wrapper

    return logMemTimeDecorator


def extract(probes):
    """Extract data from probes.

    :param list probes: Probes.
    :return: Numpy array of shape (numTimesteps, numProbes).
    :rtype: np.ndarray
    """

    return np.stack([p.data for p in probes], 1)
