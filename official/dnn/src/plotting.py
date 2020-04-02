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

"""Various plotting utilities for DNNs on Loihi."""

import os
from typing import TYPE_CHECKING

import numpy as np

from nxsdk_modules.dnn.src.utils import normalizeImageDim, importPlt
from matplotlib.ticker import IndexLocator, AutoMinorLocator

plt = importPlt()

if TYPE_CHECKING:
    from nxsdk_modules.dnn.src.data_structures import Layer


plotproperties = {
    'font.size': 12,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'x-large',
    'xtick.labelsize': 'x-large',
    'xtick.major.size': 7,
    'xtick.minor.size': 5,
    'ytick.labelsize': 'x-large',
    'ytick.major.size': 7,
    'ytick.minor.size': 5,
    'legend.fontsize': 'x-large',
    'lines.markersize': 6,
    'figure.figsize': (7, 3),
    'savefig.format': 'pdf',
    'savefig.dpi': 300}

# matplotlib.rcParams.update(plotproperties)

COLORS = ['firebrick', 'forestgreen', 'gold', 'skyblue', 'maroon',
          'darkblue', 'grey']


def plotMat(mat, fontSize=0, backgroundVal=0, showColorBar=False, title=None,
            savepath=None):
    """Plot a matrix showing numeric values for each entry.

    :param np.ndarray mat: Matrix to plot.
    :param int fontSize: Fontsize for values in ``mat``.
    :param int backgroundVal: Entries with this value in ``mat`` are
        considered background.
    :param bool showColorBar: Whether to display the color bar.
    :param str title: Figure title.
    :param str savepath: If given, where to save figure.
    """

    fig, ax = plt.subplots()

    ax.matshow(mat, cmap='Blues')
    if fontSize > 0:
        assert mat.ndim == 2
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if val != backgroundVal:
                    ax.text(j, i, str(val), va='center', ha='center',
                            fontdict={'fontsize': fontSize})

    if title is not None:
        ax.set_title(title)

    plt.axis('equal')
    plt.axis('tight')

    if showColorBar:
        fig.colorbar()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')


def plot_sweep_results(path, sweep, xlabel, shape=None):
    """Plot the results of a sweep across several network configurations.

    :param str path: Where to load the data from.
    :param str sweep: Name of parameter that was varied.
    :param str xlabel: Label for x-axis of plots.
    :param tuple | list | np.ndarray shape: Input shape, only used for labels.
    """

    data_path = os.path.join(path, 'partitions')

    plot_path = os.path.join(path, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    data_path_sweep = os.path.join(data_path, sweep)
    plot_path_sweep = os.path.join(plot_path, sweep)
    if not os.path.exists(plot_path_sweep):
        os.makedirs(plot_path_sweep)

    core_counts = {}
    core_occupancy = {}
    for scale in os.listdir(data_path_sweep):
        core_counts[scale] = 0
        core_occupancy[scale] = []
        for layer_file in os.listdir(os.path.join(data_path_sweep, scale)):
            data = np.load(os.path.join(data_path_sweep, scale, layer_file))
            core_counts[scale] += data['numCores']
            core_occupancy[scale] += list(data['coreOccupancy'].flatten())

    scales = [eval(k) for k in core_counts.keys()]

    if shape is not None:
        xtick_labels = ['{}'.format(int(scale * shape[0]))
                        for scale in scales]
        scales = np.array(scales) ** 2
        xtick_locs = scales
        # Scale factor reduces network size quadratically
        xtick_kwargs = {'fontsize': 11}
    else:
        xtick_locs = scales
        xtick_labels = scales
        xtick_kwargs = {}

    plt.figure()
    plt.scatter(scales, core_counts.values(), label='measured scaling')
    num_cores_per_chip = 128
    form_factors = {'Loihi': 1, 'KapohoBay': 2, 'WolfMountain': 4,
                    'Nahuku8': 8, 'Nahuku32': 32}
    for label, form_factor in form_factors.items():
        plt.hlines(form_factor * num_cores_per_chip, -0.01, 1.05, colors='k',
                   linewidth=0.5)
        plt.text(0.8, (0.6 * form_factor) * num_cores_per_chip, label,
                 fontsize=11)
    # plt.plot([0, 1], [0, core_counts['1']], color='orange',
    #          label='linear scaling')
    # plt.title("Cost vs network size for ResNet")
    plt.xlabel(xlabel)
    plt.ylabel("Num cores")
    plt.yscale('log', basey=2)
    plt.xticks(xtick_locs, xtick_labels, **xtick_kwargs)
    plt.yticks(2**np.arange(7) * num_cores_per_chip,
               2**np.arange(7) * num_cores_per_chip)
    # plt.grid()
    # plt.legend()
    plt.xlim(-0.01, 1.05)
    plt.ylim(32, None)
    plt.savefig(os.path.join(plot_path_sweep, 'num_cores'),
                bbox_inches='tight')

    plt.figure()
    plt.boxplot([100 * np.array(data) / 1024
                 for data in core_occupancy.values()],
                positions=scales, widths=0.02, meanline=True,
                manage_xticks=False, showmeans=True, showcaps=False)
    plt.xlabel(xlabel)
    plt.ylabel("Core occupancy [%]")
    plt.xticks(xtick_locs, xtick_labels, **xtick_kwargs)
    plt.xlim(0, None)
    plt.ylim(-5, 105)
    plt.savefig(os.path.join(plot_path_sweep, 'core_occupancy'),
                bbox_inches='tight')

    plt.figure()
    for scale, data in zip(scales, core_occupancy.values()):
        plt.scatter(np.repeat(scale, len(data)),
                    100 * np.array(data) / 1024, color='steelblue')
    plt.xlabel(xlabel)
    plt.ylabel("Core occupancy [%]")
    plt.xticks(xtick_locs, xtick_labels, **xtick_kwargs)
    plt.grid()
    plt.xlim(0, None)
    plt.ylim(0, 100)
    plt.savefig(os.path.join(plot_path_sweep, 'core_occupancy2'),
                bbox_inches='tight')


def plot_core_utilization(layers, path):
    """Plot how efficiently the resources of cores are used.

    For each layer, draw the distribution of each resource type as a box plot.
    Resource types include compartments, synaptic memory, input and output axon
    config entries.

    :param list[Layer] layers: List of partitioned layers.
    :param str path: Where to load the data from.
    """

    compartments = []
    input_axons = []
    output_axons = []
    synapses = []
    for layer in layers:
        compartments.append([])
        input_axons.append([])
        output_axons.append([])
        synapses.append([])
        for partition in layer.partitions:
            compartments[-1].append(len(partition.compartmentGroup.cxIds)
                                    * 100 / 1024)
            input_axons[-1].append(partition.inputAxonCost * 100)
            output_axons[-1].append(partition.outputAxonCost * 100)
            synapses[-1].append(partition.synapseCost * 100)

    labels = ['cx', 'inAx', 'outAx', 'syn']
    num_cost_terms = len(labels)
    num_layers = len(layers)
    xticks = np.arange(num_layers)
    jitter = (np.arange(num_cost_terms) - (num_cost_terms - 1) / 2) / 10
    # colors = plt.cm.get_cmap('Set1', num_cost_terms).colors
    colors = COLORS

    fig, ax = plt.subplots()

    use_boxplot = True

    for i, data in enumerate([compartments, input_axons,
                              output_axons, synapses]):
        color = colors[i]
        if use_boxplot:
            bp = ax.boxplot(data, positions=xticks+jitter[i], widths=0.09,
                            meanline=True, manage_xticks=False, showmeans=True,
                            showcaps=False, patch_artist=True,
                            medianprops={'linewidth': 10, 'linestyle': ':'},
                            meanprops={'linewidth': 10},
                            flierprops={'markerfacecolor': color,
                                        'marker': '.',
                                        'markeredgecolor': color})
            # Color
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps',
                            'means']:
                plt.setp(bp[element], color=color)
            for patch in bp['boxes']:
                patch.set(facecolor='white')
            # Legend
            ax.text(num_layers - 0.4, 100 - i * 5, labels[i], color=color)

        else:
            label = labels[i]
            for j, column in enumerate(data):
                plt.scatter(np.repeat(j + jitter[i], len(column)), column,
                            color=color, label=label)
                label = None
            ax.legend()

    ax.set_xlabel("Layer number")
    ax.set_ylabel("Core utilization [%]")
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks(np.arange(0, num_layers, num_layers // 5 + 1))
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_ylim(-5, 105)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(b=True, axis='x', which='minor')
    fig.savefig(os.path.join(path, 'core_utilization'), bbox_inches='tight')


def plot_multiplicity(m, path, name):
    """Plot multiplicityMap.

    :param np.array m: multiplicityMap.
    :param str path: Where to save figure.
    :param str name: Name of partition.
    """

    m = normalizeImageDim(m)

    fig, ax = plt.subplots()
    im = ax.imshow(m, cmap='Blues', vmin=0)
    vals = np.unique(m)
    fig.colorbar(im, ticks=[0] + list(vals[::(len(vals) // 10) + 1]),
                 fraction=0.02, pad=0.04)
    ax.set_xticks(np.arange(0, m.shape[1], m.shape[1] // 5 + 1))
    ax.set_yticks(np.arange(0, m.shape[0], m.shape[0] // 5 + 1))
    ax.set_title('Multiplicity map of input to {}'.format(name))
    ax.tick_params(which='both', left=False, bottom=False)
    ax.xaxis.set_minor_locator(IndexLocator(1, 1))
    ax.yaxis.set_minor_locator(IndexLocator(1, 1))
    ax.grid(which='minor')
    fig.savefig(os.path.join(path, 'multiplicityMap_{}'.format(name)),
                bbox_inches='tight')


def plot_coreIdMap(m, path, name):
    """Plot coreIdMap.

    :param np.array m: coreIdMap.
    :param str path: Where to save figure.
    :param str name: Name of partition.
    """

    yy = normalizeImageDim(m)
    shape = yy.shape

    fig, ax = plt.subplots()
    im = ax.imshow(yy, cmap='Blues', vmin=0)
    vals = np.unique(yy)
    fig.colorbar(im, ticks=vals[::len(vals) // 10 + 1],
                 fraction=0.02, pad=0.04)
    ax.set_xticks(np.arange(0, shape[1], shape[1] // 5 + 1))
    ax.set_yticks(np.arange(0, shape[0], shape[0] // 5 + 1))
    ax.tick_params(which='both', left=False, bottom=False)
    ax.xaxis.set_minor_locator(IndexLocator(1, 1))
    ax.yaxis.set_minor_locator(IndexLocator(1, 1))
    ax.grid(which='minor')

    if m.ndim == 3:
        num_depth_partitions = len(np.unique(m[0, 0, :]))
        num_channels = m.shape[-1]
        s = '' if num_depth_partitions == 1 else 's'
        ss = '' if num_channels == 1 else 's'
        ax.set_title('Core ID map of layer {}\n({} partition{} along '
                     '{} channel{}.)'.format(name, num_depth_partitions, s,
                                             num_channels, ss))
    else:
        ax.set_title('Core ID map of layer {}'.format(name))

    fig.savefig(os.path.join(path, 'partition_{}'.format(name)),
                bbox_inches='tight')


def plot_core_occupancy(occ, path, name):
    """Plot number of compartments per core of a layer.

    :param np.ndarray occ: Core occupancy to plot. Shape is equal to the number
        of cores per axis.
    :param str path: Where to save figure.
    :param str name: Name of partition.
    """

    occ = normalizeImageDim(occ)

    fig, ax = plt.subplots()
    im = ax.imshow(occ, cmap='Blues', vmin=0, vmax=1024)
    vals = np.unique([0] + list(np.ravel(occ)))
    if 1024 - np.max(vals) > 100:
        vals = np.concatenate([vals, [1024]])
    fig.colorbar(im, ticks=vals[::(len(vals) // 10) + 1],
                 fraction=0.02, pad=0.04)
    ax.set_xticks(np.arange(0, occ.shape[1], occ.shape[1] // 10 + 1))
    ax.set_yticks(np.arange(0, occ.shape[0], occ.shape[0] // 10 + 1))
    ax.tick_params(which='both', left=False, bottom=False)
    ax.xaxis.set_minor_locator(IndexLocator(1, 1))
    ax.yaxis.set_minor_locator(IndexLocator(1, 1))
    ax.grid(which='minor')
    ax.set_title('Core occupancy of layer {}'.format(name))
    fig.savefig(os.path.join(path, 'coreOccupancy_{}'.format(name)),
                bbox_inches='tight')


def visualize_partitions(path):
    """Visualize the partition result of a layer.

    :param str path: Where to load the data from and save figures.
    """

    data_path = os.path.join(path, 'model_dumps', 'partitions')

    for layer_name in os.listdir(data_path):

        data = np.load(os.path.join(data_path, layer_name))

        name = data['id']

        y = data['multiplicityMap']
        if y.size:
            plot_multiplicity(y, path, name)

        plot_coreIdMap(data['coreIdMap'], path, name)

        plot_core_occupancy(data['coreOccupancy'], path, name)


def plot_cost_graph(path):
    """Visualize the cost of a number of layer partitions.

    :param str path: Where to load the data from.
    """

    filepath = os.path.join(path, 'candidate_costs.npz')
    if not os.path.exists(filepath):
        print("Plotting cost graph failed: Could not load {}.".format(
            filepath))
        return

    data = np.load(filepath)

    all_costs = data['all_costs']
    num_candidates, num_layers = all_costs.shape
    num_optimal_candidates = int(np.sqrt(num_candidates))
    # Candidates are already sorted by cost.
    optimal_costs = all_costs[:num_optimal_candidates]

    fig, ax = plt.subplots()

    # Plot the cost of each candidate of each layer as data points, together
    # with the mean and variance across the candidates of a layer.
    colormap = plt.cm.get_cmap('prism', num_candidates)
    colors = np.array([colormap(i) for i in np.repeat(np.arange(
        num_optimal_candidates), num_optimal_candidates)])
    for layer_num, partition_costs in enumerate(all_costs.T):
        ax.errorbar(layer_num, np.mean(partition_costs),
                    2 * np.sqrt(np.var(partition_costs)),
                    fmt='_', color='black', alpha=0.4)
        ax.scatter(np.repeat(layer_num, num_candidates),
                   np.array(partition_costs), color=colors)

    # Plot optimal partitioning costs across layers as lineplot.
    colormap = plt.cm.get_cmap('gray', num_optimal_candidates * 10)
    for i, costs in enumerate(optimal_costs):
        ax.plot(costs, '-', color=colormap(i * 10))

    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(0, num_layers, num_layers // 5 + 1))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Partition cost")
    fig.savefig(os.path.join(path, 'costGraph'), bbox_inches='tight')


def plot_cost_terms(path, hide_post=True):
    """Visualize the cost terms of a layer partition.

    :param str path: Where to load the data from.
    :param bool hide_post: If ``True`` (default), remove the cost term
        corresponding to the postLayerPartition.
    """

    data = dict(np.load(os.path.join(path, 'cost_terms.npz')))

    if hide_post:
        data.pop('postLayerCost')

    cost_terms = sorted(data.keys())

    fig, ax = plt.subplots()

    values = np.array([data[k] for k in cost_terms]).transpose()

    num_layers, num_cost_terms = values.shape

    bottoms = np.concatenate([np.zeros((num_layers, 1)),
                              np.cumsum(values, 1)], 1)

    layer_idxs = np.arange(num_layers)

    # colors = plt.cm.get_cmap('Set1', num_cost_terms).colors
    colors = COLORS
    for i in range(num_cost_terms):
        ax.bar(layer_idxs, values[:, i], color=colors[i], label=cost_terms[i],
               bottom=bottoms[:, i])

    ax.set_xticks(np.arange(0, num_layers, num_layers // 5 + 1))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Partition cost")
    ax.legend()
    fig.savefig(os.path.join(path, 'costTerms'), bbox_inches='tight')


def plot_cx_syn(path, model):
    sizes = []
    parameters = []
    for layer in model.layers:
        sizes.append(np.prod(layer.output_shape[1:]))
        if len(layer.weights):
            parameters.append(np.prod(layer.weights[0].shape).value)
        else:
            parameters.append(0)

    num_layers = len(model.layers)
    sizes = np.array(sizes) / np.max(sizes)
    parameters = np.array(parameters) / np.max(parameters)

    fig, ax = plt.subplots()
    ax.plot(sizes, color=COLORS[0], label='cx')
    ax.plot(parameters, color=COLORS[3], label='params')
    ax.set_xticks(np.arange(0, num_layers, num_layers // 5 + 1))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized count")
    ax.legend()
    fig.savefig(os.path.join(path, 'cx_syn'), bbox_inches='tight')


def plot_exclusion_criteria_hit_count(path):
    """Show how often and for what reason a partition candidate was rejected.

    :param str path: Where to load the data from and save the figures.
    """

    data = np.load(os.path.join(path, 'exclusion_criteria_hit_count.npz'))
    data = dict(data)

    criteria = sorted(data.keys())

    hit_counts = np.array([data[k] for k in criteria]).transpose()

    num_layers, num_criteria = hit_counts.shape

    bottoms = np.concatenate([np.zeros((num_layers, 1)),
                              np.cumsum(hit_counts, 1)], 1)

    layer_idxs = np.arange(num_layers)

    # colors = plt.cm.get_cmap('Set1', num_criteria).colors
    colors = COLORS
    
    fig, ax = plt.subplots()

    for i in range(num_criteria):
        ax.bar(layer_idxs, hit_counts[:, i], color=colors[i],
               label=criteria[i], bottom=bottoms[:, i])

    ax.set_xticks(np.arange(0, num_layers, num_layers // 5 + 1))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Exclusion count")
    ax.set_ylim(0, None)
    ax.legend()
    fig.savefig(os.path.join(path, 'exclusion_criteria_hit_count'),
                bbox_inches='tight')


def plot_layer_partition(preLayer, path):
    """Visualize the partition configuration of a layer.

    :param Layer preLayer: Layer to visualize.
    :param str path: Where to save figure.
    """

    layer = preLayer.postLayer

    if layer is None:
        return

    inputSize = np.asscalar(np.prod(preLayer.coreIdMap.shape))
    outputSize = np.asscalar(np.prod(layer.coreIdMap.shape))

    # Add a column for cxIds.
    numColsDestCxIds = 1
    xShift = numColsDestCxIds
    # Add rows for cxIds, multiplicity, and axonGroups, and insert an empty row
    # between each set of rows.
    numRowsOutAxons = preLayer.numCores + 1
    numRowsInAxons = layer.numCores + 1
    numRowsMultiplicity = 2
    numRowsSrcCxIds = 1
    yShift = numRowsOutAxons + numRowsInAxons + numRowsMultiplicity + \
        numRowsSrcCxIds

    shape = np.array([outputSize + yShift, inputSize + xShift])

    numPxPerUnit = 3
    figsize = numPxPerUnit * np.array([1, shape[0] / shape[1]])

    partitionColors = plt.cm.get_cmap(
        'Pastel1', layer.numCores + preLayer.numCores).colors

    synGroupColors = plt.cm.get_cmap('gist_ncar')

    fig = plt.figure(figsize=figsize)
    ax01 = plt.subplot2grid(shape, (0, xShift),
                            colspan=inputSize, rowspan=yShift)
    ax10 = plt.subplot2grid(shape, (yShift, 0),
                            colspan=xShift, rowspan=outputSize)
    ax11 = plt.subplot2grid(shape, (yShift, xShift),
                            colspan=inputSize, rowspan=outputSize)

    markersize = (numPxPerUnit * 72 / shape[1] / 4) ** 2
    fontsize = min(max(1, 100 / shape[1]), 20)
    linewidth = 0.1

    ax01.set_xlim(-0.5, inputSize)
    ax01.set_ylim(-0.5, yShift)
    ax01.axis('off')
    ax01.invert_yaxis()

    # Place horizontal lines between each distinct set of rows.
    ax01.axhline(numRowsOutAxons - 1,
                 linestyle='-', linewidth=linewidth*5, color='k', alpha=0.1)
    ax01.axhline(numRowsOutAxons + numRowsInAxons - 1,
                 linestyle='-', linewidth=linewidth*5, color='k', alpha=0.1)
    ax01.axhline(numRowsOutAxons + numRowsInAxons + numRowsMultiplicity - 1,
                 linestyle='-', linewidth=linewidth*5, color='k', alpha=0.1)

    # +++++++++ SrcCxIds ++++++++++ #

    yShiftSrcIds = numRowsOutAxons + numRowsInAxons + numRowsMultiplicity
    coreIds = np.zeros(inputSize, int)
    for relCoreId, prePartition in enumerate(preLayer.partitions):
        relToAbsDestCxIdxMap = \
            prePartition.compartmentGroup.relToAbsDestCxIdxMap
        # Increase relCoreId of preLayer by number of cores of current layer
        # so these cores get a different color.
        coreIds[relToAbsDestCxIdxMap] = relCoreId + layer.numCores

    # Draw srcCxIds numbers.
    srcCxIds = np.arange(inputSize)
    for j, srcCxId in enumerate(srcCxIds):
        ax01.text(j, yShiftSrcIds, str(srcCxId),
                  va='center', ha='center', fontsize=fontsize)

    # Draw colors.
    x = srcCxIds
    y = yShiftSrcIds * np.ones(inputSize)
    colors = partitionColors[coreIds]
    ax01.scatter(x, y, c=colors, marker='s', s=markersize)

    # +++++++++ InputAxonGroups ++++++++++ #

    yShiftInAxons = numRowsOutAxons

    srcIdOffsets = {}
    matrix = -np.ones((layer.numCores, inputSize), int)
    for relCoreId, partition in enumerate(layer.partitions):
        for inputAxonGroup in partition.inputAxonGroups:
            matrix[relCoreId, inputAxonGroup.srcNodeIds] = relCoreId
            srcIds = inputAxonGroup.srcNodeIds
            srcIdOffsets[inputAxonGroup.id] = srcIds[0]

            # Draw cxId numbers.
            for relSrcId, srcId in enumerate(srcIds):
                ax01.text(srcId, yShiftInAxons + relCoreId, str(relSrcId),
                          va='center', ha='center', fontsize=fontsize)

            # Place a vertical dash after each inputAxonGroup.
            ax01.axvline(srcIds[-1] + 0.5,
                         1 - (yShiftInAxons + relCoreId + 1) / yShift,
                         1 - (yShiftInAxons + relCoreId) / yShift,
                         color='k', linewidth=linewidth)

    # Draw colors.
    y, x = np.nonzero(matrix >= 0)
    coreIds = matrix[y, x]
    colors = partitionColors[coreIds]
    ax01.scatter(x, yShiftInAxons + y, c=colors, marker='s', s=markersize)

    # +++++++++ OutputAxonGroups ++++++++++ #

    yShiftOutAxons = 0
    matrix = -np.ones((preLayer.numCores, inputSize), int)

    for relCoreId, partition in enumerate(preLayer.partitions):
        for outputAxonGroup in partition.outputAxonGroups:
            cxIds = outputAxonGroup.cxIds
            relSrcIds = outputAxonGroup.relSrcIds
            srcIdOffset = srcIdOffsets[outputAxonGroup.inAxGrpId]
            matrix[relCoreId, srcIdOffset + relSrcIds] = \
                relCoreId + layer.numCores

            # Draw cxId numbers.
            for relSrcId, cxId in zip(relSrcIds, cxIds):
                ax01.text(srcIdOffset + relSrcId, yShiftOutAxons + relCoreId,
                          str(cxId), va='center', ha='center',
                          fontsize=fontsize)

            # Place a vertical dash after each inputAxonGroup.
            ax01.axvline(srcIdOffset + relSrcIds[-1] + 0.5,
                         1 - (yShiftOutAxons + relCoreId + 1) / yShift,
                         1 - (yShiftOutAxons + relCoreId) / yShift,
                         color='k', linewidth=linewidth)

    # Draw colors.
    y, x = np.nonzero(matrix >= 0)
    coreIds = matrix[y, x]
    colors = partitionColors[coreIds]
    ax01.scatter(x, yShiftOutAxons + y, c=colors, marker='s', s=markersize)

    # +++++++++ Multiplicities ++++++++++ #

    yShiftMult = numRowsOutAxons + numRowsInAxons

    multiplicities = np.zeros(inputSize, int)
    for relCoreId, partition in enumerate(layer.partitions):
        for inputAxonGroup in partition.inputAxonGroups:
            multiplicities[inputAxonGroup.srcNodeIds] = \
                inputAxonGroup.multiplicity

    # Draw colors.
    x = np.arange(inputSize)
    y = yShiftMult * np.ones(inputSize)
    colors = multiplicities
    ax01.scatter(x, y, c=colors, cmap='coolwarm', marker='s', s=markersize)

    # Draw multiplicity numbers.
    for j, m in enumerate(multiplicities):
        ax01.text(j, yShiftMult, str(m), va='center', ha='center',
                  fontsize=fontsize)

    fig.subplots_adjust(wspace=0, hspace=0)

    # +++++++++ kernelIds ++++++++++ #
    # +++++++++ DestCxIds ++++++++++ #

    figureFormat = 'pdf'

    # No interleaving; cores separated; kernelId color code.
    drawKernelIds10(ax11, layer, inputSize, outputSize, fontsize, markersize)
    drawDestCxIds0(ax10, layer, outputSize, fontsize, markersize,
                   partitionColors, xShift)
    fig.savefig(os.path.join(path, 'partition_{}_{}.pdf'.format(
        0, layer.id)), bbox_inches='tight', format=figureFormat)

    # No interleaving; cores concatenated; kernelId color code.
    drawKernelIds2(ax11, layer, inputSize, outputSize, fontsize, markersize)
    drawDestCxIds1(ax10, layer, outputSize, fontsize, markersize,
                   partitionColors, xShift)
    fig.savefig(os.path.join(path, 'partition_{}_{}.pdf'.format(
        1, layer.id)), bbox_inches='tight', format=figureFormat)

    # Interleaving; cores concatenated; kernelId color mode.
    drawKernelIds11(ax11, layer, inputSize, fontsize, markersize, linewidth,
                    partitionColors)
    drawDestCxIds2(ax10, layer, fontsize, markersize, partitionColors, xShift)
    fig.savefig(os.path.join(path, 'partition_{}_{}.pdf'.format(
        2, layer.id)), bbox_inches='tight', format=figureFormat)

    # Interleaving; cores concatenated; synGroup color mode.
    drawKernelIds01(ax11, layer, inputSize, fontsize, markersize,
                    synGroupColors, linewidth, partitionColors)
    drawDestCxIds2(ax10, layer, fontsize, markersize, partitionColors, xShift)
    fig.savefig(os.path.join(path, 'partition_{}_{}.pdf'.format(
        3, layer.id)), bbox_inches='tight', format=figureFormat)

    # No interleaving; cores separated; synGroup color mode.
    drawKernelIds00(ax11, layer, inputSize, outputSize, fontsize, markersize,
                    synGroupColors)
    drawDestCxIds0(ax10, layer, outputSize, fontsize, markersize,
                   partitionColors, xShift)
    fig.savefig(os.path.join(path, 'partition_{}_{}.pdf'.format(
        4, layer.id)), bbox_inches='tight', format=figureFormat)


def drawDestCxIds0(ax, layer, outputSize, fontsize, markersize,
                   partitionColors, xShift):
    """Draw the destination compartment ids into an existing figure.

    The displayed ids are not interleaved and are not grouped into partitions.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int outputSize: Number of output neurons.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    :param partitionColors: Color scheme to distinguish partitions.
    :param int xShift: Horizontal offset.
    """

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, xShift)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    xShiftDestIds = 0
    cxIdsFull = -np.ones(outputSize, int)
    coreIdsFull = -np.ones(outputSize, int)
    for partition in layer.partitions:
        cxIdsInterleaved = -np.ones(partition.sizeInterleaved, int)
        coreIdsInterleaved = -np.ones(partition.sizeInterleaved, int)
        cxIds = partition.compartmentGroup.cxIds
        cxIdsInterleaved[cxIds] = np.arange(len(cxIds))
        coreIdsInterleaved[cxIds] = partition.id

        # Undo interleaving.
        relToAbsDestCxIdxMap = partition.compartmentGroup.relToAbsDestCxIdxMap

        cxIdsFull[relToAbsDestCxIdxMap] = cxIdsInterleaved[cxIds]

        coreIdsFull[relToAbsDestCxIdxMap] = coreIdsInterleaved[cxIds]

    # Draw colors.
    matrix = coreIdsFull
    y = np.flatnonzero(matrix >= 0)
    x = xShiftDestIds * np.ones(len(y))
    coreIds = matrix[y]
    colors = partitionColors[coreIds]
    ax.scatter(x, y, c=colors, marker='s', s=markersize)

    # Draw cxIds.
    for j, cxId in enumerate(cxIdsFull):
        ax.text(xShiftDestIds, j, str(cxId),
                va='center', ha='center', fontsize=fontsize)


def drawDestCxIds1(ax, layer, outputSize, fontsize, markersize,
                   partitionColors, xShift):
    """Draw the destination compartment ids into an existing figure.

    The displayed ids are not interleaved and are grouped into partitions.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int outputSize: Number of output neurons.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    :param partitionColors: Color scheme to distinguish partitions.
    :param int xShift: Horizontal offset.
    """

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, xShift)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    xShiftDestIds = 0
    coreOffset = 0
    for partition in layer.partitions:
        cxIdsInterleaved = -np.ones(partition.sizeInterleaved, int)
        coreIdsInterleaved = -np.ones(partition.sizeInterleaved, int)
        cxIds = partition.compartmentGroup.cxIds
        cxIdsInterleaved[cxIds] = np.arange(len(cxIds))
        coreIdsInterleaved[cxIds] = partition.id

        # Undo interleaving.
        cxIdsCore = cxIdsInterleaved[cxIds]
        coreIdsCore = coreIdsInterleaved[cxIds]

        # Draw colors.
        matrix = coreIdsCore
        y = np.flatnonzero(matrix >= 0)
        x = xShiftDestIds * np.ones(len(y))
        coreIds = matrix[y]
        colors = partitionColors[coreIds]
        ax.scatter(x, coreOffset + y, c=colors, marker='s', s=markersize)

        # Draw cxIds.
        for j, cxId in enumerate(cxIdsCore):
            ax.text(xShiftDestIds, coreOffset + j, str(cxId),
                    va='center', ha='center', fontsize=fontsize)

        coreOffset += np.max(y) + 1


def drawDestCxIds2(ax, layer, fontsize, markersize, partitionColors, xShift):
    """Draw the destination compartment ids into an existing figure.

    The displayed ids are interleaved and are grouped into partitions.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    :param list partitionColors: Color scheme to distinguish partitions.
    :param int xShift: Horizontal offset.
    """

    outputSize = np.asscalar(
        np.sum([partition.sizeInterleaved for partition in layer.partitions]))

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, xShift)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    xShiftDestIds = 0
    coreOffset = 0
    for partition in layer.partitions:
        cxIds = partition.compartmentGroup.cxIds

        # Draw destCxIds numbers.
        for cxId in cxIds:
            ax.text(xShiftDestIds, coreOffset + cxId, str(cxId),
                    va='center', ha='center', fontsize=fontsize)

        # Draw colors.
        x = xShiftDestIds * np.ones(len(cxIds))
        colors = partitionColors[partition.id]
        ax.scatter(x, coreOffset + cxIds, c=colors, marker='s', s=markersize)

        coreOffset += partition.sizeInterleaved


def drawKernelIds10(ax, layer, inputSize, outputSize, fontsize, markersize):
    """Draw the kernel ids into an existing figure.

    The displayed ids are not interleaved and are not grouped into partitions.
    The ids are color-coded according to their value.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int inputSize: Number of input neurons.
    :param int outputSize: Number of output neurons.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    """

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, inputSize)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    kMapFull = np.zeros((outputSize, inputSize), int)
    for relCoreId, partition in enumerate(layer.partitions):
        kMapInterleaved = np.zeros((partition.sizeInterleaved, inputSize), int)
        for inputAxonGroup in partition.inputAxonGroups:
            for relSrcId, srcId in enumerate(inputAxonGroup.srcNodeIds):
                for synEntry in inputAxonGroup.synGroup.synEntries[relSrcId]:
                    destIds = inputAxonGroup.cxBase + synEntry.getCxIds()
                    kMapInterleaved[destIds, srcId] = synEntry.kernelIds

        # Undo interleaving.
        kMapCore = kMapInterleaved[partition.compartmentGroup.cxIds]
        kMapFull[partition.compartmentGroup.relToAbsDestCxIdxMap] = kMapCore

    # Draw colors.
    matrix = kMapFull
    y, x = np.nonzero(matrix)
    colors = matrix[y, x]
    cmap = 'Blues'
    ax.scatter(x, y, c=colors, cmap=cmap, marker='s', s=markersize)

    # Draw kernelIds.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, str(val),
                        va='center', ha='center', fontsize=fontsize)


def drawKernelIds2(ax, layer, inputSize, outputSize, fontsize, markersize):
    """Draw the kernel ids into an existing figure.

    The displayed ids are not interleaved and are grouped into partitions.
    The ids are color-coded according to their value.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int inputSize: Number of input neurons.
    :param int outputSize: Number of output neurons.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    """

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, inputSize)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    coreOffset = 0
    for relCoreId, partition in enumerate(layer.partitions):
        kMapInterleaved = np.zeros((partition.sizeInterleaved, inputSize), int)
        for inputAxonGroup in partition.inputAxonGroups:
            for relSrcId, srcId in enumerate(inputAxonGroup.srcNodeIds):
                for synEntry in inputAxonGroup.synGroup.synEntries[relSrcId]:
                    destIds = inputAxonGroup.cxBase + synEntry.getCxIds()
                    kMapInterleaved[destIds, srcId] = synEntry.kernelIds

        # Undo interleaving.
        kMapCore = kMapInterleaved[partition.compartmentGroup.cxIds]

        # Draw colors.
        matrix = kMapCore
        y, x = np.nonzero(matrix)
        colors = matrix[y, x]
        cmap = 'Blues'
        ax.scatter(x, coreOffset + y, c=colors, cmap=cmap,
                   marker='s', s=markersize)

        # Draw kernelIds.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, coreOffset + i, str(val),
                            va='center', ha='center', fontsize=fontsize)

        coreOffset += np.max(y) + 1


def drawKernelIds11(ax, layer, inputSize, fontsize, markersize, linewidth,
                    partitionColors):
    """Draw the kernel ids into an existing figure.

    The displayed ids are interleaved and are grouped into partitions.
    The ids are color-coded according to their value.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int inputSize: Number of input neurons.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    :param float linewidth: Width of box drawn around ids with nonzero
        cIdxOffset.
    :param list partitionColors: Color scheme to distinguish partitions.
    """

    outputSize = np.asscalar(
        np.sum([partition.sizeInterleaved for partition in layer.partitions]))

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, inputSize)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    coreOffset = 0
    for relCoreId, partition in enumerate(layer.partitions):
        kMapInterleaved = np.zeros((partition.sizeInterleaved, inputSize), int)
        cIdxOffsets = np.zeros((partition.sizeInterleaved, inputSize), int)
        for inputAxonGroup in partition.inputAxonGroups:
            cxBase = inputAxonGroup.cxBase
            srcIds = inputAxonGroup.srcNodeIds
            for relSrcId, srcId in enumerate(srcIds):
                for synEntry in inputAxonGroup.synGroup.synEntries[relSrcId]:
                    destIds = cxBase + synEntry.getCxIds()
                    kMapInterleaved[destIds, srcId] = synEntry.kernelIds
                    cIdxOffsets[destIds, srcId] = \
                        synEntry.synFmt.cIdxOffset

            # cxBase
            ax.axhline(cxBase - 0.4 + relCoreId / layer.numCores / 10,
                       (srcIds[0]) / inputSize, (srcIds[-1] + 1) / inputSize,
                       linestyle='--', color=partitionColors[relCoreId],
                       linewidth=linewidth * 2)

        # Draw colors.
        y, x = np.nonzero(kMapInterleaved)
        colors = kMapInterleaved[y, x]
        cmap = 'Blues'
        ax.scatter(x, coreOffset + y, c=colors, cmap=cmap, marker='s',
                   s=markersize)

        # Draw edges for nonzero cIdxOffsets.
        y, x = np.nonzero(cIdxOffsets)
        ax.scatter(x, coreOffset + y, marker='s', s=markersize*3,
                   facecolors='none', edgecolors='k', linewidth=linewidth)

        # Draw kernelIds.
        for i in range(kMapInterleaved.shape[0]):
            for j in range(kMapInterleaved.shape[1]):
                val = kMapInterleaved[i, j]
                if val > 0:
                    ax.text(j, coreOffset + i, str(val),
                            va='center', ha='center', fontsize=fontsize)

        coreOffset += partition.sizeInterleaved


def drawKernelIds01(ax, layer, inputSize, fontsize, markersize, synGroupColors,
                    linewidth, partitionColors):
    """Draw the kernel ids into an existing figure.

    The displayed ids are interleaved and are grouped into partitions.
    The ids are color-coded according to the unique synapse group they belong
    to.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int inputSize: Number of input neurons.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    :param list synGroupColors: Color scheme to distinguish synapse groups.
    :param float linewidth: Width of box drawn around ids with nonzero
        cIdxOffset.
    :param list partitionColors: Color scheme to distinguish partitions.
    """

    outputSize = np.asscalar(
        np.sum([partition.sizeInterleaved for partition in layer.partitions]))

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, inputSize)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    coreOffset = 0
    for relCoreId, partition in enumerate(layer.partitions):
        shape = (partition.sizeInterleaved, inputSize)
        kMapInterleaved = np.zeros(shape, int)
        synGroupIdsInterleaved = -np.ones(shape, int)
        cIdxOffsets = np.zeros(shape, int)
        for inputAxonGroup in partition.inputAxonGroups:
            srcIds = inputAxonGroup.srcNodeIds
            cxBase = inputAxonGroup.cxBase
            for relSrcId, srcId in enumerate(srcIds):
                for synEntry in inputAxonGroup.synGroup.synEntries[relSrcId]:
                    destIds = cxBase + synEntry.getCxIds()
                    kMapInterleaved[destIds, srcId] = synEntry.kernelIds
                    cIdxOffsets[destIds, srcId] = \
                        synEntry.synFmt.cIdxOffset
                    synGroupIdsInterleaved[destIds, srcId] = \
                        inputAxonGroup.synGroup.id

            # cxBase
            ax.axhline(cxBase - 0.4 + relCoreId / layer.numCores / 10,
                       (srcIds[0]) / inputSize, (srcIds[-1] + 1) / inputSize,
                       linestyle='--', color=partitionColors[relCoreId],
                       linewidth=linewidth * 2)

        # Draw colors.
        y, x = np.nonzero(synGroupIdsInterleaved >= 0)
        colors = synGroupIdsInterleaved[y, x]
        cmap = synGroupColors
        ax.scatter(x, coreOffset + y, c=colors, cmap=cmap,
                   marker='s', s=markersize)

        # Draw edges for nonzero cIdxOffsets.
        y, x = np.nonzero(cIdxOffsets)
        ax.scatter(x, coreOffset + y, marker='s', s=markersize*3,
                   facecolors='none', edgecolors='k', linewidth=linewidth)

        # Draw kernelIds.
        for i in range(kMapInterleaved.shape[0]):
            for j in range(kMapInterleaved.shape[1]):
                val = kMapInterleaved[i, j]
                if val > 0:
                    ax.text(j, coreOffset + i, str(val),
                            va='center', ha='center', fontsize=fontsize)

        coreOffset += partition.sizeInterleaved


def drawKernelIds00(ax, layer, inputSize, outputSize, fontsize, markersize,
                    synGroupColors):
    """Draw the kernel ids into an existing figure.

    The displayed ids are not interleaved and are not grouped into partitions.
    The ids are color-coded according to the unique synapse group they belong
    to.

    :param plt.Axes ax: Matplotlib axes.
    :param Layer layer: The layer to visualize.
    :param int inputSize: Number of input neurons.
    :param int outputSize: Number of output neurons.
    :param int fontsize: Fontsize of numeric values in plot.
    :param int markersize: Size of the squares drawn at each id location.
    :param list synGroupColors: Color scheme to distinguish synapse groups.
    """

    ax.clear()
    ax.axis('off')
    ax.set_xlim(-0.5, inputSize)
    ax.set_ylim(-0.5, outputSize)
    ax.invert_yaxis()

    kMapFull = np.zeros((outputSize, inputSize), int)
    synGroupIdsFull = -np.ones((outputSize, inputSize), int)
    for relCoreId, partition in enumerate(layer.partitions):
        kMapInterleaved = np.zeros((partition.sizeInterleaved, inputSize), int)
        synGroupIdsInterleaved = -np.ones((partition.sizeInterleaved,
                                           inputSize), int)
        for inputAxonGroup in partition.inputAxonGroups:
            for relSrcId, srcId in enumerate(inputAxonGroup.srcNodeIds):
                for synEntry in inputAxonGroup.synGroup.synEntries[relSrcId]:
                    destIds = inputAxonGroup.cxBase + synEntry.getCxIds()
                    kMapInterleaved[destIds, srcId] = synEntry.kernelIds
                    synGroupIdsInterleaved[destIds, srcId] = \
                        inputAxonGroup.synGroup.id

        # Undo interleaving.
        permutedDestCxIdxs = partition.compartmentGroup.cxIds
        relToAbsDestCxIdxMap = partition.compartmentGroup.relToAbsDestCxIdxMap

        kMapCore = kMapInterleaved[permutedDestCxIdxs]
        kMapFull[relToAbsDestCxIdxMap] = kMapCore

        synGroupIdsCore = synGroupIdsInterleaved[permutedDestCxIdxs]
        synGroupIdsFull[relToAbsDestCxIdxMap] = synGroupIdsCore

    # Draw colors.
    y, x = np.nonzero(synGroupIdsFull >= 0)
    colors = synGroupIdsFull[y, x]
    cmap = synGroupColors
    ax.scatter(x, y, c=colors, cmap=cmap, marker='s', s=markersize)

    # Draw kernelIds.
    for i in range(kMapFull.shape[0]):
        for j in range(kMapFull.shape[1]):
            val = kMapFull[i, j]
            if val > 0:
                ax.text(j, i, str(val),
                        va='center', ha='center', fontsize=fontsize)
