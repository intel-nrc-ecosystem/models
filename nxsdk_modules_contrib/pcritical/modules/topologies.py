"""Converted from pcritical.modules.topologies to support python v3.5"""

import logging
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


_logger = logging.getLogger(__name__)


class SmallWorldTopology(nx.DiGraph):
    """Create a small-world type topology by creating a i1*i2*i3 cube of neurons separated by distance
    neuron_spacing in a 3d shape of j1*j2*j3 cubes distanced by minicolumn_spacing. i1, i2, i3 is the
    minicolumn_shape while j1, j2, j3 is the macrocolumn_shape. Connectivity is distance based with prob-
    ability of p_max * e^(-dist / intracolumnar_sparseness).
    """

    class Configuration:
        def __init__(self, **nargs):
            self.minicolumn_shape = (2, 2, 2)
            self.macrocolumn_shape = (4, 4, 4)
            self.neuron_spacing = 10.0
            self.minicolumn_spacing = 100.0
            self.p_max = 0.056
            self.intracolumnar_sparseness = 3 * 125

            # Construct the topology using sparse matrices of size mem_available
            self.sparse_init = False
            self.mem_available = 8 * 1024

            self.init_weights = (
                True  # If true, init the weights with following parameters:
            )
            self.inhibitory_prob = 0.2  # Ratio of inhibitory neurons [0, 1]
            self.inhibitory_init_weight_range = (0.01, 0.2)
            self.excitatory_init_weight_range = (0.1, 0.7)

            self.spectral_radius_norm = False

            self.__dict__.update(nargs)

    def __init__(self, configs):
        if type(configs) == nx.DiGraph:  # Assume we're creating a copy
            super().__init__(configs)
            return
        elif type(configs) == dict:
            configs = SmallWorldTopology.Configuration(**configs)

        super().__init__()
        self.__dict__.update(configs.__dict__)

        assert (
            len(self.minicolumn_shape) == 3
        ), "Minicolumn shape must be of dimension 3 (3D)"
        assert (
            len(self.macrocolumn_shape) == 3
        ), "Macrocolumn shape must be of dimension 3 (3D)"

        # Initial neuron positions (all separated by neuron_spacing)
        i, j, k = np.multiply(self.macrocolumn_shape, self.minicolumn_shape)
        grid = np.mgrid[:i, :j, :k].reshape(3, -1)
        x, y, z = grid * self.neuron_spacing

        # Adding minicolumnSpacing (from random to small world topology)
        if self.minicolumn_spacing > 0:
            for d in range(3):  # For each dimension
                grid[d] //= self.minicolumn_shape[d]
            x += grid[0] * self.minicolumn_spacing
            y += grid[1] * self.minicolumn_spacing
            z += grid[2] * self.minicolumn_spacing

        positions = map(lambda p: {"position": p}, zip(x, y, z))
        self.add_nodes_from(zip(range(len(x)), positions))

        # Distance-based random connectivity
        positions = np.stack(np.asarray(self.nodes.data("position"))[:, 1])

        distances = cdist(positions, positions, "euclidean")
        probabilities = self.p_max * np.exp(-distances / self.intracolumnar_sparseness)
        np.fill_diagonal(probabilities, 0)  # Avoid self-connections
        rand_matrix = np.random.random(probabilities.shape)
        i, j = np.nonzero(rand_matrix < probabilities)
        self.add_edges_from(zip(i, j))

        n_neurons = self.number_of_nodes()
        self.inhibitory_neurons = set(
            np.random.permutation(n_neurons)[: int(n_neurons * self.inhibitory_prob)]
        )

        for u, v in self.edges:
            if u in self.inhibitory_neurons:
                self.edges[u, v]["weight"] = -np.random.uniform(
                    *self.inhibitory_init_weight_range
                )
            else:
                self.edges[u, v]["weight"] = np.random.uniform(
                    *self.excitatory_init_weight_range
                )

        if self.spectral_radius_norm:
            spectral_radius = lambda matrix: np.max(np.abs(np.linalg.eigvals(matrix)))
            adj = nx.adjacency_matrix(self, weight="weight").todense()
            scale = 1.0 / spectral_radius(np.abs(adj))

            for i, (u, v) in enumerate(self.edges):
                self.edges[u, v]["weight"] = self.edges[u, v]["weight"] * scale
