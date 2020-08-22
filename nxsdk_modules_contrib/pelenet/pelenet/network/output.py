import numpy as np
from scipy import sparse

"""
@desc: Get weights for synapses between reservoir and output neurons
"""
def drawOutputMaskAndWeights(self):

    # TODO
    # * Test speed with output neuron probes

    # Define some helper variables
    cs = self.p.partitioningClusterSize  # Size of output clusters

    # Define empty mask
    mask = np.zeros((self.p.reservoirExSize, self.p.numOutputNeurons)).astype(int)
    #mask = np.zeros((self.p.reservoirExSize, self.p.numOutClusters)).astype(int)

    # Get indices of network topology
    topologyIndices = np.arange(self.p.reservoirExSize).reshape((self.p.topologySize, self.p.topologySize))

    # Get indices of shifted network topology
    topologyIndicesRolled = np.roll(topologyIndices, int(cs/2), axis=(0,1))

    k = 0  # k counts number of output cluster
    for i in range(self.p.numOutDimSize):
        # Define from and to variables for index i
        ifr, ito = i*cs, (i+1)*cs
        for j in range(self.p.numOutDimSize):
            # Define from and to variables for index j
            jfr, jto = j*cs, (j+1)*cs
            # Get topology indices and set connect neurons between output cluster and output neuron
            mask[topologyIndices[ifr:ito,jfr:jto], k] = 1
            # Overlapping output neurons
            mask[topologyIndicesRolled[ifr:ito,jfr:jto], k+1] = 1
            # Increase by 2
            k += 2
            #k += 1
    
    # Store mask
    self.outputMask = sparse.csr_matrix(mask.T)

    # Define weights
    self.outputWeights = self.outputMask * self.p.outputWeightValue


