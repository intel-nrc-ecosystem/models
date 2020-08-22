import numpy as np
from scipy import sparse

"""
Calculate spectral radius of whole weight matrix
"""
def getSpectralRadius(self, weights):
    # Stack top and bottom row of weight matrix horizontally
    top = sparse.hstack([weights.exex, weights.inex])
    bottom = sparse.hstack([weights.exin, weights.inin])

    # Stack vertically
    wgs = sparse.vstack([top, bottom])

    # Calculate and return rounded spectral radius
    maxeigval = np.absolute(sparse.linalg.eigs(wgs.asfptype() / 255., k=1, which='LM', return_eigenvectors=False)[0])
    return np.round(maxeigval*1000)/1000.

"""
Recombine weight matrix from excitatory probe chunks
"""
def recombineExWeightMatrix(self, initialExWeights, exWeightProbes):
    # Get shorthand for some variables
    init = initialExWeights
    nPerCore = self.p.neuronsPerCore
    # Calculate trained weight matrix from weight probes
    weightMatrix = []
    # Iterate over number of probes (connection chunks between cores)
    n, m = np.shape(exWeightProbes)
    for i in range(n):
        # Define from/to indices for indexing
        ifr, ito = i*nPerCore, (i+1)*nPerCore
        chunks = []
        for j in range(m):
            # Define from/to indices for indexing
            jfr, jto = j*nPerCore, (j+1)*nPerCore
            # Get number of synapses in current probe
            numSyn = np.shape(exWeightProbes[i][j])[0]
            # Iterate over number of synapses in current probe (connections from one core to another)
            data = []
            for k in range(numSyn):
                # Get weights data from probe index 0 and append to data array
                data.append(exWeightProbes[i][j][k][0].data[0])
            # Get chunk from initial matrix for defining sparse matrix of the current chunk (need indices and index pointer)
            ic = init[jfr:jto, ifr:ito]
            # Define sparse matrix, using initial weight matrix indices and index pointerm, as well as shape of chunk
            chunks.append(sparse.csr_matrix((data, ic.indices, ic.indptr), shape=np.shape(ic)))
        # Stack list of chunks together to column
        column = sparse.vstack(chunks)
        # Append column to weight matrix
        weightMatrix.append(column)

    # Stack list of columns together to the whole trained weight matrix
    return sparse.hstack(weightMatrix).tocsr()  # transform to csr, since stacking returns coo format

"""
@desc: Get mask of support weights for every cluster in the assembly
@return: Mask of the bottom-left area of the matrix
"""
def getSupportWeightsMask(self, exWeightMatrix):
    nCs = self.p.traceClusterSize
    nEx = self.p.reservoirExSize
    nC = self.p.traceClusters
    matrix = exWeightMatrix

    # Get areas in matrix
    left = matrix[:,:nC*nCs]  # left
    top = matrix[:nC*nCs,:]  # top
    bottom = matrix[nC*nCs:,:]  # bottom
    bottomLeft = matrix[nC*nCs:,:nC*nCs]  # bottom-left

    # Get single cluster colums in bottom-left area (candidates for support weights)
    cols = np.array([ bottomLeft[:,i*nCs:(i+1)*nCs] for i in range(nC)])

    # Calculate means for every column in bottom-left
    col_rowmeans = np.array([np.mean(cols[i,...], axis=1) for i in range(nC)])

    # Condition 1: Get only rows their mean is greater than total mean
    greaterMeanIndices = col_rowmeans > np.mean(bottomLeft)

    # Condition 2: Get for every row the column which has max value
    col_argmax = np.argmax(col_rowmeans, axis=0)
    maxRowIndices = np.array(col_argmax[:,None] == range(nC)).T

    # Get final mask in combining both conditions
    return np.logical_and(greaterMeanIndices, maxRowIndices)
