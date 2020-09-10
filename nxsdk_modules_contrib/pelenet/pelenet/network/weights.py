import numpy as np
from scipy import sparse

"""
@desc: Get random values from a distribution
"""
def drawWeights(size, distribution):
    weights = None

    if distribution == 'lognormal':
        hyp = 1.0
        m = np.log(0.2) + hyp #np.log(0.12) + hyp  # Default: np.log(0.2)+1.0; according to Teramae, Tsubo & Fukai (2012)
        s = hyp  # Default: 1.0; according to Teramae, Tsubo & Fukai (2012)
        # Draw weight from lognormal distribution
        weights = (np.random.lognormal(m, s, size)*(255/20.)).astype(int)
    if distribution == 'normal':
        m = 10  # mean
        s = 5  # standard deviation
        weights = np.random.normal(m, s, size).astype(int)
    if distribution == 'uniform':
        weights = np.random.uniform(0, 255 ,size).astype(int)

    return weights

"""
@desc: Draw whole weight matrix with given dimensions
        for every mask value which equals 1
"""
def drawSparseWeightMatrix(self, mask, distribution='lognormal'):
    # Get indices and index pointer from mask
    indices = mask.indices
    indptr = mask.indptr

    # Get number of synapses
    numSynapses = np.sum(len(indices))
    # Draw weights
    weights = drawWeights(numSynapses, distribution)

    # Get indices of values were weight is greater than 255
    idxNewValues = np.where(weights > 255)[0]
    # Redraw values as long as all values are below 255
    while len(idxNewValues) > 0:
        # Draw new values
        newValues = np.random.rand(len(idxNewValues))
        # Replace previous values with new values
        np.put(weights, idxNewValues, newValues)
        # Get indices of values greater 255 again
        idxNewValues = np.where(weights > 255)[0]

    # Build sparse weight matrix and return
    return sparse.csr_matrix((weights, indices, indptr), shape=np.shape(mask))

"""
@desc: Draw random weight matrix for reservoir network
"""
def drawAndSetSparseReservoirWeightMatrix(self, mask, *args, **kwargs):
    we = self.drawSparseWeightMatrix(mask, *args, **kwargs)

    # Define and store sub matrices for weights
    nEx = self.p.reservoirExSize
    nAll = self.p.reservoirSize
    self.initialWeights.exex = we[0:nEx, 0:nEx]
    self.initialWeights.inin = -1 * we[nEx:nAll, nEx:nAll]  # change sign of weights
    self.initialWeights.inex = -1 * we[0:nEx, nEx:nAll]  # change sign of weights
    self.initialWeights.exin = we[nEx:nAll, 0:nEx]

"""
@desc: Sets a static weight matrix for reservoir network
"""
def setSparseReservoirWeightMatrix(self, mask, *args, **kwargs):
    # Define and store sub matrices for weights
    nEx = self.p.reservoirExSize
    nAll = self.p.reservoirSize
    self.initialWeights.exex = mask[0:nEx, 0:nEx]*self.p.weightExCoefficient
    self.initialWeights.inin = -1 * mask[nEx:nAll, nEx:nAll]*self.p.weightInCoefficient  # change sign of weights
    self.initialWeights.inex = -1 * mask[0:nEx, nEx:nAll]*self.p.weightInCoefficient  # change sign of weights
    self.initialWeights.exin = mask[nEx:nAll, 0:nEx]*self.p.weightExCoefficient

"""
@desc: Create mask that determines the connections to establish
"""
def drawSparseMaskMatrix(self, p=None, nrows=None, ncols=None, avoidSelf=True):
    # Preprocess some variables
    n = self.p.reservoirExSize + self.p.reservoirInSize
    pc = self.p.reservoirConnProb if p is None else p
    nrows = n if nrows is None else nrows
    ncols = n if ncols is None else ncols

    # Prepare sparse matrix
    indices = []  # column indices
    indptr = [0]  # index pointer, start with 0
    prevRowSum = 0

    # Iterate over rows
    for i in range(nrows):
        # Randomly draw a row
        row = np.random.choice([0, 1], size=(ncols-1,), p=[1-pc, pc])
        # Insert zero value at diagonal (avoid self connections of neurons)
        if avoidSelf: row = np.insert(row, i, 0)
        
        # Get indices where row entries are 1
        rowIdx = np.where(row)[0]
        indices.extend([rowIdx])
        
        # Get cumulative sum of ones in row
        rowSum = prevRowSum + np.sum(row)
        indptr.extend([rowSum])
        
        # Store current value of row sum for next iteration
        prevRowSum = rowSum

    # Flatten indices
    indices = [x for y in indices for x in y]
    # Create data: ones with length of indices
    data = np.ones(len(indices)).astype(int)

    # Build and return sparse mask matrix
    return sparse.csr_matrix((data, indices, indptr), shape=(nrows, ncols))

"""
@desc: Draw mask matrix for reservoir network
"""
def drawAndSetSparseReservoirMaskMatrix(self, *args, **kwargs):
    ma = self.drawSparseMaskMatrix(*args, **kwargs)

    # Define and store sub matrices for masks
    nEx = self.p.reservoirExSize
    nAll = self.p.reservoirSize
    self.initialMasks.exex = ma[0:nEx, 0:nEx]
    self.initialMasks.inin = ma[nEx:nAll, nEx:nAll]
    self.initialMasks.inex = ma[0:nEx, nEx:nAll]
    self.initialMasks.exin = ma[nEx:nAll, 0:nEx]

    return ma

"""
TODO: Shift to utils/weights.py
@desc: Apply mask to weight matrix in order to calculate spectral radius
"""
def getMaskedWeights(self, weights, mask):
    # Apply mask to weights
    maskedWeights = np.ma.array(weights, mask=np.logical_not(mask.toarray()))
    # Set all masked values to zero and divide result by 255 to get weights
    # between 0 and 1, since nxSDK is using weights between 0 and 255
    return maskedWeights.filled(0)

"""
@desc: Get weight matrix from weight probe
"""
def getWeightMatrixFromProbe(self, probeIndex=-1):
    n = self.p.reservoirExSize
    weightMatrix = np.zeros((n,n))

    # Get reservoir mask
    ws = self.initialMasks.exex
    
    # Initialize matrix index
    matrixIndex = 0

    # Loop through matrix
    for i in range(n):
        for j in range(n):
            # If mask value is zero, set matrix value to zero
            if ws[i,j] == 0:
                weightMatrix[i,j] = 0
            # If mask value is one, set matrix value to probe value
            else:
                weightMatrix[i,j] = self.reservoirConnProbe[matrixIndex][0].data[probeIndex]
                matrixIndex += 1

    # Return weight matrix from probe
    return weightMatrix
