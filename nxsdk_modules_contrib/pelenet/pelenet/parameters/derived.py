import numpy as np

"""
@desc: Some values are derived and need to be computed
@note:  - This function is necessary, since it needs to be called
          when parameter optimization is performed
        - Derived parameters are not allowed to change within one experiment
          (only inbetween experiments), for values which are more flexibel,
          use system variables
"""
def computeDerived(self):

    # Initialize some derived parameters
    self.totalSteps = None  # number of simulation steps
    self.reservoirSize = None  # total size of reservoir
    self.totalTrialSteps = None  # number of total steps per trial
    
    """
    Define some derived parameters
    """
    
    # Derived output values
    self.numCores = self.numChips*self.numCoresPerChip
    self.numOutClusters = int(self.reservoirExSize / np.square(self.partitioningClusterSize))
    self.numOutputNeurons = 2 * self.numOutClusters
    self.numOutDimSize = int(np.sqrt(self.numOutClusters))

    """
    Define conditional parameters
    """

    # Size of inhibitory/excitatory network
    if self.reservoirInSize is None:
        self.reservoirInSize = int(self.reservoirInExRatio * self.reservoirExSize)
    if self.reservoirInExRatio is None:
        self.reservoirInExRatio = int(self.reservoirExSize / self.reservoirInSize)

    # Calculate total size of the network
    self.reservoirSize = self.reservoirInSize + self.reservoirExSize

    # Calculate steps
    self.resetOffset = self.resetSteps + self.resetRelaxation if self.isReset else 0
    self.totalTrialSteps = self.stepsPerTrial + self.resetOffset
    self.totalSteps = self.totalTrialSteps * self.trials

    # If noiseNeurons is not set (None), calculate it with given share
    if self.noiseNeurons is None:
        self.noiseNeurons = int(self.noiseNeuronsShare * self.reservoirExSize)
    
    # Target neurons input size
    if self.inputShareTargetNeurons is None:
        self.inputShareTargetNeurons = int(self.inputNumTargetNeurons / self.reservoirExSize)
    if self.inputNumTargetNeurons is None:
        self.inputNumTargetNeurons = int(self.inputShareTargetNeurons * self.reservoirExSize)

    # Calculate connectivity
    if self.reservoirConnProb is None and self.reservoirConnPerNeuron is not None:
        self.reservoirConnProb = float(self.reservoirConnPerNeuron / self.reservoirSize)
    if self.reservoirConnPerNeuron is None and self.reservoirConnProb is not None:
        self.reservoirConnPerNeuron = int(self.reservoirConnProb * self.reservoirSize)
