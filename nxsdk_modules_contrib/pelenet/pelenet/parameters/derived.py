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
    self.resetOffset = None  # offset due to reset (reset steps + reset relaxation)

    """
    Define some derived parameters
    """

    # Set some derived parameters directly
    self.patchNeurons = np.square(self.patchSize)  # Number of patch input neurons
    self.topologySize = int(np.sqrt(self.reservoirExSize))  # Size of one dimension of topology
    self.constSize = int(self.constSizeShare * self.reservoirExSize)
    
    # Derived output values
    self.numOutClusters = int(self.reservoirExSize / np.square(self.partitioningClusterSize))
    self.numOutputNeurons = 2 * self.numOutClusters
    self.numOutDimSize = int(np.sqrt(self.numOutClusters))

    # Set datalog path for the current experiment, depending on the current time
    #self.expLogPath

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

    # Calculate reset offset if reset is applied
    if self.isReset:
        self.resetOffset = self.resetSteps + self.resetRelaxation

    # Calculate steps
    self.inputOffset = self.inputSteps + self.inputRelaxation
    self.trialSteps = self.inputOffset + self.stepsPerTrial
    self.totalTrialSteps = self.trialSteps + self.resetOffset
    self.totalSteps = self.totalTrialSteps * self.trials

    # If noiseNeurons is not set (None), calculate it with given share
    if self.noiseNeurons is None:
        self.noiseNeurons = int(self.noiseNeuronsShare * self.reservoirExSize)

    # Trace input size
    if self.traceClusterShare is None:
        self.traceClusterShare = int(self.traceClusterSize / self.reservoirExSize)
    if self.traceClusterSize is None:
        self.traceClusterSize = int(self.traceClusterShare * self.reservoirExSize)

    # Calculate connectivity
    if self.reservoirConnProb is None:
        self.reservoirConnProb = float(self.reservoirConnPerNeuron / self.reservoirSize)
    if self.reservoirConnPerNeuron is None:
        self.reservoirConnPerNeuron = int(self.reservoirConnProb * self.reservoirSize)
