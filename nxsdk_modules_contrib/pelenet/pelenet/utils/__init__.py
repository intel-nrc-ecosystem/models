from ..parameters import Parameters
from lib.helper.singleton import Singleton
import logging

"""
@desc: Singleton for util functions, like calculating more involved evaluations of data
"""
@Singleton
class Utils():

    """
    @desc: Initiates plot object, gets relation to another object for getting the data
    """
    def __init__(self):
        # Store parameters
        self.p = Parameters()

    """
    @desc: Parameters for utils can be changed manually
    """
    def setParameters(self, parameters):
        # Manually set parameters
        self.p = parameters

    """
    @note: Import functions from files
    """
    # Functions to evaluate spikes
    from .spikes import (
        getSpikesFromActivity, cor, getFilteredSpikes, fano, cv,
        getGaussianFilteredSpikes, getSingleExponentialFilteredSpikes, getHoltDoubleExponentialFilteredSpikes
    )
    # Functions to evaluate weights
    from .weights import (
        getSpectralRadius, recombineExWeightMatrix, getSupportWeightsMask
    )
    # Functions to evaluate target
    from .target import (
        loadTarget, prepareDataset, estimateMovement, estimateMultipleTrajectories3D
    )
    # Other functions
    from .misc import (
        trainOLS, pca
    )
