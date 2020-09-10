from ..parameters import Parameters
from lib.helper.singleton import Singleton
import logging
import warnings

"""
@desc: Singleton for util functions, like calculating more involved evaluations of data
"""
@Singleton
class Utils():

    """
    @desc: Initiates plot object, gets relation to another object for getting the data
    """
    def __init__(self, parameters=None):
        if parameters is None:
            warnings.warn(
                "No parameters were specified and default parameters are used, this may cause unexpected behavior. "
                "It is more save to pass parameters when instanciating Utils singleton for the first time. "
                "Do it with Utils.instance(parameters=your_parameters)"
            )

        # Take default parameters if not parameter argument is given
        self.p = Parameters() if parameters is None else parameters

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
