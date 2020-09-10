import numpy as np
from tabulate import tabulate

from lib.helper.exceptions import ParamaterNotValid

"""
@desc: Contains all parameters and claculates derived values
"""
class Parameters(ParamaterNotValid):

    from .system import includeSystem
    from .experiment import includeExperiment
    from .derived import computeDerived

    """
    @desc: Initializes all parameters
    @note: Parameters defined as 'None' are computed values (function computeDerivedParameters below)
    """
    def __init__(self, includeDerived = True, update = None):
        # Include parameters
        self.includeSystem()
        self.includeExperiment()

        # If parameters are updated, add/update them in list
        if update is not None:
            for parameter, value in update.items():
                setattr(self, parameter, value)

        # Check if all paramaters are valid before calculating derived parameters
        self.vadilityCheckBefore()

        # If derived values shall be included, calculate them
        if includeDerived: self.computeDerived()

        # Check if all paramaters are valid after calculating derived parameters
        self.vadilityCheckAfter()
    
    """
    @desc: Transform all available parameters to string table format for printing
    """
    def __str__(self):
        # Get names of all elements in parameter module
        parNames = dir(self)
        # Initialize parameter table
        parTable = []

        # Loop through all variables names in paramater module
        for pn in parNames:
            # If variable starts with __ continue with next loop step
            if pn.startswith('__') or pn.startswith('np'):
                continue
            # Append current variable name and variable content as list to tab list
            parTable.append([pn, getattr(self, pn)])
            
        # Return list of variables in table format
        return tabulate(parTable)

    """
    @desc: Check if values make sense BEFORE calculating derived parameters and raise error if not
    """
    def vadilityCheckBefore(self):
        # inputNumTargetNeurons and inputShareTargetNeurons are both not set
        if self.inputNumTargetNeurons is None and self.inputShareTargetNeurons is None:
            raise ParamaterNotValid('inputNumTargetNeurons and inputShareTargetNeurons are both set to None. Specify one of them.')

        # Only inputNumTargetNeurons or inputShareTargetNeurons can be set
        if self.inputNumTargetNeurons is not None and self.inputShareTargetNeurons is not None:
            raise ParameterNotValid('Either inputNumTargetNeurons or inputShareTargetNeurons can be set, not both. Set one of them to None to avoid ambiguity.')

        # reservoirConnProb and reservoirConnPerNeuron are both not set
        if self.reservoirConnProb is None and self.reservoirConnPerNeuron is None:
            raise ParamaterNotValid('reservoirConnProb and reservoirConnPerNeuron are both set to None. Specify one of them.')

        # Only the connection probability or the number of connections per neuron can be set
        if self.reservoirConnProb is not None and self.reservoirConnPerNeuron is not None:
            raise ParameterNotValid('Either reservoirConnProb or reservoirConnPerNeuron can be set, not both. Set one of them to None to avoid ambiguity.')

    """
    @desc: Check if values make sense AFTER calculating derived parameters and raise error if not
    """
    def vadilityCheckAfter(self):
        # If number of connections per neuron is larger than total number of neurons
        if self.reservoirConnPerNeuron > (self.reservoirExSize + self.reservoirInSize):
            raise ParamaterNotValid('Number of connections per neuron must be larger than number of neurons in the network.')

        # Check if size of patch input is smaller than network site
        if self.inputNumTargetNeurons > self.reservoirSize:
            raise ParamaterNotValid('Input size is too large, cannot be larger than network size.')

        # Check if number of neurons per core is properly chosen
        if int(self.reservoirExSize/self.neuronsPerCore) > self.numChips*self.numCoresPerChip:
            raise ParameterNotValid('Number of cores exceeded, increase number of neurons per core.')
