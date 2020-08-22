# Loihi modules
import nxsdk.api.n2a as nx

# Official modules
import numpy as np
import logging
from copy import deepcopy
import os

# Pelenet modules
from ..system import System
from ..system.datalog import Datalog
from ..parameters import Parameters
from ..utils import Utils
from ..plots import Plot
from .readout import ReadoutExperiment
from ..network import ReservoirNetwork

"""
@desc: Class for comparing anisotropic nest simulation with anisotropic loihi simulation
"""
class NestComparison(ReadoutExperiment):

    """
    @desc: Initiates the experiment
    """
    def __init__(self):
        super().__init__()

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self):
        # Update patameters from parent
        p = super().updateParameters()

        return {
            # Parameters from parent
            **p,
            # Experiment
            'trials': 1,
            'stepsPerTrial': 500,
            # Network
            'refractoryDelay': 2, # Sparse activity (high values) vs. dense activity (low values)
            'compartmentVoltageDecay': 400, #400, #500,  # Slows down / speeds up
            'compartmentCurrentDecay': 380, #425, #500  # Variability (higher values) vs. Stability (lower values)
            'thresholdMant': 1000,  # Slower spread (high values) va. faster spread (low values)
            # Probes
            'isExSpikeProbe': True,
            'isOutSpikeProbe': False
        }

        # 400, 380, 1000

        # 500, 400, 900 -> equal firing rate
        # 400, 375, 1000 -> equal firing rate
        # 300, 400, 1000 -> equal firing rate
        # lower threshold and higher decays looks good! (e.g. 600, 600, 800) -> influence on performance?
