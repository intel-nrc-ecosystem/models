# Official modules
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate
from copy import deepcopy
from scipy import sparse
import logging

# Sprizer modules
import lib.anisotropic.lcrn_network as lcrn
import lib.anisotropic.connectivity_landscape as cl

# Own modules
from ..system import System
from ..system.datalog import Datalog
from ..parameters import Parameters
from ..utils import Utils
from ..plots import Plot
from ..network import ReservoirNetwork

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class RewardExperiment():

    """
    @desc: Initiates the experiment
    """
    def __init__(self):
        self.p = Parameters()  # Parameters

        #self.targetFunctions = np.zeros((self.p.traceClusters, self.p.traceSteps))

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p)
        self.system.setDatalog(datalog)
        
        # Instantiate utils and plot
        self.utils = Utils.instance()
        self.plot = Plot(self)

    """
    @desc: Build network
    """
    def build(self):
        # Instanciate network
        self.net = ReservoirNetwork(self.p)

        # Draw mask and weights
        self.drawMaskAndWeights()

        # Connect network
        self.net.addReservoirNetworkDistributed()

        # Add cue
        #self.net.addCueGenerator()

        # Add output neuron
        #self.net.addOutputRewardStructure()

        # Build the network structure
        #self.net.build()
    
    """
    @desc: Draw mask and weights
    """
    def drawMaskAndWeights(self):
        # Draw and store mask matrix
        nAll = self.p.reservoirExSize + self.p.reservoirInSize
        mask = self.net.drawAndSetSparseReservoirMaskMatrix(self.p.reservoirDens, nAll, nAll, avoidSelf=True)

        # Draw and store weight matrix
        self.net.drawAndSetSparseReservoirWeightMatrix(mask)

    """
    @desc: Run network
    """
    def run(self):
        # Run network
        self.net.run()
