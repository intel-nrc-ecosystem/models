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
from .anisotropic import AnisotropicExperiment
from ..network import ReservoirNetwork

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class ReadoutExperiment(AnisotropicExperiment):

    """
    @desc: Initiates the experiment
    """
    def __init__(self):
        # Parameters
        self.p = Parameters(update = self.updateParameters())

        self.net = None

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p)
        self.system.setDatalog(datalog)

        # Instantiate utils and plot
        self.utils = Utils.instance()
        self.utils.setParameters(self.p)
        self.plot = Plot(self)

        # Define some further variables
        self.target = self.utils.loadTarget()

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
            'trials': 25,
            'stepsPerTrial': 210, #500,
            # Network
            'refractoryDelay': 2, # Sparse activity (high values) vs. dense activity (low values)
            'compartmentVoltageDecay': 400,  # Slows down / speeds up
            'compartmentCurrentDecay': 380,  # Variability (higher values) vs. Stability (lower values)
            'thresholdMant': 1000,  # Slower spread (high values) va. faster spread (low values)
            # Input
            'patchNeuronsShiftX': 44,
            'patchNeuronsShiftY': 24,
            # Output
            'partitioningClusterSize': 10, #6, #6/10  # size of clusters connected to an output neuron
            # Probes
            'isExSpikeProbe': False,
            'isOutSpikeProbe': True,
            # Target
            'targetFilename': 'test1_rec.txt',
            'targetOffset': 1000
        }
    
    """
    @desc: Build all networks
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)
        self.net.landscape = None

        # Draw anisotropic mask and weights
        self.drawMaskAndWeights()

        # Draw output weights
        self.net.drawOutputMaskAndWeights()

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Connect reservoir to output
        self.net.connectOutput()

        # Add patch input
        self.net.addRepeatedPatchGenerator()

        # Add background noise
        #self.net.addNoiseGenerator()

        # Build the network structure
        self.net.build()
    
    """
    @desc: Run whole experiment
    """
    def run(self):
        # Compile network
        compiler = nx.N2Compiler()
        board = compiler.compile(self.net.nxNet)
        logging.info('Network successfully compiled')

        # Add snips and channel
        resetInitSnips = self.net.addResetSnips(board)  # add snips
        resetInitChannels = self.net.createAndConnectResetInitChannels(board, resetInitSnips)  # create channels for transfering initial values for the reset SNIP
        
        # Start board
        board.start()
        logging.info('Board successfully started')

        # Write initial data to channels
        for i in range(self.p.numChips):
            resetInitChannels[i].write(3, [
                self.p.neuronsPerCore,  # number of neurons per core
                self.p.totalTrialSteps,  # reset interval
                self.p.resetSteps  # number of steps to clear voltages/currents
            ])
        logging.info('Initial values transfered to SNIPs via channel')

        # Run and disconnect board
        board.run(self.p.totalSteps)
        board.disconnect()

        # Perform postprocessing
        self.net.postProcessing()
