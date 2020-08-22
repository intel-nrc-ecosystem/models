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
from ..network import ReservoirNetwork

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class ReadoutRandomExperiment():

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
        return {
            # Experiment
            'trials': 25,
            'stepsPerTrial': 210,
            # Network
            'reservoirConnProb': 0.035,
            #'reservoirConnPerNeuron': 50,
            'weightExCoefficient': 12, #12
            'weightInCoefficient': 48,
            # Output
            'partitioningClusterSize': 10, #6, #6/10  # size of clusters connected to an output neuron
            # Neurons
            'refractoryDelay': 2, #5, #2 # Sparse activity (high values) vs. dense activity (low values)
            'compartmentVoltageDecay': 100, #500, #100,#400 #500,  # Slows down / speeds up
            'compartmentCurrentDecay': 4096, #500, #300, #500,  # Variability (higher values) vs. Stability (lower values)
            'thresholdMant': 50, #400, #1000, #800,  # Slower spread (high values) va. faster spread (low values)
            'outputWeightValue': 1,
            # Probes
            'isExSpikeProbe': True,
            'isOutSpikeProbe': True,
            # Target
            'targetFilename': 'test1_rec.txt',
            'targetOffset': 1000
        }

    # r, vol, cur,  thr 
    # 2, 100, 2000, 200
    # 2, 100, 1000, 300
    # 2, 100, 4096, 50  X
    # 2, 20,  4096, 50
    
    """
    @desc: Build all networks
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)

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

    """
    @desc: Draw mask and weights
    """
    def drawMaskAndWeights(self):
        # Draw and store mask matrix
        ma = self.net.drawAndSetSparseReservoirMaskMatrix()

        # Draw and store weight matrix
        self.net.setSparseReservoirWeightMatrix(ma)
        #self.net.drawAndSetSparseReservoirWeightMatrix(ma)
