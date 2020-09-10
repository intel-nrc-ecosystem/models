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
class AnisotropicReadoutExperiment(AnisotropicExperiment):

    """
    @desc: Initiates the experiment
    """
    def __init__(self, name='', parameters={}):
        # Parameters
        self.p = Parameters(update = self.updateParameters(parameters))

        self.net = None

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p, name=name)
        self.system.setDatalog(datalog)

        # Instantiate utils and plot
        self.utils = Utils.instance(parameters=self.p)
        self.plot = Plot(self)

        # Define some further variables
        #self.target = self.utils.loadTarget()

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self, jupP={}):
        # Parent parameters
        aniP = super().updateParameters()

        expP = {
            # Experiment
            'seed': 3,  # Random seed
            'trials': 25,  # Number of trials
            'stepsPerTrial': 110,  # Number of simulation steps for every trial
            'isReset': True,  # Activate reset after every trial
            # Network
            'refractoryDelay': 2, # Refactory period
            'voltageTau': 10.24,  # Voltage time constant
            'currentTau': 10.78,  # Current time constant
            'thresholdMant': 1000,  # Spiking threshold for membrane potential
            'reservoirConnProb': 0.05,
            # Anisotropic
            'anisoStdE': 12,  # Space constant, std of gaussian for excitatory neurons
            'anisoStdI': 9,  # Space constant, std of gaussian for inhibitory neurons (range 9 - 11)
            'anisoShift': 1,  # Intensity of the shift of the connectivity distribution for a neuron
            #'percShift': 1,  # Percentage of shift (default 1)
            'anisoPerlinScale': 4,  # Perlin noise scale, high value => dense valleys, low value => broad valleys
            'weightExCoefficient': 12,  # Coefficient for excitatory anisotropic weight
            'weightInCoefficient': 48,  # Coefficient for inhibitory anisotropic weight
            # Input
            'inputIsTopology': True,  # Activate a 2D input area
            'inputIsLeaveOut': True,  # Leaves one target neuron out per trial
            'patchNeuronsShiftX': 44,  # x-position of the input area
            'patchNeuronsShiftY': 24,  # y-position of the input area
            'inputNumTargetNeurons': 25,  # Number of target neurons for the input
            'inputSteps': 5,  # Number of steps the network is activated by the input
            'inputWeightExponent': 0,    # The weight exponent of the weights from the generator to the target neurons
            'inputGenSpikeProb': 1.0,  # Spiking probability of the spike generators
            # Output
            'partitioningClusterSize': 10, # Size of clusters connected to an output neuron (6|10)
            # Probes
            'isExSpikeProbe': True,  # Probe excitatory spikes
            'isInSpikeProbe': True,   # Probe inhibitory spikes
            'isOutSpikeProbe': True   # Probe output spikes
        }

        # Parameters from jupyter notebook overwrite parameters from experiment definition
        # Experiment parameters overwrite parameters from parent experiment
        return { **aniP, **expP, **jupP}
    
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
        self.net.addInput()

        # Add Probes
        self.net.addProbes()
    
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
