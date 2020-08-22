# Official modules
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate
from copy import deepcopy

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
class StaticExperiment():

    """
    @desc: Initiates the experiment
    """
    def __init__(self):
        self.p = Parameters(update = self.updateParameters())

        self.net = None
        self.trainSpikes = None

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p)
        self.system.setDatalog(datalog)

        # Instantiate utils and plot
        self.utils = Utils.instance()
        self.utils.setParameters(self.p)
        self.plot = Plot(self)

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self):
        return {
            # Experiment
            'trials': 1,
            'stepsPerTrial': 600,
            # Network
            #'reservoirConnPerNeuron': 40,
            'reservoirConnProb': 0.004,
            #'weightExCoefficient': 12, #12
            #'weightInCoefficient': 48,
            # Neurons
            'refractoryDelay': 2, #5, #2 # Sparse activity (high values) vs. dense activity (low values)
            'compartmentVoltageDecay': 100, #500, #100,#400 #500,  # Slows down / speeds up
            'compartmentCurrentDecay': 4096, #500, #300, #500,  # Variability (higher values) vs. Stability (lower values)
            'thresholdMant': 50, #400, #1000, #800,  # Slower spread (high values) va. faster spread (low values)
            # Input
            'patchSize': 20,
            # Probes
            'isExSpikeProbe': True,
            'isInSpikeProbe': True
        }
    
    """
    @desc: Build network
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)

        # Draw random mask and weights
        self.drawMaskAndWeights()

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add patch input
        self.net.addRepeatedPatchGenerator()

        # Plot histogram of weights and calc spectral radius
        #self.net.plot.initialExWeightDistribution()

        # Plot weight matrix
        #self.net.plot.initialExWeightMatrix()

        # Build the network structure
        self.net.build()
    """
    @desc: Run network
    """
    def run(self):
        # Run network
        self.net.run()

    """
    @desc: Draw mask and weights
    """
    def drawMaskAndWeights(self):
        # Draw and store mask matrix
        ma = self.net.drawAndSetSparseReservoirMaskMatrix()

        # Draw and store weight matrix
        #self.net.setSparseReservoirWeightMatrix(ma)
        self.net.drawAndSetSparseReservoirWeightMatrix(ma)
    
    """
    @desc: Call several function in order to evaluate results
    """
    def evaluateExperiment(self):
        idx = self.p.cueSteps + self.p.relaxationSteps

        # Prepare spikes for evaluation
        self.trainSpikes = np.array([net.reservoirProbe[2].data for net in self.trainNets])
        trainSpikesMean = np.mean(self.trainSpikes, axis=0)
        self.testSpikes = self.testNet.reservoirProbe[2].data

        # Plot autocorrelation function
        self.plot.autocorrelation(trainSpikesMean[:,idx:], numNeuronsToShow=1)
        # TODO is it a good idea to mean over trials? maybe use index selection like for fano factors

        # Plot crosscorrelation function
        self.plot.crosscorrelation(self.testSpikes)

        # Plot spike missmatch
        self.plot.spikesMissmatch(self.trainSpikes[i,:,fr:to], self.testSpikes[:,fr:to])

        # Plot fano factors of spike counts (test spikes)
        self.plot.ffSpikeCounts(self.testSpikes, neuronIdx = [1,5,10,15])

        # Plot first 2 components of PCA of all spikes 
        self.plot.pca(self.testSpikes)

        # Evalulate assembly weights
        self.evaluateAssemblyWeights()

        # Plot pre synaptic trace
        self.trainNet.plot.preSynapticTrace()
