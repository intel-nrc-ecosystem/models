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
@desc: Randomly connected network
"""
class RandomExperiment():

    """
    @desc: Initiates the experiment
    """
    def __init__(self, name='', parameters={}):
        self.p = Parameters(update = self.updateParameters(parameters))

        self.net = None
        self.trainSpikes = None

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p, name=name)
        self.system.setDatalog(datalog)

        # Instantiate utils and plot
        self.utils = Utils.instance(parameters=self.p)
        self.plot = Plot(self)

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self, jupP={}):
        expP = {
            # Experiment
            'seed': 1,  # Random seed
            'trials': 1,  # Number of trials
            'stepsPerTrial': 600,  # Number of simulation steps for every trial
            # Network
            'reservoirExSize': 400,  # Number of excitatory neurons
            'reservoirConnPerNeuron': 40,  # Number of connections per neuron
            # Neurons
            'refractoryDelay': 2,  # Refactory period
            'voltageTau': 20,  # Voltage time constant
            'currentTau': 5,  # Current time constant
            'thresholdMant': 800,  # Spiking threshold for membrane potential
            # Input
            'inputGenSpikeProb': 0.5,  # Probability of spikes for the spike generators
            'inputWeightExponent': 2,  # Weight exponent for the connections from the generators to the reservoir neurons
            'inputNumTargetNeurons': 50,  # Number of neurons targeted by the spike generators
            'inputSteps': 10,  # Number of steps the input is active
            # Probes
            'isExSpikeProbe': True,  # Probe excitatory spikes
            'isInSpikeProbe': True  # Probe inhibitory spikes
        }

        # Parameters from jupyter notebook overwrite parameters from experiment definition
        return { **expP, **jupP}
    
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
        self.net.addInput()

        # Add Probes
        self.net.addProbes()
    
    """
    @desc: Run experiment
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
