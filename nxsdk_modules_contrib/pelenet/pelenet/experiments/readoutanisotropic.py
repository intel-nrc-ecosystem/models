# Pelenet modules
from .anisotropic import AnisotropicExperiment
from ..network import ReservoirNetwork
from ._abstract import Experiment

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class AnisotropicReadoutExperiment(AnisotropicExperiment):

    """
    # @desc: Define parameters for this experiment
    # """
    def defineParameters(self):
        # Parent parameters
        aniP = super().defineParameters()

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

        # Experiment parameters overwrite parameters from parent experiment
        return { **aniP, **expP }
    
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
