# Official modules
import numpy as np
from scipy import sparse
import logging

# Sprizer modules
import lib.anisotropic.lcrn_network as lcrn
import lib.anisotropic.connectivity_landscape as cl

# Pelenet modules
from ..network import ReservoirNetwork
from ._abstract import Experiment

"""
@desc: Class for running an experiment, usually contains performing
       several networks (e.g. for training and testing)
"""
class AnisotropicExperiment(Experiment):

    """
    # @desc: Define parameters for this experiment
    # """
    def defineParameters(self):
        return {
            # Experiment
            'seed': 3,  # Random seed
            'trials': 1,  # Number of trials
            'stepsPerTrial': 600,  # Number of simulation steps for every trial
            # Network
            'refractoryDelay': 2, # Refactory period
            'voltageTau': 10.24,  # Voltage time constant
            'currentTau': 10.78,  # Current time constant
            'thresholdMant': 1000,  # Spiking threshold for membrane potential
            'reservoirConnProb': 0.05,  # Connection probability
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
            'inputNumTargetNeurons': 36,  # Number of target neurons for the input
            'inputSteps': 5,  # Number of steps the network is activated by the input
            'inputWeightExponent': 0,  # The weight exponent of the weights from the generator to the target neurons
            'inputGenSpikeProb': 1.0,  # Spiking probability of the spike generators
            # Probes
            'isExSpikeProbe': True,  # Probe excitatory spikes
            'isInSpikeProbe': True   # Probe inhibitory spikes
        }
    
    """
    @desc: Build network
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)
        self.net.landscape = None

        # Draw anisotropic mask and weights
        self.drawMaskAndWeights()

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add leave-n-out input
        self.net.addInput()

        # Add Probes
        self.net.addProbes()

    # """
    # @desc: Run experiment
    # """
    # def run(self):
    #     # Run network
    #     self.net.run()

    """
    @desc: Summary of some plots about the network
    """
    def plotSummary(self):
        # Plot histogram of weights and calc spectral radius
        self.net.plot.initialExWeightDistribution()

        # Plot weight matrix
        self.net.plot.initialExWeightMatrix()

    """
    @desc: Draw mask and weights
    """
    def drawMaskAndWeights(self):
        # Draw and store mask matrix
        self.drawSparseAnisotropicMaskMatrix()
        
        # Define and store weight matrix
        self.setSparseWeightMatrix()

    """
    @desc: Draw anisotropic mask matrix
    """
    def drawSparseAnisotropicMaskMatrix(self):
        # Get population sizes from parameters
        npopE = self.p.reservoirExSize
        npopI = self.p.reservoirInSize

        # Get numbber of columns and rows of network topology, calculated from population sizes
        nrowE, ncolE = int(np.sqrt(npopE)), int(np.sqrt(npopE)) #120, 120
        nrowI, ncolI = int(np.sqrt(npopI)), int(np.sqrt(npopI)) #60, 60

        # Predefine some parameter shorthands
        p = self.p.reservoirConnProb
        stdE = self.p.anisoStdE
        stdI = self.p.anisoStdI
        
        # Directions
        move = cl.move(nrowE)

        # Generate landscape
        landscape = cl.Perlin(nrowE, {'size': self.p.anisoPerlinScale})

        # Initialize weight matrix masks for Loihi
        ee = np.zeros((npopE, npopE))
        ei = np.zeros((npopI, npopE))
        ie = np.zeros((npopE, npopI))
        ii = np.zeros((npopI, npopI))

        # Excitatory landscape from Sebastian, choose symmetric
        for idx in range(npopE):

            # E-> E
            source = idx, nrowE, ncolE, nrowE, ncolE, int(p * npopE), stdE
            targets, delay = lcrn.lcrn_gauss_targets(*source)
            if landscape[idx] != 0:  # asymmetry
                #if np.random.rand() <= self.p.percShift:  # for perc_shift < 1, some are not shifted
                #    targets = (targets + self.p.anisoShift * move[landscape[idx] % len(move)]) % npopE
                #else:
                #    not_shifted.append(idx)
                targets = (targets + self.p.anisoShift * move[landscape[idx] % len(move)]) % npopE
            targets = targets[targets != idx]
            
            # Set Loihi mask value
            ee[targets, idx] = 1
            
            # E-> I
            source = idx, nrowE, ncolE, nrowI, ncolI, int(p * npopI), stdI
            targets, delay = lcrn.lcrn_gauss_targets(*source)
            
            # Set Loihi mask value
            ei[targets, idx] = 1
                        
        # inhibitory connections
        for idx in range(npopI):

            # I-> E
            source = idx, nrowI, ncolI, nrowE, ncolE, int(p * npopE), stdE
            targets, delay = lcrn.lcrn_gauss_targets(*source)

            # Set Loihi mask value
            ie[targets, idx] = 1
            
            # I-> I
            source = idx, nrowI, ncolI, nrowI, ncolI, int(p * npopI), stdI
            targets, delay = lcrn.lcrn_gauss_targets(*source)
            targets = targets[targets != idx]
            
            # Set Loihi mask value
            ii[targets, idx] = 1

        # Store landscape
        self.net.landscape = landscape

        # Store masks
        self.net.initialMasks.exex = sparse.csr_matrix(ee)
        self.net.initialMasks.inin = sparse.csr_matrix(ii)
        self.net.initialMasks.inex = sparse.csr_matrix(ie)
        self.net.initialMasks.exin = sparse.csr_matrix(ei)

        # Log that weight matrix was generated
        logging.info('Anisotropic weight matrix was succesfully drawn')
    
    """
    @desc: Set sparse weight matrix for anisotropic network
    """
    def setSparseWeightMatrix(self):
        # Set constant weights for excitatory and inhibitory neurons
        self.net.initialWeights.exex = self.p.weightExCoefficient * self.net.initialMasks.exex
        self.net.initialWeights.inin = -self.p.weightInCoefficient * self.net.initialMasks.inin  # change sign of weights
        self.net.initialWeights.inex = -self.p.weightInCoefficient * self.net.initialMasks.inex  # change sign of weights
        self.net.initialWeights.exin = self.p.weightExCoefficient * self.net.initialMasks.exin
    