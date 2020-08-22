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
class AnisotropicExperiment():

    """
    @desc: Initiates the experiment
    """
    # TODO user decorator for default stuff (like creating instances),
    # maybe some stuff from basicNetwork can be included?
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
        self.plot = Plot(self)

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self):
        return {
            'anisoStdE': 12, #6 #10 #12  # space constant, std of gaussian for excitatory neurons
            'anisoStdI': 9, #4.5 #8 #9  # space constant, std of gaussian for inhibitory neurons (range 9 - 11)
            'anisoShift': 1, # intensity of the shift of the connectivity distribution for a neuron
            #'percShift': 1, #0.8  # percentage of shift (default 1)
            'anisoPerlinScale': 4, #8 # 4-12  # perlin noise scale, high value => dense valleys, low value => broad valleys
            'weightExCoefficient': 12, #16 #8 #16 # 8 #8 #16 #8 #4  # coefficient for excitatory anisotropic weight
            'weightInCoefficient': 48 #64 #32 #64 # 28 #32 #64 #28 sieht gut aus!! #32 #22  # coefficient for inhibitory anisotropic weight, Perlin scale 4: 25-30 ok, 25-28 good
        }
    
    """
    @desc: Build network
    """
    def build(self):
        # Instanciate innate network
        #self.net = ReservoirNetwork(self.p)
        #self.net.landscape = None

        # Draw anisotropic mask and weights
        #self.drawMaskAndWeights()

        # Connect network
        #self.net.addReservoirNetworkDistributed()

        # Add cue
        #self.net.addCueGenerator()

        # Add stop signal
        #self.net.addStopGenerator()

        # Add background noise
        #self.net.addNoiseGenerator()

        # Build the network structure
        #self.net.build()
        pass

    """
    @desc: Summary of some plots about the network
    """
    def plotSummary(self):
        # Plot histogram of weights and calc spectral radius
        self.net.plot.initialExWeightDistribution()

        # Plot weight matrix
        self.net.plot.initialExWeightMatrix()

    """
    @desc: Run network
    """
    def run(self):
        # Run network
        self.net.run()

    """
    @desc: Saves network in numpy file
    """
    # FIXME Does not work
    #def save(self):
        # Saves whole experiment
        #np.save(self.system.datalog.dir+'data/experiment.npy', self)

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
    