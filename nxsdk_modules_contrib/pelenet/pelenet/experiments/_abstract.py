# Official modules
import logging
from abc import ABC, abstractmethod

# Loihi modules
import nxsdk.api.n2a as nx

# Own modules
from ..system import System
from ..system.datalog import Datalog
from ..parameters import Parameters
from ..utils import Utils
from ..plots import Plot
from ..network import ReservoirNetwork

"""
@desc: Abstract experiment class
"""
class Experiment(ABC):

    """
    @desc: Initiates the experiment
    """
    def __init__(self, name='', parameters={}):
        # Parameters from jupyter notebook overwrite parameters from experiment definition
        p = { **self.defineParameters(), **parameters}
        self.p = Parameters(update = p)

        # Define empty net variable
        self.net = None

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p, name=name)
        self.system.setDatalog(datalog)

        # Instantiate utils and plot
        self.utils = Utils.instance(parameters=self.p)
        self.plot = Plot(self)

        # Call onInit
        self.onInit()

    """
    @desc: Lifecycle function called after __init__
    """
    def onInit(self):
        pass

    """
    @desc: Overwrite parameters for this experiment
    """
    @abstractmethod
    def defineParameters(self):
        return {
            # 'parameterName': value
        }

    """
    @desc: Build reservoir network with all parts (input, output, noise, etc.)
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)

        # Draw anisotropic mask and weights
        self.net.drawMaskAndWeights()

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add cue
        self.net.addInput()

        # Add background noise
        if self.p.isNoise:
            self.net.addNoiseGenerator()

        # Add Probes
        self.net.addProbes()

        # Call afterBuild
        self.afterBuild()

    """
    @desc: Lifecycle function called after build
    """
    def afterBuild(self):
        pass

    """
    @desc: Run experiment
    """
    def run(self):
        # Run network
        #self.net.run()

        if not self.p.isReset:
            self.net.nxNet.run(self.p.totalSteps)
            self.net.nxNet.disconnect()

        if self.p.isReset:
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
            for i in range(self.net.numChipsUsed):
                resetInitChannels[i].write(3, [
                    self.p.neuronsPerCore,  # number of neurons per core
                    self.p.totalTrialSteps,  # reset interval
                    self.p.resetSteps  # number of steps to clear voltages/currents
                ])
            logging.info('Initial values transfered to SNIPs via channel')

            # Run and disconnect board
            board.run(self.p.totalSteps)
            board.disconnect()

        logging.info('Loihi finished simulation successfully')

        # Post processing of probes
        self.net.postProcessing()

        # Call afterRun
        self.afterRun()

    """
    @desc: Lifecycle function called after run
    """
    def afterRun(self):
        pass
    