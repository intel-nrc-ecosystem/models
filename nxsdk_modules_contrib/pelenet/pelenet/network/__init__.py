import nxsdk.api.n2a as nx
from types import SimpleNamespace
import numpy as np
import logging

# Importing own modules
from ..utils import Utils
from ..plots import Plot
from ..parameters import Parameters

"""
@desc: Reservoir network
"""
class ReservoirNetwork():

    """
    @desc: Initiates the innate network
    """
    def __init__(self, parameters=None):
        # Get parameters
        self.p = Parameters() if parameters is None else parameters

        # Set seed
        if self.p.seed is not None:
            np.random.seed(self.p.seed)

        # Instanciate nx net object
        self.nxNet = nx.NxNet()

        # Excitatory connection prototype
        self.exConnProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                  weightExponent=self.p.weightExponent, numTagBits=self.p.numTagBits,
                                                  numDelayBits=self.p.numDelayBits, numWeightBits=self.p.numWeightBits)
        # Inhibitory connection prototype
        self.inConnProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                                                  weightExponent=self.p.weightExponent, numTagBits=self.p.numTagBits,
                                                  numDelayBits=self.p.numDelayBits, numWeightBits=self.p.numWeightBits)
        # Mixed connection prototype
        self.mixedConnProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                                     weightExponent=self.p.weightExponent, numTagBits=self.p.numTagBits,
                                                     numDelayBits=self.p.numDelayBits, numWeightBits=self.p.numWeightBits)
        # Generator connection prototype
        self.genConnProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                   weightExponent=self.p.inputWeightExponent, numTagBits=self.p.numTagBits,
                                                   numDelayBits=self.p.numDelayBits, numWeightBits=self.p.numWeightBits)

        """
        Network objects
        """
        # Cores
        self.cores = np.arange(self.p.numChips*self.p.numCoresPerChip)

        # Weights
        self.initialMasks = SimpleNamespace(**{
            'exex': None, 'inin': None, 'inex': None, 'exin': None
        })

        self.initialWeights = SimpleNamespace(**{
            'exex': None, 'inin': None, 'inex': None, 'exin': None
        })
        self.trainedWeightsExex = None

        # NxSDK compartment group chunks
        self.exReservoirChunks = []
        self.inReservoirChunks = []
        self.outputLayerChunks = []
        self.connectionChunks = []

        # Probes
        self.exSpikeProbes = []
        self.inSpikeProbes = []
        self.outSpikeProbes = []
        self.exVoltageProbes = []
        self.inVoltageProbes = []
        self.outVoltageProbes = []
        self.exCurrentProbes = []
        self.inCurrentProbes = []
        self.weightProbes = []

        # Output
        self.outputMask = None
        self.outputWeights = None

        # Spikes
        self.exSpikeTrains = []
        self.inSpikeTrains = []
        self.outSpikeTrains = []

        # Voltages
        self.outVoltageTrains = []

        # Trace input
        self.traceSpikes = []
        self.traceMasks = []
        self.traceWeights = []

        # Input
        self.inputTargetNeurons = []
        self.inputSpikes = []
        self.inputWeights = None

        # Noise input spikes
        self.noiseSpikes = None
        self.noiseWeights = None

        # Instantiate utils and plot
        self.utils = Utils.instance()
        self.plot = Plot(self)
    
    """
    @desc: Run the network
    """
    def run(self):
        self.nxNet.run(self.p.totalSteps)
        self.nxNet.disconnect()

        # Post processing of probes
        self.postProcessing()

    """
    @note: Import functions from files
    """
    from .connect import removeCoreFromList, connectReservoir, connectOutput
    from .input import addInput
    from .noise import addNoiseGenerator, addConstantGenerator
    from .output import drawOutputMaskAndWeights
    from .probes import addProbes, condenseSpikeProbes, postProcessing
    from .snips import addResetSnips, createAndConnectResetInitChannels
    from .weights import (
        drawAndSetSparseReservoirWeightMatrix, drawSparseWeightMatrix,
        drawAndSetSparseReservoirMaskMatrix, drawSparseMaskMatrix,
        setSparseReservoirWeightMatrix, getMaskedWeights, getWeightMatrixFromProbe
    )
