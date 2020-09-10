import numpy as np
import logging
import itertools
import warnings
from scipy import sparse

"""
@desc: Generates a sinus signal (one 'hill') in given length of time steps
"""
def generateSinusInput(length):
    # Draw from a sin wave from 0 to 3,14 (one 'hill')
    probCoeff = 1 #1.5  # the higher the probability coefficient, the more activity is in the network
    probs = probCoeff * np.abs(np.sin((np.pi / length) * np.arange(length)))
    randoms = np.random.rand(length)

    # Get indices of spike
    spikeInd = np.where(randoms < probs)[0]

    # Return spikes
    return spikeInd

"""
@desc: Generates a simple input signal
"""
def generateUniformInput(length, prob=0.1):
    spikes = np.zeros((length, 1))
    randoms = np.random.rand(length)

    # Draw spikes
    for i in range(length):
        spikes[i] = (randoms[i] < prob)
    
    # Get indices of spike
    spikeTimes = np.where(spikes)[0]

    # Return spikes
    return spikeTimes
"""
@desc:  Adds an input to the network for every trial,
        either a sequence or a single input
"""
def addInput(self, *args, **kwargs):
    if self.p.inputIsSequence:
        addInputSequence(self, *args, **kwargs)
    else:
        addInputSingle(self, *args, **kwargs)

"""
@desc:  Create a sequence of inputs
NOTE a topology is currently not supported in a sequence input
"""
def addInputSequence(self):
    if self.p.inputIsTopology:
        warnings.warn("inputIsTopology is currently not supported for an input sequence and will be ignored")

    # Get number of generators
    numGens = self.p.inputNumTargetNeurons

    # Iterate over number of inputs
    for i in range(self.p.inputSequenceSize):
        # Create spike generator
        sg = self.nxNet.createSpikeGenProcess(numPorts=numGens)

        # Draw spikes for input generators for current input
        inputSpikes = drawSpikesForAllGenerators(self, numGens, offset=i*self.p.inputSteps)
        self.inputSpikes.append(inputSpikes)

        # Add spikes s to generator i
        for k, s in enumerate(inputSpikes):
            if type(s) is not list: s = s.tolist()
            sg.addSpikes(spikeInputPortNodeIds=k, spikeTimes=s)

        # Get inidices of target neurons for current input
        inputTargetNeurons = np.arange(i*self.p.inputNumTargetNeurons, (i+1)*self.p.inputNumTargetNeurons)
        self.inputTargetNeurons.append(inputTargetNeurons)

        # Connect spike generators to reservoir
        self.inputWeights = connectSpikeGenerator(self, sg, inputTargetNeurons)

    # Log that input was added
    logging.info('Input sequence was added to the network')

"""
@desc:  Connects a single input per trial to the reservoir network
@note:  If inputIsTopology is true the number of target neurons may differ
        due to rounding in getTargetNeurons() function
        therefore self.p.inputNumTargetNeurons cannot be used here,
        but instead len(self.inputTargetNeurons) must be used
@params:
        inputSpikeIndices:      indices of input spikes for spike generators
                                if not given, they are drawn (default)
        targetNeuronIndices:    indices of reservoir neurons to connect input to
                                if not given, indices are taken successively (default)
"""
def addInputSingle(self, inputSpikeIndices=[], targetNeuronIndices=[]):
    # Get inidices of target neurons if not already given
    self.inputTargetNeurons = targetNeuronIndices if len(targetNeuronIndices) else getTargetNeurons(self)

    # Get number of generators
    numGens = len(self.inputTargetNeurons)

    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=numGens)

    # Draw spikes for input generators if not already given
    self.inputSpikes = inputSpikeIndices if len(inputSpikeIndices) else drawSpikesForAllGenerators(self, numGens=numGens)

    # Add spikes s to generator i
    for i, s in enumerate(self.inputSpikes):
        if type(s) is not list: s = s.tolist()
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=s)

    # Connect spike generators to reservoir
    self.inputWeights = connectSpikeGenerator(self, sg, self.inputTargetNeurons)

    # Log that input was added
    logging.info('Input was added to the network')

"""
@desc: Draw spikes for ALL spike generators
"""
def drawSpikesForAllGenerators(self, numGens, offset=0):
    # Initialize array for spike indices
    inputSpikes = []

    # Define empty variable for combinations
    combinations = None
    # If leave out should be applied, define all possible combinations to leave one out
    if self.p.inputIsLeaveOut:
        combinations = np.array(list(itertools.combinations(np.arange(len(self.inputTargetNeurons)), self.p.inputNumLeaveOut)))

    # Iterate over target neurons to draw spikes
    for i in range(numGens):
        # Initialize array for spike indices for generator i
        spikeTimes = []

        # Defines spikes for generator i for all trials
        for k in range(self.p.trials):
            apply = True

            # If leave out should be applied, update apply boolean
            if self.p.inputIsLeaveOut:
                # Add spike times only when i != k
                apply = np.all([combinations[k, m] != i for m in range(self.p.inputNumLeaveOut)])

            # If spike generator produces input for the current trial, add it to spikeTimes
            if apply:
                off = offset + self.p.stepsPerTrial*k + self.p.resetOffset*(k+1)
                spks = drawSpikes(self, offset=off)
                spikeTimes.append(spks)

        # Add spike indices to inputSpikes array
        inputSpikes.append(list(itertools.chain(*spikeTimes)))

    return inputSpikes

"""
@desc: Draw spikes for ONE spike generator
"""
def drawSpikes(self, offset=0):
    s = []
    # Generate spikes, depending on input type
    if self.p.inputType == 'uniform':
        s = self.p.inputOffset + offset + generateUniformInput(self.p.inputSteps, prob=self.p.inputGenSpikeProb)
    if self.p.inputType == 'sinus':
        s = self.p.inputOffset + offset + generateSinusInput(self.p.inputSteps)

    return s

"""
@desc: Define target neurons in reservoir to connect generators with
"""
def getTargetNeurons(self):
    # Initialize array for target neurons
    targetNeurons = []

    # If topology should NOT be considered, just take first n neurons as target
    if not self.p.inputIsTopology:
        targetNeurons = np.arange(self.p.inputNumTargetNeurons)

    # If topology should be considered, define square input target area
    if self.p.inputIsTopology:
        # Define size and 
        targetNeuronsEdge = int(np.sqrt(self.p.inputNumTargetNeurons))
        exNeuronsEdge = int(np.sqrt(self.p.reservoirExSize))

        # Get shifts for the input area of the target neurons
        sX = self.p.inputShiftX
        sY = self.p.inputShiftY

        # Define input region in network topology and store their indices
        topology = np.zeros((exNeuronsEdge,exNeuronsEdge))
        topology[sY:sY+targetNeuronsEdge,sX:sX+targetNeuronsEdge] = 1
        targetNeurons = np.where(topology.flatten())[0]

    return targetNeurons
    
"""
@desc:  Connect spike generators with target neurons of reservoir
        Every spike generator is connected to one neuron
        Finally draws uniformly distributed weights
@params:
        spikeGenerators: nxsdk spike generators
        inputTargetNeurons: indices of reservoir target neurons
"""
def connectSpikeGenerator(self, spikeGenerators, inputTargetNeurons):
    # Creates empty mask matrix
    inputMask = np.zeros((self.p.reservoirExSize, len(inputTargetNeurons)))

    # Every generator is connected to one 
    for i, idx in enumerate(inputTargetNeurons):
        inputMask[idx,i:i+1] = 1

    # Transform to sparse matrix
    inputMask = sparse.csr_matrix(inputMask)

    # Draw weights
    inputWeights = self.drawSparseWeightMatrix(inputMask, distribution='uniform')

    # Connect generator to the reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = inputMask[fr:to, :].toarray()
        we = inputWeights[fr:to, :].toarray()
        spikeGenerators.connect(self.exReservoirChunks[i], prototype=self.genConnProto, connectionMask=ma, weight=we)

    return inputWeights
