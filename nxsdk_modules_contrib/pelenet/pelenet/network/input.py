import numpy as np
import logging
import itertools

"""
TODO: Shift to utils/misc.py
@desc: Generates a sinus signal (one 'hill') in given length of time steps
"""
def generateSinSignal(length, start=0):
    # Draw from a sin wave from 0 to 3,14 (one 'hill')
    probCoeff = 1 #1.5  # the higher the probability coefficient, the more activity is in the network
    probs = probCoeff * np.abs(np.sin((np.pi / length) * np.arange(length)))
    randoms = np.random.rand(length)

    # Get indices of spike
    spikeInd = np.where(randoms < probs)[0]
    # Shift sin signal by 'start' and return spikes
    return (spikeInd + start)

"""
TODO: Shift to utils/misc.py
@desc: Generates a simple input signal
        Respects tEpoch of STDP learning rule
"""
def generateInputSignal(length, prob=0.1, start=0):
    spikes = np.zeros((length, 1))
    refCtr = 0  # initialize refractory value
    randoms = np.random.rand(length)

    for i in range(length):
        spikes[i] = (randoms[i] < prob) and refCtr <= 0
        # After spike, set refractory value
        #if spikes[i]:
        #    refCtr = self.p.learningEpoch + 1
        # Reduce refractory value by one
        #refCtr -= 1

    # Get indices of spike
    spikeTimes = np.where(spikes)[0]
    # Shift sin signal by 'start' and return spikes
    return (spikeTimes + start)

# """
# @desc: Create input spiking generator to add a cue signal,
#         the input is connected to the reservoir network,
#         an excitatory connection prototype is used
# """
# def addCueGenerator(self, inputSpikes = None):
#     # Create spike generator
#     sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.cueGens)

#     cueSpikes = []
#     for i in range(self.p.cueGens):
#         # Generate spikes and add them to spike generator
#         cueSpikesInd = None
#         # If input spikes are not given
#         if inputSpikes is None:
#             cueSpikesInd = generateInputSignal(self.p.cueSteps, prob=self.p.cueSpikeProb) #generateSinSignal(self.p.cueSteps)
#             # Store cue input in object
#             cueSpikes.append(cueSpikesInd)
#         # If input spikes are given
#         else:
#             cueSpikesInd = inputSpikes[i]
        
#         sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=cueSpikesInd.tolist())

#     if len(self.cueSpikes) == 0:
#         self.cueSpikes = np.array(cueSpikes)  # If train, store generated spikes

#     # Define mask
#     cueMask = self.drawSparseMaskMatrix(self.p.cueDens, self.p.reservoirExSize, self.p.cueGens)

#     cueSize = int(np.sqrt(self.p.cuePatchNeurons))
#     exNeuronsTopSize = int(np.sqrt(self.p.reservoirExSize))

#     #cueMask = np.zeros((cueSize, cueSize))
#     #cueMask[self.p.cueSize:, :] = 0  # set all mas values behind last neuron of cue input to zero

#     # Set all values zero which are not part of the patch
#     topology = np.ones((exNeuronsTopSize,exNeuronsTopSize))
#     shift = self.p.cuePatchNeuronsShift
#     topology[shift:shift+cueSize,shift:shift+cueSize] = 0
#     #topology[0,0] = 1
#     idc = np.where(topology.flatten())[0]
#     cueMask[idc,:] = 0

#     # Define weights
#     cueWeights = self.p.cueMaxWeight*np.random.rand(self.p.reservoirExSize, self.p.cueGens)

#     # Connect generator to the reservoir network
#     for i in range(len(self.exReservoirChunks)):
#         fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
#         ma = cueMask[fr:to, :]
#         we = cueWeights[fr:to, :]
#         sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)
    
#     self.cueWeights = self.getMaskedWeights(cueWeights, cueMask)

#     # Log that cue generator was added
#     logging.info('Cue generator was added to the network')

# """
# @desc: Create input spiking generator to add a cue signal,
#         the input is connected to the reservoir network,
#         an excitatory connection prototype is used
# """
# def addRepeatedCueGenerator(self):
#     # Create spike generator
#     sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.cueGens)

#     for i in range(self.p.cueGens):
#         # Generate spikes for spike current generator
#         spikes = (np.random.rand(self.p.cueSteps) < self.p.cueSpikeProb)
#         # Get indices from spikes
#         cueSpikesInd = []
#         for j in range(self.p.trials):
#             # Draw neurons to flip with probability flipProb
#             flips = (np.random.rand(self.p.cueSteps) < self.p.flipProb)
#             # Apply flips to cue input
#             noisedSpikes = np.logical_xor(spikes, flips)
#             # Transform to event indices
#             noisedIndices = np.where(noisedSpikes)[0] + self.p.trialSteps*j + self.p.resetOffset*(j+1)
#             cueSpikesInd.append(noisedIndices)

#         self.cueSpikes.append(list(itertools.chain(*cueSpikesInd)))
            
#         # Add spikes indices to current spike generator
#         sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=list(itertools.chain(*cueSpikesInd)))

#     # Define mask
#     cueMask = self.drawSparseMaskMatrix(self.p.cueDens, self.p.reservoirExSize, self.p.cueGens)

#     cueSize = int(np.sqrt(self.p.cuePatchNeurons))
#     exNeuronsTopSize = int(np.sqrt(self.p.reservoirExSize))

#     #cueMask = np.zeros((cueSize, cueSize))
#     #cueMask[self.p.cueSize:, :] = 0  # set all mas values behind last neuron of cue input to zero

#     # Set all values zero which are not part of the patch
#     shift = self.p.cuePatchNeuronsShift
#     topology = np.ones((exNeuronsTopSize,exNeuronsTopSize))
#     topology[shift:shift+cueSize,shift:shift+cueSize] = 0
#     #topology[0,0] = 1
#     idc = np.where(topology.flatten())[0]
#     cueMask[idc,:] = 0

#     # Define weights
#     cueWeights = self.p.patchMaxWeight*np.random.rand(self.p.reservoirExSize, self.p.cueGens)

#     # Connect generator to the reservoir network
#     for i in range(len(self.exReservoirChunks)):
#         fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
#         ma = cueMask[fr:to, :]
#         we = cueWeights[fr:to, :]
#         sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)
    
#     self.cueWeights = self.getMaskedWeights(cueWeights, cueMask)

#     # Log that cue generator was added
#     logging.info('Cue generator was added to the network')

"""
@desc: Create input spiking generator to add a patch signal,
        the input is connected to the reservoir network,
        an excitatory connection prototype is used
"""
def addRepeatedPatchGenerator(self, idc=None):
    patchGens = int(self.p.patchGensPerNeuron*self.p.patchNeurons)

    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=patchGens)

    combinations = np.array(list(itertools.combinations(np.arange(self.p.patchNeurons), self.p.patchMissingNeurons)))

    # Initialize counter
    cnt = 0
    # Iterate over patch neurons
    for i in range(self.p.patchNeurons):
        
        spikeTimes = []
        # Iterate over trials
        for k in range(self.p.trials):
            # Add spike times only when i != k
            apply = np.all([combinations[k, m] != i for m in range(self.p.patchMissingNeurons)])
            # If patch neuron indices are given, add spikes to all patch neurons
            if idc is not None:
                apply = True

            if (apply):
                spks = np.arange(self.p.inputSteps) + self.p.trialSteps*k + self.p.resetOffset*(k+1)
                spikeTimes.append(spks)

        spikeTimes = list(itertools.chain(*spikeTimes))
            
        # Iterate over generators
        for j in range(self.p.patchGensPerNeuron):
            # Add spikes indices to current spike generator
            sg.addSpikes(spikeInputPortNodeIds=cnt, spikeTimes=spikeTimes)
            # Increase counter
            cnt += 1

        # Add spike indices to patchSpikes array
        self.patchSpikes.append(spikeTimes)

    if idc is None:
        patchSize = int(np.sqrt(self.p.patchNeurons))
        exNeuronsTopSize = int(np.sqrt(self.p.reservoirExSize))

        #patchMask = np.zeros((patchSize, patchSize))
        #patchMask[self.p.patchSize:, :] = 0  # set all mas values behind last neuron of patch input to zero

        # Set all values zero which are not part of the patch
        shiftX = self.p.patchNeuronsShiftX
        shiftY = self.p.patchNeuronsShiftY
        #shiftX = 44
        #shiftY = 24
        topology = np.zeros((exNeuronsTopSize,exNeuronsTopSize))
        topology[shiftY:shiftY+patchSize,shiftX:shiftX+patchSize] = 1
        #topology[0,0] = 1
        idc = np.where(topology.flatten())[0]

        # In every trial remove another 
        #self.idc = idc
    
    # Store patch neurons
    self.patchNeurons = idc

    # Define mask for connections
    patchMask = np.zeros((self.p.reservoirExSize, patchGens))
    for i, idx in enumerate(idc):
        fr, to = i*self.p.patchGensPerNeuron, (i+1)*self.p.patchGensPerNeuron
        patchMask[idx,fr:to] = 1

    # Define weights
    self.patchWeights = patchMask*self.p.patchMaxWeight

    # Connect generator to the reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = patchMask[fr:to, :]
        we = self.patchWeights[fr:to, :]
        sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)

    # Log that patch generator was added
    logging.info('Cue generator was added to the network')

"""
@desc: Create input spiking generator to add a trace signal,
        the input is connected to a share of the reservoir network,
        an excitatory connection prototype is used
"""
def addTraceGenerator(self, clusterIdx):
    start = self.p.inputOffset + clusterIdx*self.p.traceSteps
    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.traceGens)

    traceSpikes = []
    for i in range(self.p.traceGens):
        # Generate spikes for one training step
        #traceSpikesInd = generateSinSignal(self.p.traceSteps, start)
        traceSpikesInd = generateInputSignal(self.p.traceSteps, prob=self.p.traceSpikeProb, start=start)

        # Multiply spikes to all training steps
        spikeRange = range(0, self.p.totalSteps, self.p.totalTrialSteps)
        traceSpikesInds = np.ndarray.flatten(np.array([traceSpikesInd + i for i in spikeRange]))
        
        # Add all spikes to spike generator
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=traceSpikesInds.tolist())

        # Store trace input in object
        traceSpikes.append(traceSpikesInds)
    
    self.traceSpikes.append(np.array(traceSpikes))

    # Connect generator to the reservoir network
    startNeuron = clusterIdx*self.p.traceClusterSize + self.p.constSize
    endNeuron = startNeuron+(self.p.traceClusterSize-1)

    #traceMask = np.zeros((self.p.reservoirExSize, self.p.traceGens)).astype(int)
    #traceMask[startNeuron:endNeuron, :] = 1
    #traceMask = sparse.csr_matrix(traceMask)
    traceMask = self.drawSparseMaskMatrix(self.p.traceDens, self.p.reservoirExSize, self.p.traceGens)
    traceMask[endNeuron:, :] = 0  # set all mas values behind last neuron of cluster to zero

    #traceWeights = self.p.traceMaxWeight*np.random.rand(self.p.reservoirExSize, self.p.traceGens)
    traceWeights = self.drawSparseWeightMatrix(traceMask)

    # Connect generator to the excitatory reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = traceMask[fr:to, :].toarray()
        we = traceWeights[fr:to, :].toarray()
        sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)

    self.traceMasks.append(traceMask)
    #self.traceWeights.append(self.getMaskedWeights(traceWeights, traceMask))
    self.traceWeights.append(traceWeights)