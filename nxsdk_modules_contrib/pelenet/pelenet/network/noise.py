"""
@desc: Adds a generator which produces random spikes and
        connects it to the excitatory reservoir neurons
"""
def addNoiseGenerator(self):
    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.noiseNeurons)

    # Create random spikes
    noise = np.random.rand(self.p.noiseNeurons, self.p.totalSteps)
    noise[noise < (1-self.p.noiseSpikeprob)] = 0
    noise[noise >= (1-self.p.noiseSpikeprob)] = 1

    # Store spikes in object
    self.noiseSpikes = noise.astype(int)

    # Add spike times to spike generator
    for i in range(self.p.noiseNeurons):
        spikes = np.where(self.noiseSpikes[i,:])[0].tolist()
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikes)

    # Create mask for noise/reservoir connections
    noiseMask = self.drawSparseMaskMatrix(self.p.noiseDens, self.p.reservoirExSize, self.p.noiseNeurons)

    # Create weights for noise/reservoir connections between -noiseMaxWeight and +noiseMaxWeight
    randoms = ((np.random.rand(self.p.reservoirExSize, self.p.noiseNeurons)*2*self.p.noiseMaxWeight) - self.p.noiseMaxWeight)
    self.noiseWeights = sparse.csr_matrix(np.round(self.p.noiseMaxWeight*randoms).astype(int))
    #sign = np.random.rand(self.p.reservoirExSize, self.p.noiseNeurons)
    #sign[sign < 0.5] = -1
    #sign[sign >= 0.5] = 1
    #self.noiseWeights = self.drawSparseWeightMatrix(noiseMask).multiply(sign).tocsr()

    # Connect noise network to the reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = noiseMask[fr:to, :].toarray()
        we = self.noiseWeights[fr:to, :].toarray()
        sg.connect(self.exReservoirChunks[i], prototype=self.mixedConnProto, connectionMask=ma, weight=we)
    
    # Log that background noise was added
    logging.info('Background noise was added to the network')

"""
@desc: Create input spiking generator to add a constant signal,
        the input is connected to a share of the reservoir network,
        an excitatory connection prototype is used
"""
def addConstantGenerator(self):
    # Create spike generator
    sg = self.nxNet.createSpikeGenProcess(numPorts=self.p.constGens)

    constSpikes = []
    for i in range(self.p.constGens):
        # Generate spikes for one training step
        constSpikesInd = self.generateInputSignal(self.p.totalSteps, prob=self.p.constSpikeProb, start=0)
        
        # Add all spikes to spike generator
        sg.addSpikes(spikeInputPortNodeIds=i, spikeTimes=constSpikesInd)

        # Store const input in object
        constSpikes.append(constSpikesInd)
    
    self.constSpikes.append(np.array(constSpikes))

    # Connect generator to the reservoir network
    startNeuron = 0
    endNeuron = (self.p.constSize-1)

    # Sample mask for constant input
    constMask = self.drawSparseMaskMatrix(self.p.constDens, self.p.reservoirExSize, self.p.constGens)
    constMask[endNeuron:, :] = 0  # set all mask values behind last neuron of cluster to zero

    # Sample weights for constant input
    constWeights = self.drawSparseWeightMatrix(constMask)

    # Connect generator to the excitatory reservoir network
    for i in range(len(self.exReservoirChunks)):
        fr, to = i*self.p.neuronsPerCore, (i+1)*self.p.neuronsPerCore
        ma = constMask[fr:to, :].toarray()
        we = constWeights[fr:to, :].toarray()
        sg.connect(self.exReservoirChunks[i], prototype=self.exConnProto, connectionMask=ma, weight=we)

    #self.constMasks.append(constMask)
    #self.constWeights.append(constWeights)