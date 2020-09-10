import nxsdk.api.n2a as nx
import numpy as np
import logging

"""
@desc: Post processing of probes
"""
def postProcessing(self):
    # Calculate spikes from probes activities
    #self.exSpikeTrains = self.utils.getSpikesFromActivity(self.exActivityProbes)
    #self.inSpikeTrains = self.utils.getSpikesFromActivity(self.inActivityProbes)

    # Combine spike probes from all chunks together for excitatory neurons
    if self.p.isExSpikeProbe:
        spks = []
        for i in range(len(self.exSpikeProbes)):
            spks.append(self.exSpikeProbes[i].data)
        self.exSpikeTrains = np.vstack(spks)

    # Combine spike probes from all chunks together for inhibitory neurons
    if self.p.isInSpikeProbe:
        spks = []
        for i in range(len(self.inSpikeProbes)):
            spks.append(self.inSpikeProbes[i].data)
        self.inSpikeTrains = np.vstack(spks)

    # Combine spike probes from all chunks together for inhibitory neurons
    if self.p.isOutSpikeProbe:
        spks = []
        for i in range(len(self.outSpikeProbes)):
            spks.append(self.outSpikeProbes[i].data)
        self.outSpikeTrains = np.vstack(spks)

    # Combine spike probes from all chunks together for inhibitory neurons
    if self.p.isOutVoltageProbe:
        spks = []
        for i in range(len(self.outVoltageProbes)):
            spks.append(self.outVoltageProbes[i].data)
        self.outVoltageTrains = np.vstack(spks)

    # Recombine all weights from probe chunks together to a matrix again
    if self.p.isWeightProbe:
        self.trainedWeightsExex = self.utils.recombineExWeightMatrix(self.initialWeights.exex, self.weightProbes)

    # Log that post processing has finished
    logging.info('Post processing succesfully completed')

"""
@desc: Remove offsets from data
"""
def condenseSpikeProbes(self, raw):
    # Get total offset
    offset = self.p.resetOffset + self.p.inputOffset

    # Define empty data list and iterate over all trials
    data = []
    for i in range(self.p.trials):
        # Get 'from' and 'to' position of every trial
        fr, to = (i+1)*offset+i*self.p.stepsPerTrial, (i+1)*offset+(i+1)*self.p.stepsPerTrial
        # Cut every trial and append it to data
        data.append(raw[:,fr:to])
    
    # Transform list to data array and return
    return np.array(data)

"""
@desc: Add probing
"""
def addProbes(self):
    # Add voltage probe for excitatory network
    if self.p.isExVoltageProbe:
        for idx, net in enumerate(self.exReservoirChunks):
            self.exVoltageProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE ])[0])

    # Add voltage probe for inhibitory network
    if self.p.isInVoltageProbe:
        for idx, net in enumerate(self.inReservoirChunks):
            self.inVoltageProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE ])[0])

    # Add voltage probe for output neurons
    if self.p.isOutVoltageProbe:
        for idx, net in enumerate(self.outputLayerChunks):
            self.outVoltageProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_VOLTAGE ])[0])

    # Add current probe for excitatory network
    if self.p.isExCurrentProbe:
        for idx, net in enumerate(self.exReservoirChunks):
            self.exCurrentProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_CURRENT ])[0])

    # Add current probe for inhibitory network
    if self.p.isInCurrentProbe:
        for idx, net in enumerate(self.inReservoirChunks):
            self.inCurrentProbes.append(net.probe([nx.ProbeParameter.COMPARTMENT_CURRENT ])[0])
    
    # Probe weights
    if self.p.isWeightProbe:
        probeCond = nx.IntervalProbeCondition(tStart=self.p.totalSteps-1, dt=self.p.totalSteps)
        #probeCond = nx.IntervalProbeCondition(tStart=self.p.stepsPerIteration-1, dt=self.p.stepsPerIteration)
        n, m = np.shape(self.connectionChunks)
        for i in range(n):
            tmp = []
            for j in range(m):
                tmp.append(self.connectionChunks[i][j].probe([nx.ProbeParameter.SYNAPSE_WEIGHT], probeConditions=[probeCond]))
            self.weightProbes.append(tmp)

    # Add spike probe for excitatory network
    if self.p.isExSpikeProbe:
        #probeCond = nx.SpikeProbeCondition(tStart=1, dt=5)
        for net in self.exReservoirChunks:
            self.exSpikeProbes.append(net.probe([nx.ProbeParameter.SPIKE])[0])#, probeConditions=[probeCond])[0])

    # Add spike probe for excitatory network
    if self.p.isInSpikeProbe:
        #probeCond = nx.SpikeProbeCondition(tStart=1, dt=5)
        for net in self.inReservoirChunks:
            self.inSpikeProbes.append(net.probe([nx.ProbeParameter.SPIKE])[0])#, probeConditions=[probeCond])[0])

    # Add spike probe for output neurons
    if self.p.isOutSpikeProbe:
        for net in self.outputLayerChunks:
            self.outSpikeProbes.append(net.probe([nx.ProbeParameter.SPIKE])[0])

    # Log that probes were added to network
    logging.info('Probes added to Loihi network')
