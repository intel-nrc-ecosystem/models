import nxsdk.api.n2a as nx
import numpy as np
from scipy import sparse
import logging

"""
@desc: Removes a core from list of cores
       If number of cores is exceeded, throws an error
"""
def removeCoreFromList(self):
    # Check if cores are still available
    if len(self.cores) > 0:
        # Remove first element of array
        self.cores = np.delete(self.cores, [0])
    else:
        logging.error('Available cores exceeded, change setup parameters or rearange network')
        raise RuntimeError('Available cores exceeded, change setup parameters or rearange network.')

"""
@desc: Connects reservoir neurons
"""
def connectReservoir(self):
    # Predefine some helper variables
    nEx = self.p.reservoirExSize
    nExCores = int(np.ceil(nEx / self.p.neuronsPerCore))
    nLastExCore = nEx % self.p.neuronsPerCore  # number of excitatory neurons in last core

    nIn = self.p.reservoirInSize
    nInCores = int(np.ceil(nIn / self.p.neuronsPerCore))
    nLastInCore = nIn % self.p.neuronsPerCore  # number of inhibitory neurons in last core

    exConnProto = None
    # Create learning rule
    if self.p.isLearningRule:
        # Define learning rule
        lr = self.nxNet.createLearningRule(dw=self.p.learningRule, tEpoch=self.p.learningEpoch,
                                            x1Impulse=self.p.learningImpulse, x1TimeConstant=self.p.learningTimeConstant,
                                            y1Impulse=self.p.learningImpulse, y1TimeConstant=self.p.learningTimeConstant)
        # Define connection prototype with learning rule
        exConnProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                enableLearning=1, learningRule=lr)
    else:
        # Define connection prototype from basic network
        exConnProto = self.exConnProto

    # Define excitatory compartment prototypes and compartment groups
    # initial value is first available core id in core list
    frCore, toCore = self.cores[0], self.cores[0]+nExCores
    for i in range(frCore, toCore):
        # Excitatory compartment prototype
        exCompProto = nx.CompartmentPrototype(compartmentVoltageDecay=self.p.compartmentVoltageDecay,
                                              compartmentCurrentDecay=self.p.compartmentCurrentDecay,
                                              vThMant=self.p.thresholdMant,
                                              refractoryDelay=self.p.refractoryDelay, logicalCoreId=i,
                                              enableSpikeBackprop=self.p.isLearningRule, enableSpikeBackpropFromSelf=self.p.isLearningRule,
                                              functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
        # Calculate size of compartment: if last core has remainder, use remainder as size
        size = nLastExCore if (i == (toCore-1) and nLastExCore > 0) else self.p.neuronsPerCore
        # Excitatory compartment group
        self.exReservoirChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=exCompProto))
        # Remove used core from core list
        self.removeCoreFromList()

    # Define inhibitory compartment prototypes and compartment groups
    # initial value is first available core id in core list
    frCore, toCore = self.cores[0], self.cores[0]+nInCores
    for i in range(frCore, toCore):
        # Inhibitory compartment prototype
        inCompProto = nx.CompartmentPrototype(compartmentVoltageDecay=self.p.compartmentVoltageDecay,
                                              compartmentCurrentDecay=self.p.compartmentCurrentDecay,
                                              vThMant=self.p.thresholdMant,
                                              refractoryDelay=self.p.refractoryDelay, logicalCoreId=i,
                                              functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
        # Calculate size of compartment: if last core has remainder, use remainder as size
        size = nLastInCore if (i == (toCore-1) and nLastInCore > 0) else self.p.neuronsPerCore
        # Inhibitory compartment prototype
        self.inReservoirChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=inCompProto))
        # Remove used core from core list
        self.removeCoreFromList()

    # Interconnect excitatory and inhibitory network chunks
    connectNetworkChunks(self, fromChunks=self.exReservoirChunks, toChunks=self.exReservoirChunks, mask=self.initialMasks.exex, weights=self.initialWeights.exex, prototype=exConnProto) #store=True, prototype=exConnProto)
    connectNetworkChunks(self, fromChunks=self.inReservoirChunks, toChunks=self.inReservoirChunks, mask=self.initialMasks.inin, weights=self.initialWeights.inin, prototype=self.inConnProto)
    connectNetworkChunks(self, fromChunks=self.exReservoirChunks, toChunks=self.inReservoirChunks, mask=self.initialMasks.exin, weights=self.initialWeights.exin, prototype=self.exConnProto)
    connectNetworkChunks(self, fromChunks=self.inReservoirChunks, toChunks=self.exReservoirChunks, mask=self.initialMasks.inex, weights=self.initialWeights.inex, prototype=self.inConnProto)

    # Log that all cores are interconnected
    logging.info('All cores are sucessfully interconnected')

"""
@desc: Connects reservoir neurons
"""
def connectOutput(self):
    # Predefine some helper variables
    nOut = self.p.numOutputNeurons
    nOutCores = int(np.ceil(nOut / self.p.neuronsPerCore))
    nLastOutCore = nOut % self.p.neuronsPerCore  # number of ouput neurons in last core

    # Define output compartment prototypes and compartment groups
    # initial value is first available core id in core list
    frCore, toCore = self.cores[0], self.cores[0]+nOutCores
    for i in range(frCore, toCore):
        # Output compartment prototype
        outCompProto = nx.CompartmentPrototype(compartmentVoltageDecay=self.p.compartmentVoltageDecay,
                                               compartmentCurrentDecay=self.p.compartmentCurrentDecay,
                                               vThMant=self.p.thresholdMant,
                                               refractoryDelay=self.p.refractoryDelay, logicalCoreId=i,
                                               functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
        # Calculate size of compartment: if last core has remainder, use remainder as size
        size = nLastOutCore if (i == (toCore-1) and nLastOutCore > 0) else self.p.neuronsPerCore
        # Output layer compartment prototype
        self.outputLayerChunks.append(self.nxNet.createCompartmentGroup(size=size, prototype=outCompProto))
        # Remove used core from core list
        self.removeCoreFromList()

    # Connect excitatory neurons to output layer
    connectNetworkChunks(self, fromChunks=self.exReservoirChunks, toChunks=self.outputLayerChunks, mask=self.outputMask, weights=self.outputWeights, prototype=self.exConnProto)

"""
@desc: Interconnect all network chunks (basically interconnects cores)
"""
def connectNetworkChunks(self, fromChunks, toChunks, mask, weights, store=False, **connectionArgs):
    nCoresFrom = len(fromChunks)
    nCoresTo = len(toChunks)
    nPerCore = self.p.neuronsPerCore

    for i in range(nCoresFrom):
        # Get indices for chunk from outer loop
        ifr, ito = i*nPerCore, (i+1)*nPerCore

        tmp = []
        for j in range(nCoresTo):
            # Get indices for chunk from inner loop
            jfr, jto = j*nPerCore, (j+1)*nPerCore

            # Define chunk from sparse matrix and transform to numpy array
            ma = mask[jfr:jto, ifr:ito].toarray()
            we = weights[jfr:jto, ifr:ito].toarray()

            # Connect network chunks
            conn = fromChunks[i].connect(toChunks[j], connectionMask=ma, weight=we, **connectionArgs)
            if store:
                tmp.append(conn)
        
        # If store flag is set, also store all connection chunks
        if store:
            self.connectionChunks.append(tmp)
