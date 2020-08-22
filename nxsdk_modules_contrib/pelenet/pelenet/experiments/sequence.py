# Official modules
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tabulate import tabulate
from copy import deepcopy

# Pelenet modules
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
class SequenceExperiment():

    """
    @desc: Initiates the experiment
    """
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

        # Define some further variables
        self.targetFunctions = np.zeros((self.p.traceClusters, self.p.traceSteps))
        self.estimatedParameters = None
        self.parameterMean = None
        self.trainInNet = None
        self.testInNet = None
        self.testSpikes = None
        self.trainSpikes = None

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self):
        return {
            # Experiment
            'trials': 5,
            'stepsPerTrial': 90,
            # Network
            'reservoirExSize': 2048,
            'reservoirConnProb': None,
            'reservoirConnPerNeuron': 45,
            'isLearningRule': True,
            'learningRule': '2^-2*x1*y0 - 2^-2*y1*x0 + 2^-4*x1*y1*y0 - 2^-3*y0*w*w',
            # Probes
            'isExSpikeProbe': True
        }

    """
    @desc: Build reservoir network with all parts (input, output, noise, etc.)
    """
    def build(self):
        # Instanciate innate network
        self.net = ReservoirNetwork(self.p)

        # Draw anisotropic mask and weights
        self.drawMaskAndWeights()

        # Connect ex-in reservoir
        self.net.connectReservoir()

        # Add cue
        self.net.addTraceGenerator(0)
        self.net.addTraceGenerator(1)
        self.net.addTraceGenerator(2)

        # Add background noise
        #self.net.addNoiseGenerator()

        # Build the network structure
        self.net.build()

    """
    @desc: Draw mask and weights
    """
    def drawMaskAndWeights(self):
        # Draw and store mask matrix
        mask = self.net.drawAndSetSparseReservoirMaskMatrix(self.p.reservoirConnProb, self.p.reservoirSize, self.p.reservoirSize, avoidSelf=True)

        # Draw and store weight matrix
        self.net.drawAndSetSparseReservoirWeightMatrix(mask)

    """
    @desc: Train sequence
    """
    def trainSequence(self):
        # Instanciate innate network
        self.trainInNet = TrainNet(parameters=self.p)

        # Build the network structure
        self.trainInNet.build()

        # Print and plot statistics about network structure
        self.trainInNet.plot.weightDistribution()  # plot histogram of weights and calc spectral radius

        # Plot weight matrix
        self.trainInNet.plot.excitatoryWeightMatrix()
        
        # Run network
        self.trainInNet.run()

        # Calculate spikes from probes activities
        self.trainInNet.exSpikeTrains = self.utils.getSpikesFromActivity(self.trainInNet.exActivityTrains)
        self.trainInNet.inSpikeTrains = self.utils.getSpikesFromActivity(self.trainInNet.inActivityTrains)
    
    """
    @desc: Test sequence
    """
    def testSequence(self):
        # Instantiate parameters and turn off learning rule
        testParameters = Parameters()
        setattr(testParameters, 'learningRuleStatus', False)  # deactivate learning rule

        # Instanciate innate network
        self.testInNet = TestNet(parameters=testParameters)

        # Copy weights and masks from training net initialization
        self.testInNet.initialMasks = deepcopy(self.trainInNet.initialMasks)
        self.testInNet.initialWeights = deepcopy(self.trainInNet.initialWeights)

        # Copy cue spikes
        self.testInNet.cueSpikes = deepcopy(self.trainInNet.cueSpikes)

        # Replace initial excitatory weight matrix with trained matrix
        exex = self.trainInNet.getWeightMatrixFromProbe()
        insize = self.p.traceClusters*self.p.traceClusterSize
        #exex[insize:, 0:insize] = (0.5*exex[insize:, 0:insize]).astype(int)
        #exex[0:insize, insize:] = (0.5*exex[0:insize, insize:]).astype(int)
        #exex[insize:, insize:] = (0.5*exex[insize:, insize:]).astype(int)
        #exex[0:insize, 0:insize] = 7*exex[0:insize, 0:insize]
        self.testInNet.initialWeights.exex = exex
        # TODO: 'zoom in' in spike train, what happens?

        # Build the network structure and use initial matrices
        self.testInNet.build(reservoirMasks=self.testInNet.initialMasks, reservoirWeights=self.testInNet.initialWeights)

        # Plot weight matrix
        self.testInNet.plot.excitatoryWeightMatrix()

        # Run the network
        self.testInNet.run()

    """
    @desc: Define function to learn as ouput
    @params:
            clusterIndex: index of cluster the function is defined for
            type: 'sin', 'revsin', 'lin'
    """
    def setTargetFunction(self, clusterIndex, type):
        nTs = self.p.traceSteps

        # Define function values
        if type == 'sin': self.targetFunctions[clusterIndex,:] = 0.5+0.5*np.sin((np.pi/(0.5*nTs))*np.arange(nTs))
        elif type == 'revsin': self.targetFunctions[clusterIndex,:] = 0.5-0.5*np.sin((np.pi/(0.5*nTs))*np.arange(nTs))
        elif type == 'lin': self.targetFunctions[clusterIndex,:] = np.concatenate((0.5-(1/nTs)*np.arange(nTs/2), (1/nTs)*np.arange(nTs/2)))
        else: raise ValueError('Chosen function type is not available')
    
    """
    TODO: Shift to plots/misc.py
    @desc: Plots either target function or estimated function
    """
    def plotActionSequenceFunction(self, y, title="Target function"):
        # Plot function
        plt.figure(figsize=(16, 4))
        plt.title(title)
        plt.xlabel('Time')
        for i in range(self.p.traceClusters):
            fr, to = i*self.p.traceSteps, (i+1)*self.p.traceSteps
            plt.plot(np.arange(fr, to), y[i], label="Cluster"+str(i))
        plt.legend()
        p = plt.show()

    """
    @desc: Call several function in order to evaluate results
    """
    def evaluateExperiment(self):
        idx = self.p.cueSteps + self.p.relaxationSteps

        # Prepare spikes for evaluation
        self.trainSpikes = np.array([net.reservoirProbe[2].data for net in self.trainNets])
        trainSpikesMean = np.mean(self.trainSpikes, axis=0)
        self.testSpikes = self.testNet.reservoirProbe[2].data

        # Plot autocorrelation function
        self.plot.autocorrelation(trainSpikesMean[:,idx:], numNeuronsToShow=1)
        # TODO is it a good idea to mean over trials? maybe use index selection like for fano factors

        # Plot crosscorrelation function
        self.plot.crosscorrelation(self.testSpikes)

        # Plot spike missmatch
        self.plot.spikesMissmatch(self.trainSpikes[i,:,fr:to], self.testSpikes[:,fr:to])

        # Plot fano factors of spike counts (test spikes)
        self.plot.ffSpikeCounts(self.testSpikes, neuronIdx = [1,5,10,15])

        # Plot first 2 components of PCA of all spikes 
        self.plot.pca(self.testSpikes)

        # Evalulate assembly weights
        self.evaluateAssemblyWeights()

        # Plot pre synaptic trace
        self.trainNet.plot.preSynapticTrace()

    """
    @desc: Plot assembly weights and sum of cluster weights
    """
    def evaluateAssemblyWeights(self):
        numClusters = self.testInNet.p.traceClusters
        clusterSize = self.testInNet.p.traceClusterSize
        assemblySize = numClusters*clusterSize
        clusterWeightsAfter = self.testInNet.initialWeights.exex[:assemblySize,:assemblySize]

        # Plot assembly weights
        plt.figure(figsize=(6, 6))
        plt.imshow(clusterWeightsAfter, vmin=0, vmax=20, interpolation=None)
        plt.title('Cluster Weights')
        p = plt.colorbar()

        # Cluster sums before
        print('Cluster weights before training')
        clusterSumBefore = self.getClusterSums(self.trainInNet.initialWeights.exex[:assemblySize,:assemblySize])
        print(clusterSumBefore)

        # Cluster sums after
        clusterSumAfter = self.getClusterSums(clusterWeightsAfter)
        print('Cluster weights after training')
        print(clusterSumAfter)
        print('Cluster weights after training (normalized)')
        print((clusterSumAfter/np.sum(clusterSumAfter)*1000).astype(int))

    """
    @desc: Calclulate sum of clusters from given matrix
    """
    def getClusterSums(self, matrix):
        numClusters = self.testInNet.p.traceClusters
        clusterSize = self.testInNet.p.traceClusterSize

        clusterSums = np.zeros((numClusters, numClusters))
        for i in range(numClusters):
            iFrom = i*clusterSize
            iTo = (i+1)*clusterSize
            for j in range(numClusters):
                jFrom = j*clusterSize
                jTo = (j+1)*clusterSize
                clusterSums[i,j] = np.sum(matrix[iFrom:iTo, jFrom:jTo])
        return clusterSums.astype(int)

    """
    TODO: Shift (or shift partly?) to utils/misc.py
    @desc: Do OLM neurons specified by mask
    @params:
            mask: masks which neurons to choose
            clusterIndex: index of cluster (e.g. A: 0, B: 1, etc.)
    """
    def trainOLM(self, maskOthers, title=None, smoothing=False):
        
        # Get spikes
        x = self.testInNet.reservoirProbe.data.T
        
        # Filter activity for specific cluster
        nTs = self.p.traceSteps  # length of cluster activation (e.g. 30)
        nC = self.p.traceClusters  # number of trace clusters (e.g. 3)
        nCs = self.p.traceClusterSize
        nTc = self.p.traceClusters * self.p.testingIterations  # number of activated clusters

        maskClustersProto = np.zeros(nC * nCs).astype(bool)
        estimateAll = []
        performanceAll = []
        for clusterIndex in range(nC):
            # Set target function
            y = self.targetFunctions[clusterIndex]

            # Define mask
            maskClusters = deepcopy(maskClustersProto)
            maskClusters[clusterIndex*nCs:(clusterIndex+1)*nCs] = True
            mask = np.concatenate((maskClusters, maskOthers[clusterIndex]))

            # Filter activity for specific cluster
            clusters = np.array([x[i*nTs:(i+1)*nTs,mask] for i in range(clusterIndex,nTc,nC)])

            # Train the parameters
            paramsArray = []
            for iteration in range(self.p.testingIterations - 1,self.p.testingIterations):  # only last cluster used for estimation
            #for iteration in range(self.p.testingRelaxation, self.p.testingIterations):
                xc = self.utils.getSmoothSpikes(clusters[iteration]) if smoothing else clusters[iteration]
                model = sm.OLS(y, xc)
                paramsArray.append(model.fit().params)
                #paramsArray.append(model.fit_regularized().params)

            # Mean params over iterations and store in list
            paramsMean = np.mean(paramsArray, axis=0)

            # Estimate target function
            estIdx = self.p.testingIterations - 1  # index of cluster to estimate on
            xe = self.utils.getSmoothSpikes(clusters[estIdx]) if smoothing else clusters[estIdx]
            estimate = np.dot(xe, paramsMean)
            estimateAll.append(estimate)

            # Calculate performance
            performance = np.mean(np.square(y - estimate))
            #performance = None
            performanceAll.append(performance)

        # Lists to arrays
        estimateAll = np.array(estimateAll)
        performanceAll = np.array(performanceAll)

        # Plot estimated function sequence for all actions
        performanceString = " ".join([str(np.round(p*1000)/1000) for p in performanceAll])
        #title = title + ", MSE: " + performanceString
        self.plotActionSequenceFunction(estimateAll, title=title)
        