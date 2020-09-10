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
    def __init__(self, name='', parameters={}):
        # Parameters
        self.p = Parameters(update = self.updateParameters(parameters))

        self.net = None

        # Instantiate system singleton and add datalog object
        self.system = System.instance()
        datalog = Datalog(self.p, name=name)
        self.system.setDatalog(datalog)

        # Instantiate utils and plot
        self.utils = Utils.instance(parameters=self.p)
        self.plot = Plot(self)

    """
    @desc: Overwrite parameters for this experiment
    """
    def updateParameters(self, jupP={}):
        expP = {
            # Experiment
            'seed': 1,  # Random seed
            'trials': 10,  # Number of trials
            'stepsPerTrial': 60,  # Number of simulation steps for every trial
            # Neurons
            'refractoryDelay': 2, # Refactory period
            'voltageTau': 100,  # Voltage time constant
            'currentTau': 5,  # Current time constant
            'thresholdMant': 1200,  # Spiking threshold for membrane potential
            # Network
            'reservoirExSize': 400,  # Number of excitatory neurons
            'reservoirConnPerNeuron': 35,  # Number of connections per neuron
            'isLearningRule': True,  # Apply a learning rule
            'learningRule': '2^-2*x1*y0 - 2^-2*y1*x0 + 2^-4*x1*y1*y0 - 2^-3*y0*w*w',  # Defines the learning rule
            # Input
            'inputIsSequence': True,  # Activates sequence input
            'inputSequenceSize': 3,  # Number of input clusters in sequence
            'inputSteps': 20,  # Number of steps the trace input should drive the network
            'inputGenSpikeProb': 0.8,  # Probability of spike for the generator
            'inputNumTargetNeurons': 40,  # Number of neurons activated by the input
            # Probes
            'isExSpikeProbe': True,  # Probe excitatory spikes
            'isInSpikeProbe': True,  # Probe inhibitory spikes
            'isWeightProbe': True  # Probe weight matrix at the end of the simulation
        }

        # Parameters from jupyter notebook overwrite parameters from experiment definition
        return { **expP, **jupP}

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
        self.net.addInput()

        # Add background noise
        #self.net.addNoiseGenerator()

        # Add Probes
        self.net.addProbes()

    """
    @desc: Run experiment
    """
    def run(self):
        # Run network
        self.net.run()

    """
    @desc: Draw mask and weights
    """
    def drawMaskAndWeights(self):
        # Draw and store mask matrix
        mask = self.net.drawAndSetSparseReservoirMaskMatrix()

        # Draw and store weight matrix
        self.net.drawAndSetSparseReservoirWeightMatrix(mask)

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
        