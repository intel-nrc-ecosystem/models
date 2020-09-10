"""
@desc: Include parameters of the experiment
"""
def includeExperiment(self):

    """
    Optimization and experiment
    """

    # Optimization
    self.optimizeParameters = 'reservoirExSize'  # Parameter to optimize (TODO later extend to a list)

    # Experiment
    self.seed = 1  # seed of the simulation, 'None' is random (no seed)
    self.trials = 5  # number of trials
    self.stepsPerTrial = 200  # number of steps per trial

    # Reset after trials
    self.isReset = False  # activate/deactivate network reset
    self.resetSteps = 10  # number of time steps the stop generator should be active
    self.resetRelaxation = 10  # number time steps to relax after the stop signal

    """
    Neurons and network
    """

    # Neuron and synapses
    self.voltageTau = 40  # voltage time constant (default: inf)
    self.currentTau = 2  # current time constant (default: 1)
    self.thresholdMant = 200  # membrane potential threshold for spiking
    self.refractoryDelay = 2  # refractory period for a neuron
    self.weightExponent = 0  # weight exponent (between [-8,7])

    # Network size and connections
    self.reservoirExSize = 3600 #50 #400 #3600  # size of the excitatory network
    self.reservoirInSize = None  # size of the inhibitory network
    self.reservoirInExRatio = 1./4.  # number excitatory/inhibitory neurons ratio
    self.reservoirConnProb = None  # percentage of established connections (range 0.05 - 0.1)
    self.reservoirConnPerNeuron = None  # average number of connections per neuron

    # Plasticity
    self.isHomeostasis = False  # defines if homesostatic plasticity is active or not
    self.isLearningRule = False  # defines if learning rule is active or not (true/false)
    self.learningImpulse = 5  # impulse for learning rule for x1 and y1
    self.learningTimeConstant = 10  # time constant for learning rule for x1 and y1
    self.learningRule = "2^-2*x1*y0 - 2^-2*y1*x0"  # applied learning rule
    self.learningEpoch = 4  # learning epoch for learning rule

    # Noise
    self.noiseNeuronsShare = 0.1 # share of noise neurons in relation to excitatory neurons
    self.noiseNeurons = None  # number of noise neurons (calculated if None)
    self.noiseSpikeprob = 0.05  # in average, every 1/x th time step a noise neuron fires
    self.noiseDens = 0.05 # 0.005 # connectivity between noise neurons and reservoir
    self.noiseMaxWeight = 10 #2  # maximum weight a noise connection can have

    """
    Output
    """

    # Readout training
    self.binWindowSize = 10  # number of steps the spikes should be binned backwards for every step
    #self.smoothingWd = 3  # number of neurons to the left and right which are influenced
    #self.smoothingVar = 7  # variance of the Gaussian kernel

    # Target
    self.targetFilename = ''  # file name of the file containing target data
    self.targetOffset = None  # offset of the target data (if some data in the beginning of  file shall not be used)

    # Output layer
    self.partitioningClusterSize = 10 #5  # size of clusters connected to an output neuron
    self.outputWeightValue = 4 #12 # weight for output neurons

    """
    Input
    """

    # Basic input parameters
    self.inputNumTargetNeurons = 40  # Number of target neurons which are connected to the spike generators
    self.inputShareTargetNeurons = None  # Share of target neurons in relation to excitatory reservoir size
    self.inputType = 'uniform'  # The form of the input, can be 'uniform' or 'sinus' (TODO: exponential decay, exponential rise)
    self.inputSteps = 10  # Number of steps the input is presented
    self.inputWeightExponent = 0  # weight exponent (between [-8,7])
    self.inputGenSpikeProb = 0.1  # Spiking probability of the spike generators
    self.inputOffset = 0  # Wait some time before input starts in trial

    # Topology
    self.inputIsTopology = False  # A 2D input is applied (only important if network has topology)
    self.inputShiftX = 0  # shift in X direction of input in topology (only applies if inputIsTopology is true)
    self.inputShiftY = 0  # shift in Y direction of input in topology (only applies if inputIsTopology is true)

    # Leave-n-out
    self.inputIsLeaveOut = False  # Leaves one input neuron out in every trial
    self.inputNumLeaveOut = 1  # Only applied if inputIsLeaveOut is true

    # Sequence
    self.inputIsSequence = False  # Mulitple inputs are given in a sequence within one trial
    self.inputSequenceSize = 3  # Number of inputs stimulating the network neurons in a row within one trial

    """
    Probes
    """

    # Probes
    self.isExSpikeProbe = False  # activate/deactivate spike probes for excitatory neurons
    self.isInSpikeProbe = False  # activate/deactivate spike probes for inhibitory neurons
    self.isOutSpikeProbe = False  # activate/deactivate spike probes for output neurons
    self.isWeightProbe = False  # read weights at the end of the simulation
    self.isExVoltageProbe = False  # activate/deactivate voltage probes for excitatory neurons
    self.isInVoltageProbe = False  # activate/deactivate voltage probes for inhibitory neurons
    self.isOutVoltageProbe = False  # activate/deactivate voltage probes for output neurons
    self.isExCurrentProbe = False  # activate/deactivate current probes for excitatory neurons
    self.isInCurrentProbe = False  # activate/deactivate current probes for inhibitory neurons
