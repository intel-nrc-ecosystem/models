import numpy as np
import scipy.linalg as la
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

"""
@desc: From activity probe, calculate spike patterns
"""
def getSpikesFromActivity(self, activityProbes):
    # Get number of probes (equals number of used cores)
    numProbes = np.shape(activityProbes)[0]
    # Concatenate all probes
    activityTrain = []
    for i in range(numProbes):
        activityTrain.extend(activityProbes[i].data)
    # Transform to numpy array
    activityTrain = np.array(activityTrain)
    # Calculate spike train from activity
    #spikeTrain = activityTrain[:,1:] - activityTrain[:,:-1]
    activityTrain[:,1:] -= activityTrain[:,:-1]
    spikeTrain = activityTrain

    return spikeTrain

"""
@desc: Calculate cross correlation between spike trains of two neurons
"""
def cor(self, t1, t2):
    # Calculate standard devaition of each spike train
    sd1 = np.sqrt(np.correlate(t1, t1)[0])
    sd2 = np.sqrt(np.correlate(t2, t2)[0])

    # Check if any standard deviation is zero
    if (sd1 != 0 and sd2 != 0):
        return np.correlate(t1, t2)[0]/np.multiply(sd1, sd2)
    else:
        return 0

"""
@desc: Filter spike train
@pars: spikeTrain: has N rows (number of neurons) and T columns (number of time steps)
       filter: filter method as string, can be: 'single exponential', 'double exponential' or 'gaussian' (symmetric)
"""
def getFilteredSpikes(self, spikes, filter="single exponential"):
    if (filter == 'single exponential'):
        return self.getSingleExponentialFilteredSpikes(spikes)
    if (filter == 'double exponential'):
        return self.getHoltDoubleExponentialFilteredSpikes(spikes)
    if (filter == 'gaussian'):
        return self.getGaussianFilteredSpikes(spikes)

"""
@desc: Get symmetric gaussian filtered spikes
"""
def getGaussianFilteredSpikes(self, spikes):
    # Define some variables
    wd = self.p.smoothingWd  # width of smoothing, number of influenced neurons to the left and right
    var = self.p.smoothingVar  # variance of the Gaussian kernel
    
    # Define the kernel
    lin = np.linspace(-wd,wd,(wd*2)+1)
    kernel = np.exp(-(1/(2*var))*lin**2)

    # Prepare spike window
    spikeWindow = np.concatenate((spikes[-wd:,:], spikes, spikes[:wd,:]))

    # Prepare smoothed array
    nSteps, nNeurons = spikeWindow.shape
    smoothed = np.zeros((nSteps, nNeurons))
    
    # Add smoothing to every spike
    for n in range(nNeurons):
        for t in range(wd, nSteps - wd):
            # Only add something if there is a spike, otherwise just add zeros
            add = kernel if spikeWindow[t,n] == 1 else np.zeros(2*wd+1)
            # Add values to smoothed array
            smoothed[t-wd:t+wd+1, n] += add

    # Return smoothed activity
    return smoothed[wd:-wd,:]

"""
@desc: Get single exponential filtered spikes
"""
def getSingleExponentialFilteredSpikes(self, spikes, smoothing_level=0.1):
    # Get dimensions
    N, T = np.shape(spikes)

    filteredSpikes = []
    # Iterate over all neurons
    for i in range(N):
        # Fit values
        fit = SimpleExpSmoothing(spikes[i,:]).fit(smoothing_level=smoothing_level)
        # Append filtered values for current neuron
        filteredSpikes.append(fit.fittedvalues)

    # Transform to numpy array and return
    return np.array(filteredSpikes)

"""
@desc: Get holt double exponential filtered spikes
"""
def getHoltDoubleExponentialFilteredSpikes(self, spikes, smoothing_level=0.1, smoothing_slope=0.1):
    # Get dimensions
    N, T = np.shape(spikes)

    filteredSpikes = []
    # Iterate over all neurons
    for i in range(N):
        # Fit values, if smoothing_slope = 0, result equals single exponential solution
        fit = Holt(spikes[i,:]).fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope)
        # Append filtered values for current neuron
        filteredSpikes.append(fit.fittedvalues)

    # Transform to numpy array and return
    return np.array(filteredSpikes)

"""
@desc: Calculate fano factors
"""
def fano(self, spikes):
    # Get shape
    shp = spikes.shape
    # Iterate over all trials
    ff = []
    for i in range(shp[0]):
        # Get mean and standard deviation of all spike trains
        mn = np.mean(spikes[i], axis=1)
        var = np.var(spikes[i], axis=1)

        # Get indices of zero-values
        mask = (mn != 0)
        
        # Append mean fano factors from all neurons with spiking activity
        ff.append(np.mean(var[mask]/mn[mask]))

    # Return mean fano factors for every trial
    return ff

"""
@desc: Calculate coefficient of variation
"""
def cv(self, spikes):
    # Get shape
    shp = spikes.shape
    # Iterate over all trials
    cv = []
    for i in range(shp[0]):
        # Get mean and standard deviation of all spike trains
        mn = np.mean(spikes[i], axis=1)
        sd = np.std(spikes[i], axis=1)

        # Get indices of zero-values
        mask = (mn != 0)
        
        # Append mean fano factors from all neurons with spiking activity
        cv.append(np.mean(sd[mask]/mn[mask]))

    # Return mean fano factors for every trial
    return cv
