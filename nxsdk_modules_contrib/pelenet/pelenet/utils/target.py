import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

"""
@desc: Load robotic movement target function
"""
def loadTarget(self):
    # Target range
    fr = self.p.targetOffset
    to = None if fr is None else (fr + self.p.stepsPerTrial-self.p.binWindowSize)

    # Define file path
    filePath = self.p.targetPath + self.p.targetFilename

    # Load data and return
    return np.loadtxt(filePath)[fr:to,0:3].T

"""
@desc:
        Prepares dataset for estimation
@pars:
        data: has dimensions B (number of trials) x N (number of neurons) x T (number of time steps per trial)
        target: T-dim target function
        binSize: bin works as sliding window to smooth spikes with binSize into the past
        trainTrials: binary array for choosing trials for training (default: all but last trial)
        testTrial: integer which gives the trial number to test with (default: last trial)
@return:
        x: prepared train data
        xe: prepared test data
        y: prepared n-dim target function
"""
def prepareDataset(self, data, target, binSize=None, trainTrials=None, testTrial=None):
    # Set binSize to default parameter value if not set
    if binSize is None: binSize = self.p.binWindowSize
    # If not train trials are given, take all except the last one
    if trainTrials is None: trainTrials = np.append(np.repeat(True, self.p.trials-1), False)
    # If not test trial is given, take the last one
    if testTrial is None: testTrial = self.p.trials-1

    # Select train data
    train = np.array([np.mean(data[trainTrials,:,i:i+binSize], axis=2) for i in range(self.p.stepsPerTrial-binSize)])
    trainSpikes = np.moveaxis(train, 0, 2)

    # Concatenate train data
    x = np.hstack(tuple( trainSpikes[i,:,:] for i in range(np.sum(trainTrials)) ))
    x = np.insert(x, 0, 1.0, axis=0)  # Add intercept

    # Select test data
    test = np.array([np.mean(data[testTrial,:,i:i+binSize], axis=1) for i in range(self.p.stepsPerTrial-binSize)])
    testSpikes = np.moveaxis(test, 0, 1)
    xe = np.insert(testSpikes, 0, 1.0, axis=0)  # Add intercept
    
    # Select target
    y = None
    if len(target.shape) == 1:
        y = np.tile(target[:self.p.stepsPerTrial-binSize], np.sum(trainTrials))
    if len(target.shape) == 2:
        y = np.tile(target[:,:self.p.stepsPerTrial-binSize], np.sum(trainTrials))

    # Return dataset
    return (x, xe, y)

"""
@desc: Estimates a one dimensional function with given train and test data
@pars:
        x: train data
        xe: test data
        y: one dimensional target function
@return:
        ye: estimated target function
"""
def estimateMovement(self, x, xe, y, alpha=0.0, L1_wt=0.0):
    # Fit
    model = sm.OLS(y, x.T)
    params = model.fit_regularized(alpha=alpha, L1_wt=L1_wt).params
    
    # Predict
    ye = np.dot(xe.T, params)
    
    return ye

"""
@desc: Estimates a 3 dimensional function for multiple trajectories
@pars:
        Arguments equal arguments from prepareDataset() function
        Except for 'targets' which has one more dimension compared to 'target'
@return:
        yes: estimated target functions
"""
def estimateMultipleTrajectories3D(self, data, targets, binSize=None, trainTrials=None, testTrial=None, alpha=0.0, L1_wt=0.0):
    # Create empty array
    yes = []

    # Loop over all target functions
    for i in tqdm(range(targets.shape[0])):
        # Prepare data
        (x, xe, y) = self.prepareDataset(data, targets[i], binSize=binSize, trainTrials=trainTrials, testTrial=testTrial)

        # Estimate x, y and z
        x1 = self.estimateMovement(x, xe, y[0], alpha, L1_wt)
        x2 = self.estimateMovement(x, xe, y[1], alpha, L1_wt)
        x3 = self.estimateMovement(x, xe, y[2], alpha, L1_wt)
        
        # Append all estimates to array
        yes.append(np.array([x1, x2, x3]))

    return np.array(yes)
