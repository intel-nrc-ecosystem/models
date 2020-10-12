import numpy as np
import scipy.linalg as la
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from types import SimpleNamespace

from lib.helper.exceptions import ArgumentNotValid

import copy
import gc


"""
@desc: Trains ordinary least square model, includes filtering and regularization
@pars: trainSpikes: has dimensions B (number of trials) x N (number of neurons) x T (number of time steps per trial)
       testSpikes: has dimensions N (number of neurons) x T (number of time steps per trial)
       targetFunction: has dimensions T (number of time steps per trial)
       filter: filter method as string, can be: None, 'single exponential', 'double exponential', 'gaussian' (symmetric) or 'bins'
       binSize: if filter 'bins' is used, sets the bin size
"""
def trainOLS(self, trainSpikes, testSpikes, targetFunction, filter=None, isIntercept=True, binSize=10, alpha=1.0, l1w=0.0):

    # Preprocess if B axis does not exist
    if (len(np.shape(trainSpikes)) == 2):
        trainSpikes = trainSpikes[np.newaxis,...]

    # Get shapes (N: num neurons, T: num time steps per trial, B: num trials)
    B, N, T = np.shape(trainSpikes)
    Nt, Tt = np.shape(testSpikes)

    # Some checks
    if (len(targetFunction) != T):
        raise ArgumentNotValid('Length of target function and length of train spikes is not equal.')
    if (len(targetFunction) != Tt):
        raise ArgumentNotValid('Length of target function and length of test spikes is not equal.')
    if (Nt != N or Tt != T):
        raise ArgumentNotValid('Number of neurons or number of time steps in train and test spikes is not equal.')

    # Get filtered spike trains for train and test spikes
    x, xe = None, None
    # If not filter is chosen, apply it to raw data
    if filter is None:
        x = np.hstack(tuple( trainSpikes[i,:,:] for i in range(B) )) #trainSpikes.reshape(N, T*B)
        xe = testSpikes
    # If filter is bins, bin it with binSize
    elif filter == 'bins':
        data = np.hstack(tuple( trainSpikes[i,:,:] for i in range(B) ))
        #data = trainSpikes.reshape(N, T*B)

        #x = np.array([np.mean(data[:,i*binSize:(i+1)*binSize], axis=1) for i in range(0,T*B,binSize)]).T
        x = np.array([np.mean(data[:,i:i+binSize], axis=1) for i in range(0,T*B,binSize)]).T

        #xe = np.array([np.mean(testSpikes[:,i*binSize:(i+1)*binSize], axis=1) for i in range(0,Tt,binSize)]).T
        xe = np.array([np.mean(testSpikes[:,i:i+binSize], axis=1) for i in range(0,Tt,binSize)]).T

        # In case of bins, also target function need to be compressed
        targetFunction = np.array([np.mean(targetFunction[i:i+binSize]) for i in range(0,T,binSize)])
    # If any other filter is chosen, apply it
    else:
        x = np.array([self.getFilteredSpikes(trainSpikes[i,...], filter) for i in range(B)]).reshape(N, T*B)
        xe = self.getFilteredSpikes(testSpikes, filter)

    # Get target function for all trials
    y = np.tile(targetFunction, B)

    # Add intercept
    if (isIntercept):
        x = np.insert(x, 0, 1.0, axis=0)
        xe = np.insert(xe, 0, 1.0, axis=0)

    # Train the parameters
    model = sm.OLS(y, x.T)
    #params = model.fit().params
    params = model.fit_regularized(alpha=0.0, L1_wt=0.0).params
    #params = model.fit_regularized(alpha=alpha, L1_wt=l1w).params
    # alpha=0.2, L1_wt=0.001

    # Estimate target function for test spike train
    ye = np.dot(xe.T, params)

    # Calculate performance
    mse = np.mean(np.square(targetFunction - ye))  # MSE error
    cor = pearsonr(targetFunction, ye)[0]  # Pearson correlaton coefficient

    # Join performance measures
    performance = SimpleNamespace(**{ 'mse': mse, 'cor': cor })

    return params, ye, performance

"""
@desc: Caluclates PCA out of given data
@pars: data as 2D NumPy array
@return: data transformed in 2 dims/columns + regenerated original data
@link: https://stackoverflow.com/a/13224592/2692283
"""
def pca(self, data, dims_rescaled_data=2):
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs
