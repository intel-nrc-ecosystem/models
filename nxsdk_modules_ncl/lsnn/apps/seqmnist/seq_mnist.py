# Copyright © 2018-2021 Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#   * Neither the name of Intel Corporation nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from nxsdk_modules.lsnn.apps.seqmnist.seq_img_classifier import \
    SequentialImageClassifierLsnn
from nxsdk_modules.lsnn.datasets.mnist_dataset import loadMNIST
from nxsdk.utils.env_var_context_manager import setEnvWithinContext


def loadMnistData(trainOrTestData='test'):
    """Loads MNIST data from sklearn or web.

    :param str trainOrTestData: Must be 'train' or 'test' and specifies which \
    part of the MNIST dataset to load.
    :return: images, targets
    """

    mnist = loadMNIST()
    if trainOrTestData == 'train':
        X = np.array(mnist.data)[:60000, :].astype(np.uint8)
        y = np.array(mnist.target)[:60000].astype(np.uint8)
    elif trainOrTestData == 'test':
        X = np.array(mnist.data)[60000:, :].astype(np.uint8)
        y = np.array(mnist.target)[60000:].astype(np.uint8)
    else:
        raise ValueError("trainOrTestData must be 'train' or 'test'.")

    return X, y


def loadWeightsAndDelays(dataDir, doLoadDelays=False):
    """Loads input, recurrent and output layer weight matrices and optionally \
    input and recurrent layer delay matrices. All matrices are expected to be \
    numpy arrays.

    :param str dataDir: Directory containing weight (and optional delay \
    matrices).
    :param bool doLoadDelays: If True, also expects delay matrices to be \
    present in directory and loads them as well.
    :return: (wIn, wRec, wOut) or (wIn, wRec, wOut, dIn, dRec) tuples \
    containing numpy weight and delay matrices. All matrices have shape \
    (dstDim, srcDim).
    :rtype: numpy.ndarray
    """

    # Load weights
    path = os.path.join(dataDir, 'w_in.npy')
    wIn = np.load(path)

    path = os.path.join(dataDir, 'w_rec.npy')
    wRec = np.load(path)

    path = os.path.join(dataDir, 'w_out.npy')
    wOut = np.load(path)

    out = (wIn.T, wRec.T, wOut.T)

    # Load delays
    if doLoadDelays:
        path = os.path.join(dataDir, 'delayInArray.npy')
        dIn = np.load(path)

        path = os.path.join(dataDir, 'delayRecArray.npy')
        dRec = np.load(path)

        out = (*out, dIn.T, dRec.T)

    return out


def showWeights(wIn, wRec, wOut):
    """Plots input ,recurrent and output weights as 2D images.

    :param numpy.ndarray wIn: Matrix connecting input layer with recurrent \
    layer.
    :param numpy.ndarray wRec: Matrix connecting recurrent layer with \
    recurrent layer.
    :param numpy.ndarray wOut: Matrix connecting recurrent layer with \
    outputlayer.
    """

    fig = plt.figure(100, figsize=(12,12))
    gs = gridspec.GridSpec(1, 3, width_ratios=[80, 240, 10]) 
    plt.subplot(gs[0])
    plt.imshow(wIn, cmap='RdBu_r')
    plt.title('wIn')
    plt.xlabel('Sources')
    plt.ylabel('Destinations')

    plt.subplot(gs[1])
    plt.imshow(wRec.T, cmap='RdBu_r')
    plt.title('wRec')
    plt.xlabel('Destinations')
    plt.ylabel('Sources')
    plt.gca().axes.set_yticklabels([])

    plt.subplot(gs[2])
    im = plt.imshow(wOut.T, cmap='RdBu_r')
    plt.title('wOut')
    plt.ylabel('Sources')
    plt.xlabel('Destinations')
    plt.gca().axes.set_yticklabels([])
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=fig.get_axes(), shrink=0.49)

    plt.show()


def showImgs(imgs, dx, dy):
    """Shows images as 2D array of tiles images.

    :param numpy.ndarray imgs: Images to be shown are expected to have \
    dimentions (numImg, dx*dy).
    :param int dx: x dimension of images.
    :param int dy: y dimension of images.
    """
    from nxsdk.utils.plotutils import buildImgArray

    lImgs = np.reshape(imgs.T, (dy, dx, imgs.shape[0]))

    plt.figure(1010, figsize=(10, 5))
    img = buildImgArray(lImgs)
    plt.imshow(img, cmap='gray')
    plt.colorbar()

    plt.show()


def runSequentialMnist(wgtDir, numSamples, batchSize):
    """Sets up and executes SequentialImageClassifier LSNN network to\
    classify MNIST images on the test set.

    :param string wgtDir: directory which contains the weight matrices for\
    the network
    :param int numSamples: amount of images which should be processed
    :param int batchSize: batchSize of the SequentialImageClassifierLsnn
    """

    # Specify data directory
    dataDir = os.path.join(os.path.dirname(__file__), 'weights', wgtDir)
    # MNIST images are 28 x 28 in size
    imgDx = imgDy = 28

    # Load input, recurrent and output layer weights
    wIn, wRec, wOut = loadWeightsAndDelays(dataDir)

    # Initialize LSSN network
    sqic = SequentialImageClassifierLsnn(wIn=wIn, wRec=wRec, wOut=wOut,
                                         numInput=80, numRegular=140,
                                         numAdaptive=100, numOutput=10,
                                         cueDuration=56,
                                         imageSize=imgDx * imgDy,
                                         batchSize=batchSize)

    # Load a random set of MNIST test set images and pass to LSNN
    inputs, targets = loadMnistData('test')
    np.random.seed(0)
    imgIdx = np.random.choice(range(0, 10000), numSamples, False)
    inputs, targets = inputs[imgIdx, :], targets[imgIdx]

    # Execute network: Generates spikes from images and injects into LSNN
    sqic.classify(inputs, targets)

    # Show results
    sqic.printClassification()

    return sqic


if __name__ == "__main__":
    # name of the folder with the trainend weights
    wgtDirName = 'v25_94per'

    with setEnvWithinContext(PARTITION="nahuku32"):
        # numSamples range is 1 - 10000; batchSize must be a fraction of
        # numSamples
        runSequentialMnist(wgtDirName, numSamples=100, batchSize=10)
