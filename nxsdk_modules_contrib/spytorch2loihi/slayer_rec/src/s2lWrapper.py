# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
# 
# Copyright © 2020-2021 Intel Corporation.
# 
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express 
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy, 
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are 
# expressly stated in the License.

# Sumit Bam Shrestha 10/02/2020 10am
# =================================
'''
Auto tools for implementing SLAYER-trained models on Loihi.
This is a wrapper code that generates feedforward network from the hdf5  
network file that results from slayer training(*.net). This will also 
include modules to output a nxsdk net that can be run in Loihi.

The hdf5 field description is as follows

.. code-block:: python

    ./
    |->simulation # simulation description
    |   |->Ts # sampling time. Usually 1
    |   |->tSample # length of the sample to run
    |->layer # description of network layer blocks such as input, dense, conv, pool, flatten, average
        |->0
        |   |->{shape, type, ...} # each layer description has ateast shape and type attribute
        |->1
        |   |->{shape, type, ...}
        :
        |->n
            |->{shape, type, ...}

    input  : {shape, type}
    flatten: {shape, type}
    average: {shape, type}
    dense  : {shape, type, neuron, inFeatures, outFeatures, weight, delay(if available)}
    pool   : {shape, type, neuron, kernelSize, stride, padding, dilation, weight}
    conv   : {shape, type, neuron, inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weight, delay(if available)}
                            |-> this is the description of the compartment parameters
                            |-> {iDecay, vDecay, vThMant, refDelay, ... (other additional parameters can exist)}

'''

# requirements
# pip import h5py (h5py-2.10.0)

import os
import numpy as np
import h5py
import time
import copy
from attrdict import AttrDict
import nxsdk.api.n2a as nx
from .slayer2loihi import Slayer2Loihi as s2l

def _getCompProto(neuronConfig):
    # internal function to get compartment prototype from neuron config
    return nx.CompartmentPrototype(
        vThMant = neuronConfig['vThMant'][()],
        compartmentVoltageDecay = neuronConfig['vDecay'][()],
        compartmentCurrentDecay = neuronConfig['iDecay'][()],
        refractoryDelay = neuronConfig['refDelay'][()],
        vMinExp = neuronConfig['vMin'] if 'vMin' in neuronConfig.keys() else 23,
    )


def _solveSynapseConstraints(nNeurons, nWeights, nWeightBits, nDelayBits):
    # """
    # """
    # This function is called internally to check the SYNAPSE_MEM constraints
    # It can be made private
    # 
    # neuronsPerCore = nNeurons
    # for nGroups in range(1, nNeurons + 1):
    #     # Ignoring the headers
    #     # Assuming no tag bits
    #     # nParamsBits = nWeightBits + nDelayBits + np.ceil(np.log2(nNeurons/nGroups)).astype(int)
    #     nParamsBits = nWeightBits + nDelayBits
    #     nParamsPerWord = 64 // nParamsBits
    #     maxWeightsPossible = 16384 * nParamsPerWord
    #     newGroupsEstimate = np.ceil(nWeights / maxWeightsPossible)

    #     print(nParamsBits, nParamsPerWord, maxWeightsPossible, nGroups, newGroupsEstimate)
    #     # Cnew = maxStateBits * maxC // (nWeights * (nWeightBits + nDelayBits + np.ceil(np.log2(C)).astype(int)))
    #     # print(Cnew)

    #     if newGroupsEstimate <= nGroups:
    #         # print(nNeurons / nGroups)
    #         # print(1024 / nGroups)
    #         neuronsPerCore = nNeurons // nGroups
    #         break

    # print()
    # return neuronsPerCore

    nSynapticMemWords = 16384
    nBitsPerWord = 64
    nHeaderWords = 4 # 16 # arbitrary assigning some fields for headers and stuffs. Not exact.
    
    # Ignoring the headers
    # Assuming no tag bits
    nParamsBits = nWeightBits + nDelayBits
    nParamsPerWord = nBitsPerWord // nParamsBits
    maxWeightsPossible = (nSynapticMemWords - nHeaderWords) * nParamsPerWord 
    nGroups = np.ceil(nWeights / maxWeightsPossible)

    alpha = nGroups - nWeights / maxWeightsPossible
    pessimisticEstimate = nNeurons // nGroups
    optimisticEstimate  = np.floor(nNeurons * maxWeightsPossible / nWeights)

    # print(nParamsBits, nParamsPerWord, maxWeightsPossible, nGroups, pessimisticEstimate, optimisticEstimate)

    # return nNeurons // nGroups

    # when you don't have much margin on nGroups, you want to be close to pessimistic estimate.
    # when you have large margin on nGroups, you can be closer to optimistic estimate.
    return np.floor((1-alpha) * pessimisticEstimate + alpha * optimisticEstimate)

class Network:
    """This Class encapsulates the network creation, mapping and execution in 
    Loihi hardware. The trained network is passed as a `netCofigFile` (a hdf5 file
    that describes the sequential network configuration blocks and the trained
    synaptic weights and axonal delays)

    Note: axonal delays are implemented as synaptic delays at the moment.
    """
    def __init__(self, netConfigFile, net=None, probeOutput=True, noDelay=False):
        """Initializes the class with trained hdf5 network configuration description file.
        The network configuration description is general, independent of the training 
        platform or SLAYER.

        Args:
            netConfigFile (string): The path to the trained network configuration hdf5 file (usually stored as `*.net`)
            probeOutput (bool, optional): A boolean flag to determine if the output spike count should be probed or not. Defaults to True.
            noDelay (bool, optional): A boolean flag to ignore the axonal delays in the trained network. Defaults to False.
        """
        self.netConfig = h5py.File(netConfigFile, 'r') 
        self.net = nx.NxNet() if net is None else net
        self.inputConnectionGroup = None
        self.inputLayer = None
        self.outputLayer = None
        self.corenums = []
        self.probeOutput = probeOutput
        self.dummyProbes = None
        self.board = None
        self.savedBoardName = None
        self.spikeChannels = None
        self.core = None
        self.axons = None
        self.spikeData = None
        self.numSteps = None
        self.spikesPerPacket = None
        self.numSamples = None
        self.results = None
        self.noDelay = noDelay
        self.globalAverage = None

    def _tableStr(self, typeStr='', width=None, height=None, channel=None, kernel=None, stride=None, 
                 padding=None, delay=False, neuronsPerCore=None, nCores=None, header=False, footer=False):
        # A private helper function to print the mapping output configuration for a layer/block.
        if header is True:
            return '|{:10s}|{:5s}|{:5s}|{:5s}|{:5s}|{:5s}|{:5s}|{:5s}|{:12s}|{:5s}|'.format(
                    '   Type   ', '  W  ', '  H  ', '  C  ', ' ker ', ' str ', ' pad ', 'delay', 'neurons/Core', 'cores')
        elif footer is True and nCores is not None:
            return '|{:10s} {:5s} {:5s} {:5s} {:5s} {:5s} {:5s} {:5s} {:12s}|{:-5d}|'.format(
                    'Total', '', '', '', '', '', '', '', '', nCores)
        else:
            entry = '|'
            entry += '{:10s}|'.format(typeStr)
            entry += '{:-5d}|'.format(width)
            entry += '{:-5d}|'.format(height)
            entry += '{:-5d}|'.format(channel)
            
            if kernel is None:
                entry += '{:5s}|'.format('')
            elif len(kernel.shape) == 1:
                entry += '{:-2d},{:-2d}|'.format(kernel[1], kernel[0])
            else:
                entry += '{:-5d}|'.format(kernel)
            
            if stride is None:
                entry += '{:5s}|'.format('')
            elif len(stride.shape) == 1:
                entry += '{:-2d},{:-2d}|'.format(stride[1], stride[0])
            else:
                entry += '{:-5d}|'.format(stride)

            if padding is None:
                entry += '{:5s}|'.format('')
            elif len(padding.shape) == 1:
                entry += '{:-2d},{:-2d}|'.format(padding[1], padding[0])
            else:
                entry += '{:-5d}|'.format(padding)
            
            entry += '{:5s}|'.format(str(delay))
            entry += '{:-12d}|'.format(neuronsPerCore) if neuronsPerCore is not None else '{:12s}|'.format('')
            entry += '{:-5d}|'.format(nCores) if nCores is not None else '{:5s}|'.format('')

            return entry

    def _optimizeForOutputAxonConstraint(self, layer, compPerCore, crossChip=False):
        # A private helper function to optimize compartments per core placement with consideration to output axon 
        # constraints. This seems the most mysterious to me!
        layerConfig = self.netConfig['layer']
        MAX_OUTPUT_AXONS = 4096 if crossChip is False else 1024

        i = layer
        key = '{}'.format(i)
        layerType = layerConfig[key]['type'][0].decode('ascii')
        nextInd = i+1
        nextKey = '{}'.format(nextInd)
        nextLayerType = layerConfig[nextKey]['type'][0].decode('ascii')
        if nextLayerType == 'flatten':
            nextInd = i+2
            nextKey = '{}'.format(nextInd)
        nextLayerType = layerConfig[nextKey]['type'][0].decode('ascii')

        if nextLayerType == 'dense' and layerType != 'flatten':
            nCoresToFanOut = np.ceil(layerConfig[nextKey]['outFeatures'][()] / compPerCore[nextInd])
            selfCoreFanOut = compPerCore[i] * nCoresToFanOut
            # print(selfCoreFanOut, nCoresToFanOut)
            if selfCoreFanOut > MAX_OUTPUT_AXONS:
                additionalGrps = np.ceil(selfCoreFanOut / MAX_OUTPUT_AXONS)
                # print(compPerCore[i], additionalGrps)
                compPerCore[i] = compPerCore[i] // additionalGrps
                residue = additionalGrps - selfCoreFanOut // MAX_OUTPUT_AXONS
                if residue == 0:    # means the additional Groups doesn't have any margin. So relax the compPerCore
                    compPerCore[i] -= 1
        elif nextLayerType == 'conv':
            outChannels = layerConfig[nextKey]['outChannels'][()]
            inChannels = layerConfig[nextKey]['inChannels'][()]
            kernelSize = layerConfig[nextKey]['kernelSize'][()]
            stride = layerConfig[nextKey]['stride'][()]
            groups = layerConfig[nextKey]['groups'][()]

            # looking at the output constraint just with convolution output perspective
            # this is the minimum output constraint that a conv layer needs to satisfy, irrespective of the way neurons
            # are flattened inside the core
            if len(kernelSize.shape) == 0:
                numWeightsPerInput = outChannels * kernelSize * kernelSize 
            else:
                numWeightsPerInput = outChannels * kernelSize[0] * kernelSize[1] 

            strideProd = stride * stride if len(stride.shape) == 0 else stride[0] * stride[1]
            nCoresToFanOut = np.ceil(numWeightsPerInput / compPerCore[nextInd] / strideProd)

            # The following is dependent on how the neurons are flattened out. s2l uses WHC layout.
            kernelSizeX = kernelSize if len(kernelSize.shape) == 0 else kernelSize[1]
            kernelSizeY = kernelSize if len(kernelSize.shape) == 0 else kernelSize[0]

            cCompPerCore = compPerCore[nextInd] / outChannels
            if cCompPerCore < 1: # channed dimension does not fit inside a single destination core 
                xFanOutCores = kernelSizeX
                yFanOutCores = kernelSizeY
                cFanOutCores = np.ceil(outChannels / cCompPerCore)
            else: # channel dimension fits inside a single destination core
                yCompPerCore = compPerCore[nextInd] / outChannels
                xCompPerCore = compPerCore[nextInd] / outChannels / kernelSizeY
                if yCompPerCore <= kernelSizeY: # y dimension also fits inside a single destination core
                    xFanOutCores = np.ceil(kernelSizeX / xCompPerCore)
                    yFanOutCores = 1
                    cFanOutCores = 1
                else: # y dimension does not fit inisde a single destination core
                    xFanOutCores = kernelSizeX
                    yFanOutCores = np.ceil(kernelSizeY / yCompPerCore)
                    cFanOutCores = 1

            # use the max of the calculated number of cores to fan out
            # print(nCoresToFanOut, compPerCore[i])
            # print(xFanOutCores, yFanOutCores, cFanOutCores)
            nCoresToFanOut = max(nCoresToFanOut, xFanOutCores * yFanOutCores * cFanOutCores / strideProd)

            selfCoreFanOut = compPerCore[i] * nCoresToFanOut
            # print(i, selfCoreFanOut, nCoresToFanOut)
            if selfCoreFanOut > MAX_OUTPUT_AXONS:
                additionalGrps = np.ceil(selfCoreFanOut / MAX_OUTPUT_AXONS)
                # print('    ', compPerCore[i], additionalGrps)
                compPerCore[i] = compPerCore[i] // additionalGrps
                # residue = additionalGrps - selfCoreFanOut // MAX_OUTPUT_AXONS
                # if residue == 0:    # means the additional Groups doesn't have any margin. So relax the compPerCore
                #     compPerCore[i] -= 1            

        return compPerCore
    
    def _optimizeNeuronsPerCore(self):
        # A private helper function to optimize overall compartments per core palcement.

        # forward pass and backward pass to optimize the neurons per core for each layer
        layerConfig = self.netConfig['layer']
        MAX_COMPARTMENTS_PER_NEURONS = 1024
        MAX_INPUT_AXONS = 4096

        # start with a maximum of 1024 compartments per core
        # constraint on number of compartments
        compPerCore = MAX_COMPARTMENTS_PER_NEURONS * np.ones((len(layerConfig))) 
        
        for i in range(len(layerConfig)):
            key = '{}'.format(i)
            layerType = layerConfig[key]['type'][0].decode('ascii')

            if layerType == 'pool':
                stride = layerConfig[key]['stride'][()]
                dilation = layerConfig[key]['dilation'][()]
                # padding does not affect compartments per core (at least assuming it doesn't)
                compPerCore[i] = min(MAX_INPUT_AXONS / stride / stride * dilation * dilation, compPerCore[i])

            elif layerType == 'conv':
                inChannels = layerConfig[key]['inChannels'][()]
                outChannels = layerConfig[key]['outChannels'][()]
                kernelSize = layerConfig[key]['kernelSize'][()]
                stride = layerConfig[key]['stride'][()]
                padding = layerConfig[key]['padding'][()]
                dilation = layerConfig[key]['dilation'][()]
                groups = layerConfig[key]['groups'][()]

                if len(kernelSize.shape) == 0:
                    numWeightsPerOutput = inChannels * kernelSize * kernelSize 
                else:
                    numWeightsPerOutput = inChannels * kernelSize[0] * kernelSize[1] 
                
                # constraints on the number of input axons
                if numWeightsPerOutput > 4096:
                    raise Exception('Impossible to fit with input fan-in of {}'.format(inFeatures))

                # constraints on number of synapse memory
                nBitsPerWeight = np.ceil(np.log2(2*np.abs(layerConfig[key]['weight'][()]).max()))
                
                nOutputNeurons = layerConfig[key]['shape'][0] * layerConfig[key]['shape'][1] * layerConfig[key]['shape'][2]
                
                if 'delay' in layerConfig[key].keys() and self.noDelay is False:
                    if layerConfig[key]['delay'][()].max() == 0:
                        numDelays = nBitsPerDelay = 0
                    else:
                        numDelays = nOutputNeurons
                        nBitsPerDelay = np.ceil(np.log2(layerConfig[key]['delay'][()].max()))
                else:
                    numDelays = nBitsPerDelay = 0

                compPerCoreSynapseConstr = _solveSynapseConstraints(
                    nOutputNeurons, numWeightsPerOutput * nOutputNeurons, nBitsPerWeight, nBitsPerDelay,
                )
                compPerCore[i] = min(compPerCoreSynapseConstr, compPerCore[i])

                # constraints on number of accessable dendritic accumulators due to delay bits
                if numDelays != 0:
                    compPerCore[i] = min(compPerCore[i], 1<<(13-nBitsPerDelay.astype(int)))
                    # self.maxAddressableCompartmentsPerCoreGivenDaCfgDelayBits(delayBits)))
                    # nxsdk.compiler.nxsdkcompiler.n2_compiler_exceptions.NxBoundsValidationError: Exceeding max number 
                    # of addressable compartments in a core due to dendritic accumulator delay bits on core 40. 
                    # Configured compartments 1024. Max Addressable 256.
                    # Max Addressable compartments seems to be 2**(13-nDelayBits)
                # compPerCore[i] = 128

            elif layerType == 'dense':
                inFeatures = layerConfig[key]['inFeatures'][()]
                outFeatures = layerConfig[key]['outFeatures'][()]

                # constraints on the number of input axons
                if inFeatures > 4096:
                    raise Exception('Impossible to fit with input fan-in of {}'.format(inFeatures))

                # constraints on number of synapse memory
                numWeights = inFeatures * outFeatures
                nBitsPerWeight = np.ceil(np.log2(2*np.abs(layerConfig[key]['weight'][()]).max()))
                
                if 'delay' in layerConfig[key].keys() and self.noDelay is False:
                    if layerConfig[key]['delay'][()].max() == 0:
                        numDelays = nBitsPerDelay = 0
                    else:
                        numDelays = outFeatures
                        nBitsPerDelay = np.ceil(np.log2(layerConfig[key]['delay'][()].max()))
                else:
                    numDelays = nBitsPerDelay = 0

                compPerCoreSynapseConstr = _solveSynapseConstraints(outFeatures, numWeights, nBitsPerWeight, nBitsPerDelay)
                compPerCore[i] = min(compPerCoreSynapseConstr, compPerCore[i])

                # constraints on number of accessable dendritic accumulators due to delay bits
                if numDelays != 0:
                    # print(nBitsPerDelay)
                    compPerCore[i] = min(compPerCore[i], 1<<(13-nBitsPerDelay.astype(int)))
                    # self.maxAddressableCompartmentsPerCoreGivenDaCfgDelayBits(delayBits)))
                    # nxsdk.compiler.nxsdkcompiler.n2_compiler_exceptions.NxBoundsValidationError: Exceeding max number 
                    # of addressable compartments in a core due to dendritic accumulator delay bits on core 40. 
                    # Configured compartments 1024. Max Addressable 256.
                    # Max Addressable compartments seems to be 2**(13-nDelayBits)

        # print('output axon constraints')
        # backward pass for output axon constraint
        for i in range(len(layerConfig)-1)[::-1]:
            compPerCore = self._optimizeForOutputAxonConstraint(layer=i, compPerCore=compPerCore)

        # backward pass for cross chip output constraints
        # In this case, the core can have a fan out of only 2048
        while True:
            oldCompPerCore = copy.deepcopy(compPerCore)
            chip = 0
            core = 0
            for i in range(len(layerConfig)-1):
                key = '{}'.format(i)
                layerType = layerConfig[key]['type'][0].decode('ascii')
                
                nCores = np.ceil(np.prod(layerConfig[key]['shape']) / compPerCore[i])
                core += nCores

                # this core will now have to route to next chip
                if core >= 128:
                    core -= 128
                    chip += 1

                    __core = core
                    __chip = chip
                    for ii in range(i-1, -1, -1):
                        compPerCore = self._optimizeForOutputAxonConstraint(layer=ii, compPerCore=compPerCore, crossChip=(ii==i-1))
            
            if np.sum(np.abs(np.array(oldCompPerCore) - np.array(compPerCore))) == 0:
                break
                    
            # print(layerType, nextLayerType, compPerCore[i], compPerCore[nextInd])
        # print(compPerCore)
        return compPerCore

    def create(self, corenum=0, customCompPerCore=[], numLayers=None, relayInput=False):
        """Creates a nxNet network based on the trained network configuration 
        initialized during instance creation. This value is internally stored as
        `self.netConfig`.

        Args:
            corenum (int, optional): The core number to start placing the network from. 
                Use the default value unless you want to implement two nets side by side. Defaults to 0.
            customCompPerCore (list, optional): A list of a tuple/ordered pair to overwrite
                the automatic compartments per core calculation. To be used when the automatic
                comapartments per core calculation fails (Hope it does not). Defaults to [].

        Raises:
            Exception: If the input layer is not found at the begining of the network configuration
            Exception: If the layer type read is not one of the recognized types

        Returns:
            int: core number where the network placement ended. It can be used to place the 
                next network from. Usually, the return value can be ignored.
        """
        self.corenums.append(corenum)
        layerConfig = self.netConfig['layer']

        compPerCore = self._optimizeNeuronsPerCore()
        for index, value in customCompPerCore:
            if index < len(compPerCore):
                compPerCore[index] = value
        print('Creating Network')
        print(self._tableStr(header=True))
        if numLayers is None:
            numLayers = len(layerConfig)
        else:
            numLayers = min(numLayers, len(layerConfig))

        for i in range(numLayers):
            key = '{}'.format(i)

            layerType = layerConfig[key]['type'][0].decode('ascii')
            # print(layerType)

            if layerType == 'input':
                if i!= 0:
                    raise Exception('Expected input layer to be at the begining. Found it at i={}'.format(i))
            
                inputSpec = {
                    'sizeX': layerConfig[key]['shape'][2],
                    'sizeY': layerConfig[key]['shape'][1],
                    'sizeC': layerConfig[key]['shape'][0],
                }
                # print(inputSpec)
                if relayInput is True:
                    layer, corenum = s2l.relayLayer(self.net, inputSpec, corenum, compartmentsPerCore=compPerCore[i])
                else:
                    layer, self.inputConnectionGroup, corenum = s2l.inputLayer(self.net, inputSpec, corenum, 
                                                                               compartmentsPerCore=compPerCore[i])
                # print('Compartments Per core:', compPerCore[i])
                print(self._tableStr(
                    typeStr=layerType,
                    width=layerConfig[key]['shape'][2], 
                    height=layerConfig[key]['shape'][1], 
                    channel=layerConfig[key]['shape'][0], 
                    neuronsPerCore=int(compPerCore[i]),  
                    nCores=np.ceil(np.prod(layerConfig[key]['shape'])/compPerCore[i]).astype(int),
                ))
                self.inputLayer = layer
                self.corenums.append(corenum)
            elif layerType == 'pool':
                poolSpec = {
                    'stride': layerConfig[key]['stride'][()],
                    'padding': layerConfig[key]['padding'][()],
                    'dilation': layerConfig[key]['dilation'][()],
                    'compProto': _getCompProto(layerConfig[key]['neuron']),
                    'weight': layerConfig[key]['weight'][()],
                }
                # print(poolSpec)
                layer, corenum = s2l.poolingLayer(layer, poolSpec, corenum, compartmentsPerCore=compPerCore[i])
                # print('Compartments Per core:', compPerCore[i])
                print(self._tableStr(
                    typeStr=layerType,
                    width=layerConfig[key]['shape'][2], 
                    height=layerConfig[key]['shape'][1], 
                    channel=layerConfig[key]['shape'][0], 
                    kernel=poolSpec['stride'],
                    neuronsPerCore=int(compPerCore[i]), 
                    nCores=np.ceil(np.prod(layerConfig[key]['shape'])/compPerCore[i]).astype(int),
                ))
                self.corenums.append(corenum)

            elif layerType == 'conv':
                kernelSize = layerConfig[key]['kernelSize'][()]
                convSpec = {
                    'dimX': kernelSize if len(kernelSize.shape) == 0 else kernelSize[1],
                    'dimY': kernelSize if len(kernelSize.shape) == 0 else kernelSize[0],
                    'dimC': layerConfig[key]['outChannels'][()],
                    'stride': layerConfig[key]['stride'][()],
                    'padding': layerConfig[key]['padding'][()],
                    'dilation': layerConfig[key]['dilation'][()],
                    'groups': layerConfig[key]['groups'][()],
                    'compProto': _getCompProto(layerConfig[key]['neuron']),
                    'weight': layerConfig[key]['weight'][()],
                }
                if 'delay' in layerConfig[key].keys() and self.noDelay is False:
                    convSpec['delay'] = layerConfig[key]['delay'][()]
                if 'bias' in layerConfig[key].keys():
                    convSpec['bias'] = layerConfig[key]['bias'][()]
                # weightFile and delayFile should be changed to weight and delay only in s2l
                if len(convSpec['weight'].shape) != 4:
                    K = convSpec['dimC']
                    H = convSpec['dimY']
                    W = convSpec['dimX']
                    convSpec['weight'] = convSpec['weight'].reshape((K, -1, H, W))
                # print(convSpec)
                layer, corenum = s2l.convLayer(layer, convSpec, corenum, compartmentsPerCore=compPerCore[i])
                # print('Compartments Per core:', compPerCore[i])
                print(self._tableStr(
                    typeStr=layerType,
                    width=layerConfig[key]['shape'][2], 
                    height=layerConfig[key]['shape'][1], 
                    channel=layerConfig[key]['shape'][0], 
                    kernel=kernelSize, stride=convSpec['stride'], padding=convSpec['padding'], 
                    delay='delay' in layerConfig[key].keys() and self.noDelay is False,
                    neuronsPerCore=int(compPerCore[i]),
                    nCores=np.ceil(np.prod(layerConfig[key]['shape'])/compPerCore[i]).astype(int),
                ))
                self.corenums.append(corenum)

            elif layerType == 'dense':
                fullSpec = {
                    'dim': layerConfig[key]['outFeatures'][()],
                    'compProto': _getCompProto(layerConfig[key]['neuron']),
                    'weight': layerConfig[key]['weight'][()],
                }
                if 'delay' in layerConfig[key].keys() and self.noDelay is False:
                    fullSpec['delay'] = layerConfig[key]['delay'][()]
                if 'bias' in layerConfig[key].keys():
                    convSpec['bias'] = layerConfig[key]['bias'][()]
                
                layer, corenum = s2l.fullLayer(layer, fullSpec, corenum, compartmentsPerCore=compPerCore[i])
                # print('Compartments Per core:', compPerCore[i])
                print(self._tableStr(
                    typeStr=layerType,
                    width=1, height=1, channel=fullSpec['dim'], 
                    delay='delay' in layerConfig[key].keys() and self.noDelay is False,
                    neuronsPerCore=int(compPerCore[i]), 
                    nCores=np.ceil(np.prod(layerConfig[key]['shape'])/compPerCore[i]).astype(int), 
                ))
                self.corenums.append(corenum)


                




            # --------------------------------------------------------------------------------

            elif layerType == 'recurrent':
                fullSpec = {
                    'dim': layerConfig[key]['outFeatures'][()],
                    'compProto': _getCompProto(layerConfig[key]['neuron']),
                    'weight': layerConfig[key]['weight'][()],
                    'recWeight': layerConfig[key]['recWeight'][()],
                }
                if 'delay' in layerConfig[key].keys() and self.noDelay is False:
                    fullSpec['delay'] = layerConfig[key]['delay'][()]
                if 'bias' in layerConfig[key].keys():
                    convSpec['bias'] = layerConfig[key]['bias'][()]
                
                layer, corenum = s2l.recurrentLayer(layer, fullSpec, corenum, compartmentsPerCore=compPerCore[i])
                # print('Compartments Per core:', compPerCore[i])
                print(self._tableStr(
                    typeStr=layerType,
                    width=1, height=1, channel=fullSpec['dim'], 
                    delay='delay' in layerConfig[key].keys() and self.noDelay is False,
                    neuronsPerCore=int(compPerCore[i]), 
                    nCores=np.ceil(np.prod(layerConfig[key]['shape'])/compPerCore[i]).astype(int), 
                ))
                self.corenums.append(corenum)

            # --------------------------------------------------------------------------------






                
            elif layerType == 'flatten':
                layer = s2l.reorderLayer(layer)

            elif layerType == 'average':
                self.globalAverage = layerConfig[key]['shape'][0]
                print(self._tableStr(
                    typeStr=layerType,
                    width=layerConfig[key]['shape'][2], 
                    height=layerConfig[key]['shape'][1], 
                    channel=layerConfig[key]['shape'][0], 
                    neuronsPerCore=0,
                    nCores=0,
                ))
            else:
                # you should not have reached here
                raise Exception('Unknown layerType found in network config file. It was {}'.format(layerType))
        
        print(self._tableStr(footer=True, nCores=np.ceil(corenum).astype(int)))
        print()

        self.outputLayer = layer
        if self.probeOutput is True:
            self.dummyProbes = s2l.setupSpikeCounters(self.outputLayer)
        # return layer, inputConnectionGroup, corenum
        return corenum

    def setupIO(self, dataset, spikesPerPacket=2048, numSnips=1, blankTime=100, spikeTime=None):
        """Setup the input output connections to inject spikes to the input layer and collect 
        spike counts from the output layer.

        Args:
            dataset : Any dataset that returns an event object where events are accessible as `x`, `y`, `p`, `t` fields. 
                A generic dataset wrapped with `s2lDataset` is the suitable candidate most of the time. Custom 
                objects are also possible. 
            spikesPerPacket (int, optional): Number of spikes to be sent per packet. Defaults to 2048.
            numSnips (int, optional): Number of snips processors to use for spike injection. Defaults to 1.
            blankTime (int, optional): Number of blank steps (no input period) between two samples. This allows 
                for the neurons to reset before the next sample. Defaults to 100.
            spikeTime (int, optional): Number of time tikcs per sample to use. If None, the value from 
                `self.netConfig['simulation']['tSample']` is used implicitly. Defaults to None.
        """
        self.numSamples = len(dataset)

        self.spikesPerPacket = spikesPerPacket

        if spikeTime is None:
            spikeTime = self.netConfig['simulation']['tSample'][()]
        print('Using per sample spike time: {}steps (+ {}steps gap)'.format(spikeTime, blankTime))
        sampleLength = spikeTime + blankTime

        dataset.sampleLength = spikeTime

        s2l.writeHeader(self.outputLayer, spikesPerPacket, sampleLength)
        self.spikeChannels, self.core, self.axon = s2l.prepSpikeInjection(
            self.inputConnectionGroup, self.board,
            spikesPerPacket, sampleLength, numSnips, 
            regenerateCoreAxon = (self.savedBoardName is None),
        )

        self.spikeData, self.numSteps = s2l.prepSpikeData(
            self.core, self.axon, spikesPerPacket, self.inputLayer, 
            dataset, self.numSamples, sampleLength, numSnips
        )

        if self.probeOutput is True:
            self.spikeCntrChannel = s2l.prepSpikeCounter(
                self.board, self.numSamples, self.outputLayer.numNodes, int(self.corenums[-1])
            )

        if self.savedBoardName is not None:
            s2l.loadBoard(self.board, self.savedBoardName)
    
    def compile(self, fileName=None):
        """Compile the nxNet.

        Args:
            fileName (string, optional): If None, the network is compiled for execution on 
                the hardware. If it is a string, it should represent the path to the saved board 
                state which will be loaded instead of compiling the whole network. Defaults to None.
        """

        if fileName is not None and not os.path.isfile('temp/' + fileName + '.pkl') and not os.path.isfile('temp/' + fileName + '.board'):
            print('The board file {} is not valid. Recompiling the network.'.format(fileName))
            fileName = None

        if fileName is None:
            print('Starting compilation ...')
            tStart = time.time()
            compiler = nx.N2Compiler()
            self.board = compiler.compile(self.net)
            tEnd = time.time()
            print('Completed compilation in {:.2f} seconds'.format(tEnd-tStart))
        else:
            print('Loading from saved board state.')
            self.board, self.dummyProbes = s2l.initBoard(fileName)
            self.savedBoardName = fileName

    def save(self, fileName):
        """Save the network compiled board state to file.

        Args:
            fileName (string): The filename to save the compiled board state.
        """
        s2l.saveBoard(self.board, fileName, self.dummyProbes)
        print('Board saved as {}'.format(fileName))

    def run(self):
        """Execute the network in Loihi hardware

        Returns:
            list: A list of spike count of the output layer neurons for each input sample presented.
        """
        self.board.start()
        self.board.run(self.numSteps, aSync=True)
        tStart = time.time()
        s2l.sendSpikeData(self.spikeData, self.spikeChannels, self.spikesPerPacket)
        if self.probeOutput is True:
            self.results = s2l.getResults(self.spikeCntrChannel, self.numSamples, self.outputLayer.numNodes, self.dummyProbes)
            print('Gathered results')
            # if self.globalAverage is not None:
            #     avgResults = []
            #     for result in self.results:
            #         avgResults.append(np.mean(np.array(result).reshape((self.globalAverage, -1)), axis=1))

        self.board.finishRun()
        self.board.disconnect()
        tEnd = time.time()
        print('Completed {} timesteps in {:.2f} seconds'.format(self.numSteps, tEnd-tStart))
        return self.results
    
    def shareResources(self, network):
        # shares resources with another network
        network.board = self.board
        network.savedBoardName = self.savedBoardName
        network.spikeChannels = self.spikeChannels
        network.inputConnectionGroup = self.inputConnectionGroup
        # network.core = self.core
        # network.axons = self.axons
        # network.spikeData = self.spikeData
        # network.numSteps = self.numSteps
        # network.spikesPerPacket = self.spikesPerPacket
        # network.numSamples = self.numSamples

class s2lDataset:
    """A class that wraps a generic dataset object to a suitable form compatible with
    `slayer2loihi` spike injection mechanism. The generic dataset object must return
    a numpy event (x,y,p,t in ms) format when indexed with the sample index and 
    the number of samples when queried with `len()`. Normally, the generic dataset classed defined 
    during SLAYER training is supposed to be reused.

    A list of the lables of the samples is stored in `labels` member as the dataset is indexed. This can
    be used to find the ground truth labels to evaluate the network accuracy. Note: it is
    up to the user to reset the `labels` member after every epoch (the network is expected 
    to be run for one epoch only during inference).

    Usage:
    ::
    
        import nxsdk_modules.slayer.src as nxSlayer
        from my_dataset import MyDataset # user's custom basic dataset
        dataset = nxSlayer.auto.s2lDataset(MyDataset(<parms_for_mydataset>))
    """
    # this expects np event and label from dataset
    # np event should have events ordered in x, y, p, t(ms)
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = []
        self.sampleLength = None # if None, process the full sample
        # sampleLength will be modified by Network.setupIO internally
        
    def __getitem__(self, index):
        """Get the event for index'th sample.

        Args:
            index (int): The index of the sample in the dataset.

        Returns:
            event: Spike event object corresponding to the index.
        """
        data, label = self.dataset[index]

        events = AttrDict()
        events.x = data[:,0].astype(int)
        events.y = data[:,1].astype(int)
        events.p = data[:,2].astype(int)
        events.t = data[:,3].astype(int)

        if self.sampleLength is not None:
            events.x = events.x[events.t < self.sampleLength]
            events.y = events.y[events.t < self.sampleLength]
            events.p = events.p[events.t < self.sampleLength]
            events.t = events.t[events.t < self.sampleLength]

        # why not event and label?
        # should do that and create list of labels in s2l
        self.labels.append(label)
        return events

    def __len__(self):
        """Get the length (number of samples) of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)