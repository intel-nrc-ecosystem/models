# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
# 
# Copyright © 2019-2021 Intel Corporation.
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

"""Tools for implementing SLAYER-trained models on Loihi"""

from turtle import delay
import numpy as np
import nxsdk.api.n2a as nx
from nxsdk.arch.n2a.n2board import N2Board
from nxsdk.graph.monitor.probes import *
from nxsdk.net.groups import LogicalNodeGroup
import scipy.sparse as sps
import pickle
import os
import inspect
import errno
import yaml
import io
        
class Slayer2Loihi():
    """
    A class of static helper functions to assist with implementing 
    SLAYER trained models on Loihi.
    """
    @staticmethod
    def getModels():
        """
        Pulls the latest SLAYER models from a public repo and 
        places them in a subfolder under the slayer module 
        
        :returns: The path to the SLAYER models
        :rtype: string
        """
        # latest tested SLAYER models commit
        commitID='a8ddb8dbe45b36726c8a26c0c630b1ebd866d7c9' 
        
        s2lPath = os.path.abspath(os.path.dirname(inspect.getfile(Slayer2Loihi)))
        modelPath = s2lPath + "/slayerLoihiModelZoo"
        repo = 'https://github.com/bamsumit/slayerLoihiModelZoo.git'
            
        # clone into parent directory to share between
        if not os.path.exists(modelPath):
            os.system('git -C {} clone {}'.format(s2lPath, repo))
        else:
            os.system('git -C {} pull'.format(modelPath))

        os.system('git -C {} checkout {}'.format(modelPath, commitID))
        
        return modelPath

    @staticmethod
    def compartmentPrototype(yamlFile):
        """
        Generates a compartment prototype for the model using a yaml file
        
        :param string yamlFile: Path to the slayer network.yaml file
        
        :returns: The compartment prototype
        :rtype: CompartmentPrototype
        """
        
        with open(yamlFile, 'r') as stream:
            params = yaml.safe_load(stream)["neuron"]
        
        if "vMinExp" in params:
            vMinExp=params["vMinExp"]
        else:
            vMinExp=23
        
        compProto = nx.CompartmentPrototype(vThMant=params["vThMant"], 
                                            compartmentVoltageDecay=params["vDecay"],
                                            compartmentCurrentDecay=params["iDecay"],
                                            refractoryDelay=params["refDelay"],
                                            vMinExp=vMinExp,
                                            )
        return compProto
    
    @staticmethod
    def distributeCompartments(layer, corenum, compartmentsPerCore):
        """Distributes compartments across cores, starting on the next available
        core as determined from corenum
        
        :param CompartmentGroup layer: The group of compartments to distribute
        :param float corenum: The last used logicalCoreId
        :param int compartmentsPerCore: The maximum number of compartments per core
        
        :returns: The last used logicalCoreId
        :rtype: float
        """
        
        corenum = np.ceil(corenum)
        for comp in layer:
            comp.logicalCoreId = int(np.floor(corenum))
            corenum = corenum+1/compartmentsPerCore

        return corenum
    
    @staticmethod
    def reorderLayer(layerIn):
        """
        Converts a compartment group from WHC to CHW order.
        
        :param CompartmentGroup layerIn: The layer to reorder.
        
        :returns: The re-ordered layer
        :rtype: CompartmentGroup
        """
        
        net = layerIn.net

        layerOut = net.createCompartmentGroup()
        layerOut.sizeX = layerIn.sizeX
        layerOut.sizeY = layerIn.sizeY
        layerOut.sizeC = layerIn.sizeC
        layerOut.strideX = layerIn.strideX
        layerOut.strideY = layerIn.strideY

        for cc in range(layerIn.sizeC):
            for yy in range(layerIn.sizeY):
                for xx in range(layerIn.sizeX):
                    layerOut.addCompartments(layerIn[xx*layerIn.sizeY*layerIn.sizeC + yy*layerIn.sizeC + cc])

        return layerOut

    @classmethod
    def inputLayer(cls, net, inputSpec, corenum, compartmentsPerCore):
        """Create a new input layer

        :param dict inputSpec: Specifies the dimensions of the layer, "sizeX", "sizeY", "sizeC"
        :param float corenum: The last used logicalCoreId
        :param int compartmentsPerCore: The maximum number of compartments per core
        
        :returns: 
            - layerInput (compartmentGroup): The input compartment group
            - inputConnectionGroup (ConnectionGroup): connections to the input compartment group 
            - corenum (float): The last used logicalCoreId
        """
        
        sizeX = inputSpec["sizeX"]
        sizeY = inputSpec["sizeY"]
        sizeC = inputSpec["sizeC"]        
        
        if "stride" in inputSpec:
            strideX = inputSpec["stride"]
            strideY = inputSpec["stride"]
        else:
            strideX = 1
            strideY = 1
        
        sizeX = int(np.ceil(sizeX/strideX))
        sizeY = int(np.ceil(sizeY/strideY))
        
        connProto = nx.ConnectionPrototype(signMode = nx.SYNAPSE_SIGN_MODE.MIXED, 
                                           weight = 2,
                                           numDelayBits = 0,
                                           numTagBits = 0)

        if "vMinExp" in inputSpec:
            vMinExp = inputSpec["vMinExp"]
        else:
            vMinExp = 23
            
        compProto = nx.CompartmentPrototype(vThMant = 1,
                                            compartmentVoltageDecay = 4095,
                                            compartmentCurrentDecay = 4095,
                                            vMinExp=vMinExp)

        # create the input layer
        layerInput = net.createCompartmentGroup(size=sizeX*sizeY*sizeC, prototype=compProto)

        # add properties used by later convolution
        layerInput.sizeX = sizeX
        layerInput.sizeY = sizeY
        layerInput.sizeC = sizeC
        layerInput.strideX = strideX
        layerInput.strideY = strideY
        
        corenum = cls.distributeCompartments(layerInput, corenum, compartmentsPerCore)

        # create a dummy input connection. This creates the input axons our snips will send spikes to
        inStubGroup = net.createInputStubGroup(size=layerInput.numNodes)
        
        inputConnectionGroup = inStubGroup.connect(layerInput,
                                                   prototype = connProto,
                                                   connectionMask = sps.identity(layerInput.numNodes))

        return layerInput, inputConnectionGroup, corenum

    @classmethod
    def relayLayer(cls, net, relaySpec, corenum, compartmentsPerCore):
        """Create a spike relay layer

        :param net: nxNet object
        :param dict relaySpec: Specifies the dimensions of the layer, "sizeX", "sizeY", "sizeC"
        :param float corenum: The last used logicalCoreId
        :param int compartmentsPerCore: The maximum number of compartments per core
        
        :returns: 
            - layerRelay (compartmentGroup): The relay compartment group
            - corenum (float): The last used logicalCoreId
        """
        sizeX = relaySpec['sizeX']
        sizeY = relaySpec['sizeY']
        sizeC = relaySpec['sizeC']
        if "stride" in relaySpec:
            strideX = relaySpec["stride"]
            strideY = relaySpec["stride"]
        else:
            strideX = 1
            strideY = 1
        
        compProto = nx.CompartmentPrototype(vThMant=1,
                                            compartmentVoltageDecay=4096,
                                            compartmentCurrentDecay=4096)

        layer = net.createCompartmentGroup(size=sizeX*sizeY*sizeC, prototype=compProto)

        layer.sizeX = sizeX
        layer.sizeY = sizeY
        layer.sizeC = sizeC
        layer.strideX = strideX
        layer.strideY = strideY

        corenum = cls.distributeCompartments(layer, corenum, compartmentsPerCore)

        return layer, corenum

    @staticmethod
    def writeHeader(layerOutput, spikesPerPacket, sampleLength):
        """
        Writes the temporary header files which defines constants used by snips.
        
        :param connectionGroup inputConnectionGroup: Connections to the input neurons
        :param compartmentGroup layerOutput: The output compartment group
        :param int spikesPerPacket: How many spikes will be communicated in each channel packet
        :param int sampleLength: The duration of each sample, used to determine how frequently to report output spike counts
        """
        numPacked = int(np.ceil(layerOutput.numNodes/16))
        extraHeaderFilePath = snipDir + '/array_sizes.h'
        f = open(extraHeaderFilePath, "w")
        f.write('/* Temporary generated file for define the size of arrays before compilation */\n')
        f.write('#define spikes_per_packet ' + str(spikesPerPacket)+'\n')
        f.write('#define timesteps_per_sample ' + str(sampleLength)+'\n')
        f.write('#define num_classes ' + str(layerOutput.numNodes)+'\n')
        f.write('#define num_packed ' + str(numPacked)+'\n')

        f.close()
        
    @classmethod
    def prepSpikeInjection(cls, inputConnectionGroup, board, spikesPerPacket, sampleLength, numSnips, regenerateCoreAxon):
        """Determines the core/axon location of the model input axons and sets up snips
        which will later be used to inject spikes
        
        .. note: spike injection makes the simplifying assumption that all input connection lie \
        on the first loihi chip. Input neurons on later chips are not supported at the moment.
        
        :param ConnectionGroup inputConnectionGroup: The input connection group for the model
        :param N2Board board: The compiled board object
        :param int spikesPerPacket: The number of spikes to send per packet
        :param int sampleLength: The number of timesteps per sample
        :param int numSnips: The number of Lakemonts to distribute spikes across
        :param bool regenerateCoreAxon: Whether to load core/axon values from file or precompute them
        """
        net = inputConnectionGroup.net

        # This is incredibly slow. Save the result and load from file
        if regenerateCoreAxon is True:
            import time
            tStart = time.time()
            # Determine the core/axon addresses
            #chip = [int]*inputConnectionGroup.numNodes
            core = [int]*inputConnectionGroup.numNodes
            axon = [int]*inputConnectionGroup.numNodes
            for ii, conn in enumerate(inputConnectionGroup):
                (_, tempChip, tempCore, axon[ii]) = net.resourceMap.inputAxon(conn.inputAxon.nodeId)[0]
                #core[ii] = board.n2Chips[tempChip].n2Cores[tempCore].id
                core[ii] = tempCore
            np.save(tempDir+'/axon.npy', axon)
            np.save(tempDir+'/core.npy', core)
            tEnd = time.time()
            
        else:
            axon = np.load(tempDir+'/axon.npy')
            core = np.load(tempDir+'/core.npy')

        # Setup Spike Injection Snips
        spikeSnips = [None]*numSnips
        spikeChannels = [None]*numSnips

        for ii in range(numSnips):
            spikeSnips[ii] = board.createProcess(name="runSpikes"+str(ii),
                                                 includeDir=snipDir,
                                                 cFilePath=snipDir + "/myspiking"+str(ii)+".c",
                                                 funcName="run_spiking"+str(ii),
                                                 guardName="do_spiking"+str(ii),
                                                 phase="spiking",
                                                 lmtId=ii)

            # spikeSnips[ii] = board.createSnips(phase="spiking",
            #                                    name="runSpikes"+str(ii),
            #                                    includeDir=snipDir,
            #                                    cFilePath=snipDir + "/myspiking"+str(ii)+".c",
            #                                    funcName="run_spiking"+str(ii),
            #                                    guardName="do_spiking"+str(ii),
            #                                    lmtId=ii)

            spikeChannels[ii] = board.createChannel(('spikeAddresses'+str(ii)).encode(), messageSize=16*4, numElements=spikesPerPacket*4)
            
            spikeChannels[ii].connect(None, spikeSnips[ii])

        return spikeChannels, core, axon
    
    @staticmethod
    def prepSpikeData(core, axon, spikesPerPacket, layerInput, dataset, numSamples, sampleLength, numSnips):
        """
        Prepares spike data for injection
        
        .. note: This function makes the simplifying assumption that all input layer neurons lie on 


        :param list-int core: A list of the core id for each input neuron
        :param list-int core: A list of the axon id for each input neuron
        :param int spikesPerPacket: The number of spikes to send per packet
        :param compartmentGroup layerInput: The input compartment group
        :param dataset dataset: The dataset handle, which returns a sample when indexed
        :param int numSamples: How many samples to prepare
        :param int sampleLength: The number of timesteps per sample
        :param int numSnips: The number of Lakemonts to use for spike injection
        
        :returns:
            - snipdata (list-nparray): Spike data returned as a list with one numpy integer array per Lakemont
            - numSteps (int): The duration of the run
        """
        sizeXin = layerInput.sizeX
        sizeYin = layerInput.sizeY
        sizeCin = layerInput.sizeC
        strideX = layerInput.strideX
        strideY = layerInput.strideY      
        
        addresses = []
        timestamps = []
        for ii in range(numSamples):
            # Some code here to load x,y,p,ts
            # convert ts to milliseconds
            #sample, _ = dataset[ii]
            sample = dataset[ii]
            #add some blank time between samples
            if "address" in sample:
                addresses.extend((sample.address).tolist())
            else:
                x = sample.x//strideX
                y = sample.y//strideY
                p = sample.p
                addresses.extend((x*sizeYin*sizeCin + y*sizeCin + p).tolist())
                
            ts = sample.t.astype(int) + ii*(sampleLength)
            timestamps.extend((ts).tolist())

        # numSteps = np.max(timestamps)+1000
        numSteps = numSamples*sampleLength + 1000
        # numSteps = numSamples*sampleLength # LK
        # print("numSamples = ", numSamples) # LK
        # print("sampleLength = ", sampleLength) # LK
        # print("numSteps = ", numSteps) # LK

        # The following two lines of code is necessary for NTIDIGITS samples to work for sample length exceeding 1500 
        # steps. Otherwise the hardware is stuck while reading the spike output. It seems to be expecting more spike 
        # input.
        timestamps.append(numSteps+1000)
        addresses.append(0)
        # print("timestamps = ", timestamps) # LK

        timestamps = np.array(timestamps)
        addresses = np.array(addresses)    
        
        ts = [None]*numSnips
        add = [None]*numSnips
        snipdata = [None]*numSnips
        for ii in range(numSnips):
            ts[ii] = timestamps[ii::numSnips]
            add[ii] = addresses[ii::numSnips]

            timeReserved = int(3<<13) #special value for timer advance

            # combine timestamps and addresses into a single stream
            #round up to a packet + 1 packet
            numEvents = len(add[ii]) + np.max(ts[ii])
            numEvents = int(spikesPerPacket * np.ceil(1+numEvents/spikesPerPacket))

            snipdata[ii] = np.ones((numEvents*2,), dtype=int)*timeReserved
            evTime = 0
            snipIndex = 0
            for jj, loihiTime in enumerate(ts[ii]):
                while(evTime<loihiTime):
                    snipIndex+=2
                    evTime = evTime+1
                try:
                    snipdata[ii][snipIndex] = axon[add[ii][jj]]
                    snipIndex+=1
                    snipdata[ii][snipIndex] = core[add[ii][jj]] 
                    snipIndex+=1
                except:
                    print("ii {} jj {}".format(ii, jj))
                    print(np.array(add).shape)
                    print(np.array(axon).shape)
                    print(add[ii][jj])
                    snipdata[ii][snipIndex] = axon[add[ii][jj]]
                    
            snipdata[ii] = np.left_shift(snipdata[ii][1::2], 16) + np.bitwise_and(snipdata[ii][0::2], (2**16)-1)

        return snipdata, numSteps
    
    @staticmethod
    def sendSpikeData(snipData, spikeChannels, spikesPerPacket):
        """Sends spike data over spikeChannels to the model during runtime. 
        
        :param list(int nparray) snipData: Data to send to snips. One nparray per snip.
        :param list(channel) spikeChannels: A list of channels to send the data over
        :param int spikePerPacket: The number of spikes to send per call to channel.write()
        """
        # send spikes here
        wordsPerPacket = int(spikesPerPacket)
        packedPerPacket = int(wordsPerPacket/16)

        index = 0
        packetNum = 0
        #tStart = time.time()
        while index != len(snipData[0]):
            for ii in range(len(spikeChannels)):
                spikeChannels[ii].write(packedPerPacket, snipData[ii][index:(index+wordsPerPacket)])
            index = index + wordsPerPacket
            packetNum = packetNum + len(spikeChannels)
            
    @staticmethod
    def sendSnipInitialization(initChannels, core, axon):
        """Sends the core/axon addresses of input axons to the snips for use as a LUT

        :param list(Channel) initChannels: The initialization channel for each lakemont
        :param list(int) core: The core address for each address
        :param list(int) axon: The axons address for each address
        """
        # send snip initialization data
        for ii in range(len(core)):
            for channel in initChannels:
                #chip is ignored. For now we assume all input compartments lie on the same chip as the snips
                channel.write(3, [0, core[ii], axon[ii]])   

    @classmethod
    def poolingLayer(cls, layerInput, poolSpec, corenum, compartmentsPerCore):
        """Create a new pooling layer. Assumes that pooling and stride are equal 
        (i.e. non-overlapping pooling regions)

        :param CompartmentGroup layerInput: The input to the pooling layer
        :param dict poolSpec: Specifies the "stride", "connProto", "compProto"
        :param float corenum: The last output of distributeCompartments()
        :param int compartmentsPerCore: The maximum number of compartments per core
        
        :returns:
            - layerOutput (CompartmentGroup) : The compartments of the pooling layer
            - corenum (float): The last used logicalCoreId
        """
        # properties of the input layer
        sizeXin = layerInput.sizeX
        sizeYin = layerInput.sizeY
        sizeCin = layerInput.sizeC
        nInput = sizeCin * sizeYin * sizeXin 
        net = layerInput.net

        # properties of the pooling function
        stride = poolSpec["stride"]
        compProto = poolSpec["compProto"]
        
        if "weight" in poolSpec:
            W = poolSpec["weight"]
        else:
            weightFile = poolSpec["weightFile"]
            W = np.load(weightFile)
         
        if "delay" in poolSpec:
            delay = poolSpec["delay"]
        elif "delayFile" in poolSpec:
            delayFile = poolSpec["delayFile"]
            delay = np.load(delayFile)
        else:
            delay = np.zeros((sizeCin,))

        # properties of the output layer
        sizeXout = int(np.ceil(sizeXin/stride))
        sizeYout = int(np.ceil(sizeYin/stride))
        sizeCout = sizeCin
        nOutput = sizeXout * sizeYout * sizeCout
        
        # indices into the input layer
        x = np.arange(sizeXin, dtype=int)
        y = np.arange(sizeYin, dtype=int)
        c = np.arange(sizeCin, dtype=int)

        xx, yy, cc = np.meshgrid(x, y, c)

        # calculate the source addresses as a linear index
        src = (xx*sizeCin*sizeYin + yy*sizeCin + cc).flat

        # calculate the destination addresses as a linear index
        dst = (np.floor(xx/stride)*sizeCout*sizeYout + np.floor(yy/stride)*sizeCout + cc).astype(int).flat
        
        weight = sps.coo_matrix((W[xx%stride, yy%stride].flat,
                                 (dst,src)), 
                                shape=(nOutput,nInput))
        
        connMask = sps.coo_matrix((np.ones(nInput,),
                                   (dst,src)), 
                                  shape=(nOutput,nInput))
        
        maxD = np.max(delay)
        if maxD != 0:
            numDelayBits = np.ceil(np.log2(maxD))
            disableDelay = False
        else:
            numDelayBits = 0
            disableDelay = True
        
        connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED, 
                                           numDelayBits=numDelayBits,
                                           disableDelay=disableDelay,
                                           numTagBits=0)
        
        delay = sps.coo_matrix((delay[cc.flat],
                                   (dst,src)), 
                                  shape=(nOutput,nInput))

        for ii in [64, 32, 16, 8]:
            if ii>maxD+1:
                compProto.numDendriticAccumulators = ii
        
        layerOutput = net.createCompartmentGroup(size=nOutput, prototype=compProto)
        layerOutput.sizeX = sizeXout
        layerOutput.sizeY = sizeYout
        layerOutput.sizeC = sizeCout
        layerOutput.strideX = 1
        layerOutput.strideY = 1

        layerInput.connect(layerOutput, 
                           prototype=connProto, 
                           connectionMask=connMask, 
                           weight=weight,
                           delay=delay)
        
        corenum = cls.distributeCompartments(layerOutput, corenum, compartmentsPerCore)

        return layerOutput, corenum
    
    @classmethod
    def optimizeWeightBits(cls, weight):
        maxWeight = np.max(weight)
        minWeight = np.min(weight)
        
        if maxWeight < 0:
            signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY
            isSigned = 0
        elif minWeight >= 0:
            signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY
            isSigned = 0
        else:
            signMode=nx.SYNAPSE_SIGN_MODE.MIXED
            isSigned = 1
            
        if signMode ==  nx.SYNAPSE_SIGN_MODE.MIXED:
            posScale = 127/maxWeight
            negScale = -128/minWeight
            scale = np.min([posScale, negScale])
        elif signMode ==  nx.SYNAPSE_SIGN_MODE.INHIBITORY:
            scale = -256/minWeight
        elif signMode ==  nx.SYNAPSE_SIGN_MODE.EXCITATORY:
            scale = 255/maxWeight
                
        scaleBits = np.floor(np.log2(scale)) + isSigned
        
        precisionFound = False
        n = 8
        while (precisionFound is False) and (n>0):
            roundingError = np.sum(np.abs(weight/(2**n) - np.round(weight/(2**n))))
            if roundingError == 0:
                precisionFound = True
            else:
                n-=1
        
        n -= isSigned
        
        numWeightBits = 8 - scaleBits - n
        weightExponent = -scaleBits
        
        weight = np.left_shift(weight.astype(np.int), int(scaleBits))
        
        if numWeightBits == 7:
            numWeightBits = 8
        
        return weight, numWeightBits, weightExponent, signMode

    
    @classmethod
    def convLayer(cls, layerInput, convSpec, corenum, compartmentsPerCore):
        """Create a new convolution layer. Assumes zero padding for 'same' convolution.
        Does not yet support stride.

        :param CompartmentGroup layerInput: The input to the convolution layer
        :param dict convSpec: Specifies "dimX","dimY","dimC" of the filter, 
                             "connProto", "compProto" prototypes of the layer,
                             "weightFile" where the weights can be read in from.
        :param float corenum: The last output of distributeCompartments()
        :param int compartmentsPerCore: The maximum number of compartments per core
        
        :returns:
            - layerOutput (CompartmentGroup): The compartments of the convolution layer
            - corenum (float): The last used logicalCoreId
        """

        # properties of the input layer
        sizeXin = layerInput.sizeX
        sizeYin = layerInput.sizeY
        sizeCin = layerInput.sizeC
        nInput  = sizeXin * sizeYin * sizeCin
        net = layerInput.net

        # properties of convolution function
        compProto = convSpec['compProto']

        convX = convSpec['dimX']                 
        convY = convSpec['dimY']                 
        convC = convSpec['dimC']
        
        padX = convSpec['padX'] if 'padX' in convSpec else None
        padY = convSpec['padY'] if 'padY' in convSpec else None
        
        strideX = convSpec['strideX'] if 'strideX' in convSpec else 1
        strideY = convSpec['strideY'] if 'strideY' in convSpec else 1
        
        dilationX = convSpec['dilationX'] if 'dilationX' in convSpec else 1
        dilationY = convSpec['dilationY'] if 'dilationY' in convSpec else 1
        
        # these values will overwrite whatever was extracted before
        stride = convSpec['stride'] if 'stride' in convSpec else None
        if stride is not None:
            if not hasattr(stride, "__len__"):
                strideX = strideY = stride
            elif len(stride) == 2:
                strideX = stride[1]
                strideY = stride[0]
            else:
                raise Exception(
                    'Could not prase stride information. Got\nstride = {}\nstrideX = {}\nstrideY = {}'.format(
                        stride, strideX, strideY
                    )
                )

        dilation = convSpec['dilation'] if 'dilation' in convSpec else None
        if dilation is not None:
            if not hasattr(dilation, "__len__"):
                dilationX = dilationY = dilation
            elif len(dilation) == 2:
                dilationX = dilation[1]
                dilationY = dilation[0]
            else:
                raise Exception(
                    'Could not prase dilation information. Got\ndilation = {}\ndilationX = {}\ndilationY = {}'.format(
                        dilation, dilationX, dilationY
                    )
                )

        if padX is None:    padX = convX//2
        if padY is None:    padY = convY//2

        pad = convSpec['padding'] if 'padding' in convSpec else None
        if pad is not None:
            if not hasattr(pad, "__len__"):
                padX = padY = pad
            elif len(pad) == 2:
                padX = pad[1]
                padY = pad[0]
            else:
                raise Exception(
                    'Could not prase padding information. Got\npad = {}\npadX = {}\npadY = {}'.format(pad, padX, padY)
                )

        group = convSpec['groups'] if 'groups' in convSpec else 1
        
        W = np.load(convSpec['weightFile']) if 'weightFile' in convSpec else None                 
        W = convSpec['weight'] if 'weight' in convSpec else W
        D = np.load(convSpec['delayFile']) if 'delayFile' in convSpec else np.zeros((convC,))
        D = convSpec['delay'] if 'delay' in convSpec else D
        B = np.load(convSpec['biasFile']) if 'biasFile' in convSpec else None
        B = convSpec['bias'] if 'bias' in convSpec else B

        maxD = np.max(D)
        if maxD != 0:
            numDelayBits = np.ceil(np.log2(maxD))
            disableDelay = False
        else:
            numDelayBits = 0
            disableDelay = True
        
        W, numWeightBits, weightExponent, signMode = cls.optimizeWeightBits(W)
        
        connProto = nx.ConnectionPrototype(signMode=signMode,
                                           numWeightBits=numWeightBits,  
                                           weightExponent=weightExponent,
                                           numDelayBits=numDelayBits,
                                           numTagBits=0,
                                           disableDelay=disableDelay,
                                           compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)
        
        # properties of the output layer
        sizeXout = np.floor((sizeXin + 2*padX - dilationX*(convX - 1) -1)/strideX + 1).astype(int)
        sizeYout = np.floor((sizeYin + 2*padY - dilationY*(convY - 1) -1)/strideY + 1).astype(int)
        sizeCout = convC
        nOutput = sizeCout * sizeYout * sizeXout

        # SUMIT: Not sure what this does
        # Probably it is for the real axonal delay
        for ii in [64, 32, 16, 8]:
            if ii>maxD+1:
                compProto.numDendriticAccumulators = ii
        
        layerOutput = net.createCompartmentGroup(size=nOutput, prototype=compProto)
        layerOutput.sizeX = sizeXout
        layerOutput.sizeY = sizeYout
        layerOutput.sizeC = sizeCout
        layerOutput.strideX = strideX
        layerOutput.strideY = strideY

        yyDst, xxDst = np.meshgrid(np.arange(sizeYout), np.arange(sizeXout))
        xxDst = xxDst.flatten()
        yyDst = yyDst.flatten()
        xxSrc = xxDst * strideX
        yySrc = yyDst * strideY
        
        for grp in np.arange(group): # assuming sizeCout and sizeCin are absolutely divisible by group
            assert sizeCin%group==0, 'sizeCin must be absoultely divisible by group. Found sizeCin={} and group={}'.format(sizeCin, group)
            assert sizeCout%group==0, 'sizeCout must be absoultely divisible by group. Found sizeCout={} and group={}'.format(sizeCout, group)

            # The strategy here is to connect all the neurons in a spatial location to it's inputs
            cDst = np.arange(sizeCout//group) + grp * (sizeCout//group)

            for xDst, yDst, xSrc, ySrc in zip(xxDst, yyDst, xxSrc, yySrc):
                dst = xDst*sizeYout*sizeCout + yDst*sizeCout + cDst
                dst = dst.astype(int)
                dstNodeIds = np.asarray(layerOutput.nodeIds)[dst]
                assert np.sum(np.abs(dstNodeIds-np.arange(np.min(dstNodeIds), (np.max(dstNodeIds)+1)))) == 0, 'dst not contiguous'
                # dstNodeIds = np.tile(dstNodeIds.reshape((-1, 1)), (1, sizeCin)).flatten()
                dstNodeIds = np.tile(dstNodeIds.reshape((-1, 1)), (1, sizeCin//group)).flatten()
                genericDstGrp = LogicalNodeGroup(layerOutput.nodeSet, groupId=-1, name='TempDst', nodeIds=dstNodeIds)

                dy, dx = np.meshgrid(np.arange(convY), np.arange(convX))
                dy = dy.flatten() * dilationY
                dx = dx.flatten() * dilationX
                yy = ySrc + dy - padY
                xx = xSrc + dx - padX
                valid = np.logical_and(
                    np.logical_and(np.greater_equal(yy, 0), np.less(yy, sizeYin)),
                    np.logical_and(np.greater_equal(xx, 0), np.less(xx, sizeXin)),
                )

                for i in range(len(dx)):
                    # spent a lot of time time with 
                    # if valid[i] is True:
                    # Apparently, it does not work with np bool!
                    if valid[i] == True: 
                        # for cSrc in range(sizeCin):
                        cSrc = np.arange(sizeCin//group) + grp * (sizeCin//group)
                        src = xx[i]*sizeYin*sizeCin + yy[i]*sizeCin + cSrc
                        weight = W[cDst, :, int(dy[i]/dilationY), int(dx[i]/dilationY)].flatten()
                        delay  = np.tile(D[cDst].reshape((-1, 1)), (1, sizeCin//group)).flatten()

                        # this is how nxnet creates connections under the hood
                        srcNodeIds=np.asarray(layerInput.nodeIds)[np.tile(src, (sizeCout//group, 1)).flatten()]
                        genericSrcGrp = LogicalNodeGroup(layerInput.nodeSet, groupId=-1, name='TempSrc', nodeIds=srcNodeIds)
                        newConnectionIds = layerOutput.net.connections.createNode(genericSrcGrp, genericDstGrp, connProto)
                        g = layerOutput.net.connectionGroups.createGroup()
                        g.addConnectionsById(newConnectionIds)
                        g.setSynapseState('weight', weight)
                        g.setSynapseState('delay', delay)
                    
        if B is not None:
            layerOutput.setState('biasMant', np.tile(B, sizeYout * sizeXout)) # apply bias per channel. The compartments are ordered in channel first basis
            layerOutput.setState('biasExp', 6 + weightExponent) # set bias exponent same as weight exponent
        
        # The following remains from old code
        corenum = cls.distributeCompartments(layerOutput, corenum, compartmentsPerCore)
        connProto.numDelayBits = 0
        connProto.disableDelay = True
        connProto.weigthLimitExp = 0
        return layerOutput, corenum

    @classmethod
    def fullLayer(cls, layerInput, fullSpec, corenum, compartmentsPerCore):
        """Create a new fully connected layer.

        :param CompartmentGroup layerInput: The input to the fully connected layer
        :param dict fullSpec: Specifies "dim", the number of neurons,
                             "connProto", "compProto" prototypes of the layer,
                             "weightFile" where the weights can be read in from.
        :param float corenum: The last output of distributeCompartments()
        :param int compartmentsPerCore: The maximum number of compartments per core
        
        :returns:
            - layerOutput (CompartmentGroup): The compartments of the fully connected layer
            - corenum (float): The last used logicalCoreId
        """
        # properties of the input layer
        nInput = layerInput.numNodes
        net = layerInput.net

        # properties of the convolution function
        compProto = fullSpec["compProto"]
        dim = fullSpec["dim"]
        
        if "weight" in fullSpec:
            weight = fullSpec["weight"] # LK: * 0.85 to get the same range as in SpyTorch
        else:
            weightFile = fullSpec["weightFile"]
            weight = np.load(weightFile)

        if "delay" in fullSpec:
            D = fullSpec["delay"]
        elif "delayFile" in fullSpec:
            delayFile = fullSpec["delayFile"]
            D = np.load(delayFile)
        else:
            D = np.zeros((dim,)) # LK: nInput -> dim

        if "bias" in fullSpec:
            bias = fullSpec["bias"]
        elif "biasFile" in fullSpec:
            bias = np.load(fullSpec["biasFile"])
        else:
            bias = None
        
        maxD = np.max(D)
        if maxD != 0:
            numDelayBits = np.ceil(np.log2(maxD))
            disableDelay = False
        else:
            numDelayBits = 0
            disableDelay = True

        weight, numWeightBits, weightExponent, signMode = cls.optimizeWeightBits(weight)
        
        # LK: Debug
        # numWeightBits = recNumWeightBits = 8
        # weightExponent = recWeightExponent = 0
        # signMode = recSignMode = 1 # SYNAPSE_SIGN_MODE.MIXED
                    
        if "connProto" in fullSpec:
            connProto = fullSpec["connProto"]
        else:
            connProto = nx.ConnectionPrototype(signMode=signMode, 
                                               numDelayBits=numDelayBits,
                                               disableDelay=disableDelay,
                                               numTagBits=0,
                                               numWeightBits=numWeightBits,
                                               weightExponent=weightExponent,
                                               compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)
            
        nOutput = dim

        if "weightMult" in fullSpec:
            weight *= fullSpec["weightMult"]
        
        delay = np.zeros((nOutput, nInput))
        # for ii in range(nInput):
            # delay[:,ii] = D[ii]
            # the error was here
        for ii in range(nOutput):
            delay[ii,:] = D[ii]

        for ii in [64, 32, 16, 8]:
            if ii>maxD+1:
                compProto.numDendriticAccumulators = ii
                
        layerOutput = net.createCompartmentGroup(size=nOutput, prototype=compProto)

        conn = layerInput.connect(layerOutput,
                                prototype=connProto,
                                weight=weight,
                                delay=delay)
        
        if bias is not None:
            layerOutput.setState('biasMant', bias) # apply bias per neuron
            layerOutput.setState('biasExp', 6 + weightExponent) # set bias exponent same as weight exponent

        corenum = cls.distributeCompartments(layerOutput, corenum, compartmentsPerCore)

        connProto.delay = 0
        connProto.numDelayBits = 0
        connProto.disableDelay = True
        return layerOutput, corenum








    # --------------------------------------------------------------------------------------------

    @classmethod
    def recurrentLayer(cls, layerInput, fullSpec, corenum, compartmentsPerCore):
        """Create a new recurrent layer.
        """
        # properties of the input layer
        nInput = layerInput.numNodes
        net = layerInput.net

        # properties of the convolution function
        compProto = fullSpec["compProto"]
        dim = fullSpec["dim"]
        
        weight = fullSpec["weight"] * 1 # LK
        recWeight = fullSpec["recWeight"] * 1 # LK

        D = np.zeros((dim,)) # LK: nInput -> dim
        bias = None
        numDelayBits = 0
        disableDelay = True

        # weight, numWeightBits, weightExponent, signMode = cls.optimizeWeightBits(weight)
        # recWeight, recNumWeightBits, recWeightExponent, recSignMode = cls.optimizeWeightBits(recWeight)

        # LK: Debug
        numWeightBits = recNumWeightBits = 8
        weightExponent = recWeightExponent = 0
        signMode = recSignMode = 1 # SYNAPSE_SIGN_MODE.MIXED
        
        # print("np.shape(weight) = ", np.shape(weight))
        # print("np.unique(weight) = ", np.unique(weight))
        # print("weight = ", weight)
        # print("numWeightBits = ", numWeightBits)
        # print("weightExponent = ", weightExponent)
        # print("signMode = ", signMode)

        # print("np.shape(recWeight) = ", np.shape(recWeight))
        # print("np.unique(recWeight) = ", np.unique(recWeight))
        # print("recWeight = ", recWeight)
        # print("recNumWeightBits = ", recNumWeightBits)
        # print("recWeightExponent = ", recWeightExponent)
        # print("recSignMode = ", recSignMode)        
        
        connProto = nx.ConnectionPrototype(signMode=signMode, 
                                                numDelayBits=numDelayBits,
                                                disableDelay=disableDelay,
                                                numTagBits=0,
                                                numWeightBits=numWeightBits,
                                                weightExponent=weightExponent,
                                                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)

        recConnProto = nx.ConnectionPrototype(signMode=recSignMode, 
                                                # numDelayBits=numDelayBits,
                                                # disableDelay=disableDelay,
                                                delay=0,
                                                numDelayBits=0,
                                                disableDelay=True,
                                                numTagBits=0,
                                                numWeightBits=recNumWeightBits,
                                                weightExponent=recWeightExponent,
                                                compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)
        
        nOutput = dim
        # print("nOutput = ", nOutput)

        # maxD = np.max(D)
        # for ii in [64, 32, 16, 8]:
        #     if ii>maxD+1:
        #         print("ii = ", ii)
        #         compProto.numDendriticAccumulators = ii
        
        layerOutput = net.createCompartmentGroup(size=nOutput, prototype=compProto)         

        conn = layerInput.connect(layerOutput,
                                prototype=connProto,
                                weight=weight)

        recConn = layerOutput.connect(layerOutput,
                        prototype=recConnProto,
                        weight=recWeight)                        
        
        corenum = cls.distributeCompartments(layerOutput, corenum, compartmentsPerCore)
        # print("corenum = ", corenum)

        # connProto.delay = 0
        # connProto.numDelayBits = 0
        # connProto.disableDelay = True

        # recConnProto.delay = 1
        # recConnProto.numDelayBits = 3
        # recConnProto.disableDelay = False

        return layerOutput, corenum

    # --------------------------------------------------------------------------------------------











    @staticmethod
    def saveBoard(board, filename, dummyProbes):
        """
        Writes an N2Board to a file which can be reloaded later.
        
        :param N2Board board: The board to be written
        :param string filename: The name of the file to write (without extension)
        :param list-probes dummyProbes: The dummy probes used to setup spike counters
        """
        # Determine the information required to later load the board
        boardId = board.id
        numChips = len(board.n2Chips)
        numCoresPerChip = [None]*numChips
        numSynapsesPerCore = [None]*numChips
        for ii in range(numChips):
            numCoresPerChip[ii] = len(board.n2Chips[ii].n2Cores)
            numSynapsesPerCore[ii] = [None]*numCoresPerChip[ii]
            for jj in range(numCoresPerChip[ii]):
                numSynapsesPerCore[ii][jj] = board.n2Chips[ii].n2Cores[jj].synapses.numNodes

        with open(tempDir+ '/'+filename + '.pkl', 'wb') as fname:
            pickle.dump([dummyProbes, boardId, numChips, numCoresPerChip, numSynapsesPerCore], fname)
            
        # Dump the NeuroCores
        board.dumpNeuroCores(tempDir+ '/'+filename + '.board')
    
    @staticmethod
    def initBoard(filename):
        """
        Initializes a board object from file with the correct number of chips/cores/synapses.
        
        :param string filename: The name of the file to load from (without extension)
        
        :return N2Board board: The board object
        :returns:
            - board (N2Board): The N2Board object
            - dummyProbes (list-probes): A list of probes used to setup spike counters
        """
        with open(tempDir+ '/'+ filename + '.pkl', 'rb') as fname:
            # options, lmtOptions, dummyProbes, boardId, numChips, numCoresPerChip, numSynapsesPerCore = pickle.load(fname)
            dummyProbes, boardId, numChips, numCoresPerChip, numSynapsesPerCore = pickle.load(fname)
            
        # board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore, options=options, lmtOptions=lmtOptions)
        board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)
        
        return board, dummyProbes
        
    @staticmethod
    def loadBoard(board, filename):
        """
        Loads neurocore state from file.
        
        :param N2Board board: The board object for which neurocore state should be loaded
        :param string filename: The name of the file to load from (without extension)
        """
        board.loadNeuroCores(tempDir+ '/'+filename + '.board')
        
    @staticmethod
    def setupSpikeCounters(outputLayer):
        """
        Creates dummy probes to setup the spike counters
        
        :param CompartmentGroup outputLayer: The compartments of the last layer
        
        :returns: The dummy probes used to setup spike counters
        :rtype: list-probes
        """
        probeCond = SpikeProbeCondition(tStart=100000000)
        dummyProbes = outputLayer.probe(nx.ProbeParameter.SPIKE, probeCond)
        return dummyProbes
    
    @staticmethod
    def prepSpikeCounter(board, numSamples, numClasses, corenum):
        """
        Sets up the spike counter snip.
        
        :param N2Board board: The N2 Board object
        :param int numSamples: The number of samples to be processed in this run
        :param float corenum: The logical coreId of the output layer
        
        :returns: The channel over which results will be received
        :rtype: channel
        """
        # Infer which chip the output neurons lie on from the corenum
        chipId = int(corenum/128)
        
        # Get the spikes back from loihi
        runMgmtProcess = board.createProcess("runMgmt",
                                             includeDir=snipDir,
                                             cFilePath = snipDir + "/runmgmt.c",
                                             funcName = "run_mgmt",
                                             guardName = "do_run_mgmt",
                                             phase = "mgmt",
                                             lmtId=0,
                                             chipId=chipId)

        numPacked = int(np.ceil(numClasses/16))

        # Create a channel named spikeCntr to get the spikes count information from Lakemont
        spikeCntrChannel = board.createChannel(b'nxspkcntr', messageSize=16*4*numPacked, numElements=numSamples*numPacked+2)

        # Connecting spikeCntr from runMgmtProcess to SuperHost which is receiving spike count in the channel
        spikeCntrChannel.connect(runMgmtProcess, None)
        
        return spikeCntrChannel

    @staticmethod
    def getResults(spikeCntrChannel, numSamples, numClasses, dummyProbes, saveResults=True):
        """
        Reads results from the spike counter channel.
        
        :param channel spikeCntrChannel: The channel over which results will be received
        :param int numSamples: The number of results to receive
        :param int numClasses: The number of classes in each result
        :param list-probes dummyProbes: The dummy probes used to setup spike counters
        :param bool saveResult: Option to save results to file
        
        :returns: The spike count results for the run
        :rtype: nparray-int
        """
        numPacked = int(np.ceil(numClasses/16))
        
        results = spikeCntrChannel.read(numSamples)
        # results = []
        # for ii in range(numSamples):    
        #     print(ii)
        #     results.extend(spikeCntrChannel.read(1))

        results = np.array(results).reshape((numSamples,numPacked*16)) #16 due to the "packed" data type
        results = results[:,:numClasses]  #remove the extras
        
        counterIds = [prb.n2Probe.counterId-32 for prb in dummyProbes[0].probes]
        results = results[:, counterIds]

        if saveResults is True:
            np.savetxt(tempDir+'/spikesCounterOut.txt', results, fmt='%i')
        return results
    
    @staticmethod
    def checkAccuracy(labels, results):
        """
        Compares results to dataset labels to calculate an accuracy. 
        In the case of a split decision, an accuracy of 1/winners is assigned
        for the sample.
        
        :param list-int labels: The dataset labels
        :param nparray-int results: The results obtained from getResults()
        
        :returns: The final classification accuracy
        :rtype: float 
        """
        numSamples = results.shape[0]
        classification = [None]*numSamples
        numCorrect = 0
        for ii in range(numSamples):
            maxLogicals = np.amax(results[ii,:])==results[ii,:]
            classification[ii] = np.where(maxLogicals)[0]

            if labels[ii] in classification[ii]:
                numCorrect += 1/len(classification[ii])
        accuracy = numCorrect/len(labels)
        return accuracy
    
snipDir = os.path.abspath(os.path.dirname(inspect.getfile(Slayer2Loihi)) + "/snips")
tempDir = "temp"
try:
    os.mkdir(tempDir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise e