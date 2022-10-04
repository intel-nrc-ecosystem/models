# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021 Intel Corporation.
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

import os
import sys

import time
import numpy as np
import pickle
import inspect

import nxsdk.api.n2a as nx
from .slayer2loihi import Slayer2Loihi as s2l
from . import s2lWrapper as auto
from nxsdk.graph.processes.phase_enums import Phase
from nxsdk.arch.n2a.n2board import N2Board

class RelayInput:
    """This is a basic module for relaying input to multiple networks. 
    """
    def __init__(self, net, sizeX, sizeY=1, sizeC=1, corenum=None):
        """Creates a relay input layer of given dimension. Each
        instance of this module can connect to up to three other modules.

        Args:
            net : nxNet board object
            sizeX (int): x dimension of the layer.
            sizeY (int, optional): y dimension of the layer. Defaults to 1.
            sizeC (int, optional): channel dimension of the layer. Defaults to 1.
            corenum (int, optional): Logical core id to place the instance. If corenum is not
                specified, it will act as input layer. Otherwise it will act as a relay layer. 
                Defaults to None.
        """
        self.net = net
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.sizeC = sizeC
        inputSpec = {
                'sizeX': self.sizeX,
                'sizeY': self.sizeY,
                'sizeC': self.sizeC,
            }
        self.inputConnectionGroup = None
        self.numBranch = 0
        if corenum is None:
            self.layer, self.inputConnectionGroup, self.corenum = s2l.inputLayer(self.net, inputSpec, corenum=0, compartmentsPerCore=1024)
        else:
            self.layer, self.corenum = s2l.relayLayer(self.net, inputSpec, corenum=corenum, compartmentsPerCore=1024)

        self.connProto = nx.ConnectionPrototype(
                signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                weight=2,
                numDelayBits=0,
                numTagBits=0,
            )
        
    def connectTo(self, layer):
        """A method to conenct relay layer to another layer.

        Args:
            layer : Target layer to connect the relay layer.
        """
        assert self.sizeX == layer.sizeX, "sizeX of source and destination must match."
        assert self.sizeY == layer.sizeY, "sizeY of source and destination must match."
        assert self.sizeC == layer.sizeC, "sizeC of source and destination must match."

        # this is because 1024 compartments can route to only 4096 axons (4x)
        assert self.numBranch < 4, 'number of branch relays must be less than or equal to 4. Found {} existing branches'.format(numBra)

        self.numBranch += 1 

        self.layer.connect(
            layer, 
            prototype=self.connProto,
            weight = 2*np.eye(self.sizeX * self.sizeY * self.sizeC),
            connectionMask = np.eye(self.sizeX * self.sizeY * self.sizeC),
        )

class Network(auto.Network):
    """Class similar to `nxSlayer.auto.Network`. Points to different snip codes that allow
    independent channels to be created to different chip and lakemonts.
    """
    def __init__(self, *args, **kwargs):
        """Initializer function. It has same arguments as `nxSlayer.auto.Network`
        """
        super(Network, self).__init__(*args, **kwargs)
        self.multiBoard = None
        self.streamOnce = False

    def writeHeader(self, spikesPerPacket, sampleLength):
        """Writes the temporary header files which defines constants used by snips.

        Args:
            spikesPerPacket (int): Number of spikes to send per packet.
            sampleLength (int): Length of the sample.
        """
        numPacked = int(np.ceil(self.outputLayer.numNodes/16))
        headerFile = snipDir + '/array_sizes.h'
        with open(headerFile, 'w') as f:
            f.write('/* Temporary generated file for define the size of arrays before compilation */\n')
            f.write('#define spikes_per_packet ' + str(spikesPerPacket) + '\n')
            f.write('#define timesteps_per_sample ' + str(sampleLength) + '\n')
            f.write('#define num_classes ' + str(self.outputLayer.numNodes) + '\n')
            f.write('#define num_packed ' + str(numPacked) + '\n')
            numPackets = 0
            for data in self.spikeData:
                numPackets = max(numPackets, int(np.ceil(len(data) / spikesPerPacket)))
            if self.streamOnce is True:
                # numPackets = 4 # 4578 for NTIDIGITS
                if numPackets*spikesPerPacket > 8192:
                    numPackets = 8192//spikesPerPacket 
                f.write('#define num_packets ' + str(numPackets) + '\n')
                f.write('#define STREAM_ONCE\n')
                self.numPackets = numPackets

    def setupCoreAxon(self, regenerateCoreAxon=True):
        """Determines the core/axon location of the model input axons and sets up snips
        which will later be used to inject spikes.

        Args:
            regenerateCoreAxon (bool, optional): If True, it will regenerate the core and
            axon table of the model. Otherwise, it will try to read from saved file. 
            Defaults to True.

        Returns:
            core (numpy array): input cores of the model.
            axon (numpy array): input axons of the model.
        """
        net = self.net
        # tempDir = 'temp'

        if regenerateCoreAxon is True:
            import time
            tStart = time.time()
            # Determine the core/axon addresses
            #chip = [int]*self.inputConnectionGroup.numNodes
            core = [int]*self.inputConnectionGroup.numNodes
            axon = [int]*self.inputConnectionGroup.numNodes
            for i, conn in enumerate(self.inputConnectionGroup):
                (_, tempChip, tempCore, axon[i]) = net.resourceMap.inputAxon(conn.inputAxon.nodeId)[0]
                #core[i] = board.n2Chips[tempChip].n2Cores[tempCore].id
                core[i] = tempCore
            np.save(tempDir+'/axon.npy', axon)
            np.save(tempDir+'/core.npy', core)
            tEnd = time.time()
        else:
            axon = np.load(tempDir+'/axon.npy')
            core = np.load(tempDir+'/core.npy')

        return core, axon

    def setupSpikeChannels(self, spikesPerPacket, numChips=1, numLmts=3):
        """Sets up spike channels for communication with lakemonts.

        Args:
            spikesPerPacket (int): Number of spikes to send per packet.
            numChips (int, optional): Number of chips receiving input. Defaults to 1.
            numLmts (int, optional): Number of lakemonts per chip receiving input. Defaults to 3.

        Returns:
            spikeChannels: A 2D list (numChips x numLmts) of spike communication channels.
        """
        print(snipDir)

        # Setup Spike Injection Snips
        # spikeSnips = [[None]*numLmts]*numChips
        # spikeChannels = [[None]*numLmts]*numChips
        spikeSnips = [[None]*numLmts for _ in range(numChips)]
        spikeChannels = [[None]*numLmts for _ in range(numChips)]

        board = self.multiBoard if self.multiBoard is not None else self.board

        for chip in range(numChips):
            for lmt in range(numLmts):
                spikeSnips[chip][lmt] = board.createSnip(
                        phase = Phase.EMBEDDED_SPIKING,
                        name = 'runSpikes_{}_{}'.format(chip, lmt),
                        includeDir = snipDir,
                        cFilePath = snipDir + '/myspiking.c',
                        funcName = 'run_spiking',
                        guardName = 'do_spiking',
                        lmtId = lmt,
                        chipId = chip,
                    )
                
                spikeChannels[chip][lmt] = board.createChannel(
                        name = 'spikeAddresses_{}_{}'.format(chip, lmt).encode(),
                        messageSize = 16*4, 
                        numElements = spikesPerPacket*4,
                    )

                spikeChannels[chip][lmt].connect(None, spikeSnips[chip][lmt])                

        return spikeChannels

    def prepSpikeCounter(self, chipId=0):
        """Set up spike counter channel to read output spike count

        Args:
            chipId (int, optional): The chip to read the output from. Defaults to 0.

        Returns:
            spikeCntrChannel: Spike counter channel.
        """
        board = self.multiBoard if self.multiBoard is not None else self.board
        # Get the spikes back from loihi
        runMgmtProcess = board.createSnip(
                phase = Phase.EMBEDDED_MGMT,
                includeDir = snipDir,
                cFilePath = snipDir + '/runmgmt.c',
                funcName = 'run_mgmt',
                guardName = 'do_run_mgmt',
                lmtId = 0,
                chipId = chipId
            )

        numPacked = int(np.ceil(self.outputLayer.numNodes/16))

        # Create a channel named spikeCntr to get the spikes count information from Lakemont
        spikeCntrChannel = board.createChannel('nxspkcntr_{}'.format(chipId).encode(), messageSize=16*4*numPacked, numElements=self.numSamples*numPacked+2)

        # Connecting spikeCntr from runMgmtProcess to SuperHost which is receiving spike count in the channel
        spikeCntrChannel.connect(runMgmtProcess, None)
        
        return spikeCntrChannel

    def setupIO(self, dataset, spikesPerPacket=2048, numChips=1, numLmts=3, blankTime=100, spikeTime=None, regenerateCoreAxon=True):
        """Sets up input output processes for sending spikes, spike messages, and receiving spike count.

        Args:
            dataset (nxSlayer.s2lDataset): slayer dataset object.
            spikesPerPacket (int, optional): Number of spikes to send per packet. Defaults to 2048.
            numChips (int, optional): Number of chips receiving input. Defaults to 1.
            numLmts (int, optional): Number of lakemonts per chip receiving input. Defaults to 3.
            blankTime (int, optional): Amount of blank time between samples. Defaults to 100.
            spikeTime ([type], optional): Length of sample. If it is unspecified, the value will be read from netConfig. Defaults to None.
            regenerateCoreAxon (bool, optional): Flag to indicate weather to regenerate input core and axon list. Defaults to True.
        """
        self.numSamples = len(dataset)

        self.spikesPerPacket = spikesPerPacket

        if spikeTime is None:
            spikeTime = self.netConfig['simulation']['tSample'][()]
        print('Using per sample spike time: {}steps (+ {}steps gap)'.format(spikeTime, blankTime))
        sampleLength = spikeTime + blankTime

        dataset.sampleLength = spikeTime
        
        self.core, self.axon = self.setupCoreAxon(regenerateCoreAxon=regenerateCoreAxon)
        self.spikeChannels = self.setupSpikeChannels(spikesPerPacket, numChips=numChips, numLmts=numLmts)

        self.spikeData, self.numSteps = s2l.prepSpikeData(
            self.core, self.axon, spikesPerPacket, self.inputLayer, 
            dataset, self.numSamples, sampleLength, numLmts
        )

        self.writeHeader(spikesPerPacket, sampleLength)

        if self.probeOutput is True:
            self.spikeCntrChannel = self.prepSpikeCounter(numChips-1)

        if self.savedBoardName is not None:
            s2l.loadBoard(self.board, self.savedBoardName)

    def sendSpikeData(self):
        """Sends spike data over spikeChannels to the model during runtime.
        """
        wordsPerPacket = int(self.spikesPerPacket)
        packedPerPacket = int(wordsPerPacket/16)

        index = 0
        while index != len(self.spikeData[0]):
            for lmt in range(len(self.spikeChannels[0])):
                for core in range(len(self.spikeChannels)):
                    self.spikeChannels[core][lmt].write(packedPerPacket, self.spikeData[lmt][index:(index+wordsPerPacket)])
            index = index + wordsPerPacket
            if self.streamOnce is True:
                if index >= self.numPackets * wordsPerPacket:
                    break

    def run(self):
        """Run the network

        Returns:
            results : Output spike counts if probeOutput is set to True.
        """
        self.board.start()
        self.board.run(self.numSteps, aSync=True)
        tStart = time.time()
        self.sendSpikeData()
        if self.probeOutput is True:
            self.results = s2l.getResults(self.spikeCntrChannel, self.numSamples, self.outputLayer.numNodes, self.dummyProbes)
            print('Gathered results')

        self.board.finishRun()
        self.board.disconnect()
        tEnd = time.time()
        print('Completed {} timesteps in {:.2f} seconds'.format(self.numSteps, tEnd-tStart))
        return self.results


class MultiChipNetwork():
    """Creates copies of a network across multiple chips. It assumes a network fits within a single chip.
    """
    def __init__(self, netConfigFile, probeOutput=True, noDelay=False):
        """Initializes the multichip network object.

        Args:
            netConfigFile : Trained hdf5 network config file from SLAYER.
            probeOutput (bool, optional): Flag to indicate whether to probe the output neurons or not. Defaults to True.
            noDelay (bool, optional): Flag to indicate whether to ignore delay in the network or not. Defaults to False.
        """
        self.netConfigFile = netConfigFile
        self.probeOutput = probeOutput
        self.noDelay = noDelay
        self.net = None
        self.inputSrc = None
        self.netCopies = []
        self.numCopiesPerChip = 1
        self.numChips = 1
        self.numLmts = 1
        self.corenum = 0
        self.coresPerNet = None
        self.coresPerRepeater = None
        self.streamOnce = False
        self.board = None
    
    def create(self, customCompPerCore=[], numChips=1, streamOnce=False, maxCopies=128):
        """Creates the multi-chip network. 
        
        Note: The nxNet object created is a single chip copies of the SLAYER network.
        It should be overwritten to use this module for any general net.

        Args:
            customCompPerCore (list, optional): Custom compartment allocation for a layer/s. 
                If unspecified, defaults to automatic assignment. Defaults to [].
            numChips (int, optional): Number of chips to replicate the network. Defaults to 1.
            streamOnce (bool, optional): Flag to indicate whether to stream the input once or continuously stream the full dataset.
                It is recommended to activate streamOnce option while benchmarking energy and time. This will remove
                the spike communication bottleneck and allows the network to run as fast as possible. The input spike
                sequence is looped over and over to allow stable power readings. Defaults to False.
            maxCopies (int, optional): Maximum network copies per chip. Defaults to 128.

        Returns:
            corenum: Number of cores used in a single chip.
        """
        self.numChips = int(numChips)
        self.streamOnce = streamOnce
        if self.streamOnce is True:
            if self.probeOutput is True:
                print('Disabling output probing (probeOutput) in single streaming mode (streamOnce = True)')
                self.probeOutput = False # in streamOnce mode, do not probe output
        
        self.net = Network(self.netConfigFile, probeOutput=False, noDelay=self.noDelay)
        self.inputSrc = RelayInput(
                self.net.net,
                sizeX = self.net.netConfig['layer']['0']['shape'][2], 
                sizeY = self.net.netConfig['layer']['0']['shape'][1], 
                sizeC = self.net.netConfig['layer']['0']['shape'][0], 
            )

        # create single chip copies
        corenum = self.net.create(corenum=self.inputSrc.corenum, customCompPerCore=customCompPerCore, relayInput=True)
        self.inputSrc.connectTo(self.net.inputLayer)
        # overwrite inputLayer and inputConnectionGrp in net with inputSrc
        # this should allow to reuse the compilation and setupIO methods of net
        self.net.inputLayer = self.inputSrc.layer
        self.net.inputConnectionGroup = self.inputSrc.inputConnectionGroup

        # calculate the number of copies to fit in a chip
        self.coresPerNet = np.ceil(corenum) - np.ceil(self.inputSrc.corenum)
        self.coresPerRepeater = np.ceil(self.inputSrc.corenum)
        self.numCopiesPerChip = int(np.floor(128 / (self.coresPerNet + self.coresPerRepeater / 3))) # one repeater for 3 network copies
        self.numCopiesPerChip = min(maxCopies, self.numCopiesPerChip)

        print('# cores per network :', self.coresPerNet)
        print('# cores per repeater:', self.coresPerRepeater)
        print('Creating {} additional copies per chip.'.format(self.numCopiesPerChip - 1))

        repeater = self.inputSrc
        for i in range(1, self.numCopiesPerChip):
            print('Copy {}'.format(i))
            if i%3==0:
                # create new repeater block every 3 newtwork
                oldRepeater = repeater
                repeater = RelayInput(
                        self.net.net,
                        sizeX = self.net.netConfig['layer']['0']['shape'][2], 
                        sizeY = self.net.netConfig['layer']['0']['shape'][1], 
                        sizeC = self.net.netConfig['layer']['0']['shape'][0], 
                        corenum = corenum, # when corenum is specified, it acts as a repeater block (not input)
                    )
                oldRepeater.connectTo(repeater.layer)    
                corenum = repeater.corenum
                print('Repeater added at core {}'.format(int(corenum)))    
            self.netCopies.append(
                Network(
                    self.netConfigFile, 
                    net=self.net.net, 
                    probeOutput=(i==self.numCopiesPerChip-1 and self.probeOutput),
                    noDelay = self.noDelay,
                )
            ) # probe the output from the last copy if desired
            corenum = self.netCopies[-1].create(corenum=corenum, customCompPerCore=customCompPerCore, relayInput=True)
            repeater.connectTo(self.netCopies[-1].inputLayer)

        self.net.streamOnce  = self.streamOnce
        self.net.probeOutput = self.probeOutput
        if self.numCopiesPerChip > 1:
            self.net.dummyProbes = self.netCopies[-1].dummyProbes
        
        if self.numCopiesPerChip == 1:
            print('Not using repeater layers. Recreating Layer')
            self.net.net = nx.NxNet()
            corenum = self.net.create(customCompPerCore=customCompPerCore)

        self.corenum = int(corenum)
        return corenum

    def compile(self, boardName, regenerateBoard=True):
        """Compile the network. Single chip network is compiled and saved to file here. The
        primitives to copy the single chip network to multiple chips is done here.

        Args:
            boardName (string): Name of the board.
            regenerateBoard (bool, optional): Flag to indicate weather to regenerate input core and axon list. Defaults to True.

        Returns:
            board: nxNet board object.
        """
        fileName = boardName + '.board'
        self.regenerateBoard = regenerateBoard
        if regenerateBoard is True:
            # compile the single chip network
            print('Starting compilation of single chip copies')
            tStart = time.time()
            compiler = nx.N2Compiler()
            board = compiler.compile(self.net.net)
            tEnd = time.time()
            print('Completed compilation of single chip copies in {:.2f} seconds'.format(tEnd-tStart))
            
            # save single chip board
            board.dumpNeuroCores(fileName)
            with open(boardName + '.probe', 'wb') as f:
                pickle.dump(self.net.dummyProbes, f)

            print('Saved single chip copy to ' + fileName)
            
            board.finishRun()
            board.disconnect()
        else:
            with open(boardName + '.probe', 'rb') as f:
                self.net.dummyProbes = pickle.load(f)

        board = N2Board(id=1, numChips=self.numChips, numCores=[self.corenum]*self.numChips)
        # write headers for host snip
        with open(snipDir + '/host_snip_load_reg.h', 'w') as f:
            f.write('/* Temporary file generated for host_snip before comilation */')
            f.write('#define FILE_NAME "{}"\n'.format(os.path.abspath(fileName)))
            f.write('#define NUM_CHIPS {}\n'.format(self.numChips))

        # create host snip
        board.createSnip(
            phase = Phase.HOST_PRE_EXECUTION,
            cppFile = snipDir + '/host_snip_load_reg.cpp'
        )

        self.board = board
        self.net.multiBoard = board
        return board

    def setupIO(self, dataset, numLmts=3, spikesPerPacket=2048, blankTime=100, spikeTime=None):
        """Sets up input output processes for sending spikes, spike messages, and receiving spike count.

        Args:
            dataset (nxSlayer.s2lDataset): slayer dataset object.
            numLmts (int, optional): Number of lakemonts per chip receiving input. Defaults to 3.
            spikesPerPacket (int, optional): Number of spikes to send per packet. Defaults to 2048.
            blankTime (int, optional): Amount of blank time between samples. Defaults to 100.
            spikeTime ([type], optional): Length of sample. If it is unspecified, the value will be read from netConfig. Defaults to None.
        """
        self.numLmts = numLmts
        self.net.setupIO(
            dataset = dataset, 
            numChips = self.numChips,
            numLmts = self.numLmts,
            spikesPerPacket = spikesPerPacket, 
            blankTime = blankTime,
            spikeTime = spikeTime,
            regenerateCoreAxon = self.regenerateBoard,
        )

    def run(self, numSteps=None):
        """Run the network

        Args:
            numSteps ([type], optional): [description]. Defaults to None.

        Returns:
            results : Output spike counts if probeOutput is set to True.
        """
        if self.streamOnce is False or numSteps is None:
            numSteps = self.net.numSteps
        results = None
        self.board.start()
        self.board.run(numSteps, aSync=True)

        print('Timer starts here')
        tStart = time.time()
        self.net.sendSpikeData()
        
        if self.probeOutput is True:
            results = s2l.getResults(self.net.spikeCntrChannel, self.net.numSamples, self.net.outputLayer.numNodes, self.net.dummyProbes)
            print('Gathered results')

        self.board.finishRun()
        self.board.disconnect()
        tEnd = time.time()
        print('Completed {} timesteps in {:.2f} seconds'.format(numSteps, tEnd-tStart))

        return results

snipDir = os.path.abspath(os.path.dirname(inspect.getfile(RelayInput)) + '/benchmarkSnips')
tempDir = 'temp'
os.makedirs(tempDir, exist_ok=True)

