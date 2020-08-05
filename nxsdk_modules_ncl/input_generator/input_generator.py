"""
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2020 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
"""
import json
import pickle
import numpy as np
from typing import Callable
from nxsdk.composable.collections import Processes
from nxsdk.graph.graph import Graph
from nxsdk.graph.processes.phase_enums import Phase
from nxsdk.composable.port_impl import StateOutputPort
from nxsdk.composable.interfaces.process import Process
from nxsdk.composable.abstract_composable import AbstractComposable
from jinja2 import Environment, FileSystemLoader
from nxsdk.logutils.nxlogging import get_logger
import operator
from functools import reduce
from typing import Tuple
import os
from nxsdk.composable.interfaces.composable_enums import PortType, AddressesStorageType
from nxsdk.composable.interfaces.process_aggregator_interface import AbstractProcessAggregator
from nxsdk.composable.interfaces.addresses_storage import RangeAddressesStorage, RangeAddressWithVariableStrideStorage
from nxsdk.graph.processes.embedded.embedded_snip import EmbeddedSnip

"""
InputGenerator is a layer which can be used to encode data and inject data
in neurocore.
"""


class InputGenerator(AbstractComposable):
    """An input encoder is a layer to encode and inject input np.ndarray data"""

    def __init__(
            self,
            shape: Tuple,
            startRunning: int = 1,
            interval: int = 3000,
            packetSize: int = 1024,
            numSnipsPerChip: int = 1,
            encoderFunction: Callable[[np.ndarray], np.ndarray] = None,
            decoderSnip: str = None,
            encoderType: PortType = PortType.STATE,
            queueSize: int = 64):
        """
        InputGenerator is layer which can encode and inject data into neurocore.
        Input Generator performs following tasks:
        1. Input encoding to compress data in most suitable form for channel communication
        2. Writing input to channel
        3. Input decoding on host or embedded CPU
        4. Input injection to neuro core on host or embedded CPU via SNIP code

        :param name: Name of the Input Encoder
        :param shape: Shape of the input
        :param encoderType: Type of Encoder (SPIKE or STATE)
        :param packetSize: Size of one packet
        :param numSnipsPerChip: Number of snips per chip on which input layer is present
        :param encoderFunction: Function to encode the data
        :param decoderSnip: Snip of type EmbeddedSnip to decode the injected data
        :param encoderType: Injecting STATE or SPIKE
        :param queueSize: Number of inputs that can be queued without blocking
        """
        super().__init__()

        self._build(shape=shape,
                    startRunning=startRunning,
                    interval=interval,
                    packetSize=packetSize,
                    numSnipsPerChip=numSnipsPerChip,
                    encoderFunction=encoderFunction,
                    decoderSnip=decoderSnip,
                    encoderType=encoderType,
                    queueSize=queueSize)

    def _build(self, *args, **kwargs):
        """Builds the input encoder using the parameters passed in"""
        self.shape = kwargs["shape"]
        self.encoderType = kwargs["encoderType"]
        self.queueSize = kwargs["queueSize"]
        self.numSnipsPerChip = kwargs["numSnipsPerChip"]
        self.startRunning = kwargs["startRunning"]
        self.interval = kwargs["interval"]
        self.packetSize = kwargs["packetSize"]
        self.dataSize = reduce(operator.mul, self.shape, 1)
        self._logger = get_logger("NET.INE")

        # Holds the partition and map information
        # injectionPointsPerChip is the dictionary mapping chipId to
        # number of injection points for that chip
        self._injectionPointsPerChip = {}
        # List of size ( numChips * numSnipsPerChip) consisting of injection points for all individual snips
        self._injectionPointsPerSnip = []
        # List to store tuple consisting chipId, numData to be send
        self._dataOrder = []
        # Dictionary to store snip id to chip mapping
        self._chipIdForSnip = {}
        # List of chips being used
        self._chips = []
        # Flag to indicate if input generator is compiled and ready
        self._compiled = False
        # List of compressed representation of resource map of cxs
        self._rangeAddress = []
        # List of list of compressed reprsentation of resource maps of cxs per snip
        # Size of this list will be ( numChips * numSnipsPerChip)
        # Ultimately gets shipped to the snip via metaDataChannel
        self._metaData = []
        # List consisting of number of packets per snip
        self._packetPerSnip = []

        # List of Channels
        #   dataChannel     : Used to send injection data
        #   infoChannel     : Used to send packetSize info ToDo: Use jinja2
        #   dataSizeChannel : Used to send the numPackets info ToDo: Use jinja2
        #   metaDataChannel : Used to send range based address
        self._dataChannels = []
        self._infoChannels = []
        self._dataSizeChannels = []
        self._metaDataChannels = []

        # Portname
        self._portName = "output"

        # Bias Exp ( should be same for all Cxs)
        self.biasExp = 0

        if bool(kwargs["encoderFunction"]) != bool(kwargs["decoderSnip"]):
            raise ValueError(
                "Either both encoderFunction and decoderSnip should be specified"
                "or both should be specified. Just one among them cannot be specified")

        if kwargs["encoderFunction"] and kwargs["decoderSnip"]:
            self.encoderFunction = kwargs["encoderFunction"]
            if not isinstance(kwargs["decoderSnip"], str):
                raise ValueError(
                    "Decoder Snip should be of type str as it is path to template c file")
            self.decoderSnip = kwargs["decoderSnip"]
        else:
            self.encoderFunction = self._defaultEncoder
            if self.encoderType:
                self.decoderSnip = os.path.join(
                    os.path.dirname(__file__), 'templates', 'inject_state.c.template')
            else:
                self.decoderSnip = os.path.join(
                    os.path.dirname(__file__), 'templates', 'inject_spike.c.template')

        self._createOutputPort()
        self._addSnipPlaceholder()

    def __str__(self):
        """Returns string representation of the InputGenerator"""
        strRep = "Port Name : {} Shape of Input Encoder is : {} Encoder Type is : {}" \
                 " DataSize is : {} Packet Size is : {}".format(
                     self._portName, self.shape, self.encoderType, self.dataSize, self.packetSize)
        return strRep

    def _addSnipPlaceholder(self):
        """
        As Snips can be only be attached post mapping phase, these processes are placeholders.
        Adds the inputEncoder process in mgmt phase or spiking phase depending on the encoder type
        """
        if self.encoderType:
            includeDir = os.path.dirname(os.path.realpath(__file__))
            funcName = "runMgmt"
            guardName = "doMgmt"
            snip = Process(
                name="inputEncoder",
                phase=Phase.EMBEDDED_MGMT,
                includeDir=includeDir,
                funcName=funcName,
                guardName=guardName
            )
            self.addProcess(snip)
        else:
            includeDir = os.path.dirname(os.path.realpath(__file__))
            funcName = "runSpiking"
            guardName = "doSpiking"
            snip = Process(
                name="inputEncoder",
                phase=Phase.EMBEDDED_SPIKING,
                includeDir=includeDir,
                funcName=funcName,
                guardName=guardName,
            )
            self.addProcess(snip)

    def setBiasExp(self, biasExpValue: int):
        """Set the biasExp if other composables need a non-zero biasExp"""
        self.biasExp = biasExpValue

    def _createOutputPort(self):
        """Create output port from which input will be injected"""
        outputPort = StateOutputPort(name=self._portName, shape=self.shape)
        self.addPort(outputPort)

    def _createSnips(self):
        """Create processes to decode and inject input data after mapping phase is done"""
        self._logger.debug("Creating the snips for Input Encoder")
        snipIdx = 0
        self._calculateNumPacketsPerSnip()

        realProcesses = Processes()

        for process in self.processes:
            for chip in self._chips:
                for lmt in range(self.numSnipsPerChip):
                    templateFile = os.path.basename(self.decoderSnip)
                    headerTemplateFile = templateFile.replace(".c", ".h")
                    cPath = os.path.dirname(os.path.realpath(
                        __file__)) + "/input_generator_{}_{}.c".format(chip, lmt)
                    headerFile = os.path.dirname(os.path.realpath(
                        __file__)) + "/input_generator_{}_{}.h".format(chip, lmt)

                    context = {
                        "chip": chip,
                        "lmt": lmt,
                        "numPacket": self._packetPerSnip[snipIdx],
                        "numAddresses": len(self._metaData[snipIdx]) // 4,
                        "start": self.startRunning,
                        "interval": self.interval,
                        "biasExp": self.biasExp
                    }

                    env = Environment(
                        loader=FileSystemLoader(
                            os.path.dirname(
                                self.decoderSnip)),
                        trim_blocks=True)

                    template = env.get_template(templateFile)
                    snipFile = template.render(context)
                    with open(cPath, 'w') as snip:
                        snip.write(snipFile)

                    template = env.get_template(headerTemplateFile)
                    header = template.render()
                    with open(headerFile, 'w') as hFile:
                        hFile.write(header)

                    snip = Process(
                        name="{}_{}_{}".format(process.name, chip, lmt),
                        phase=process.phase,
                        cFilePath=cPath,
                        includeDir=process.params["includeDir"],
                        funcName=process.params["funcName"],
                        guardName=process.params["guardName"],
                        chipId=chip,
                        lmtId=lmt
                    )
                    realProcesses.add(snip)
                    snip.registerWithComposable(self)

                    snipIdx += 1

        self.processes = realProcesses

    def _calculateNumPacketsPerSnip(self):
        """Calculates the number of packets of data to be send per snip"""
        self._logger.debug(
            "Calculating the NumPacketsPerSnip for Input Encoder")
        self._logger.debug(
            "Injection Points Per Snip is : {}".format(
                self._injectionPointsPerSnip))
        for idx, _ in enumerate(self._injectionPointsPerSnip):
            if (self._injectionPointsPerSnip[idx] * 4) % self.packetSize == 0:
                numPackets = (
                    self._injectionPointsPerSnip[idx] * 4) // self.packetSize
            else:
                numPackets = (
                    self._injectionPointsPerSnip[idx] * 4) // self.packetSize + 1
            self._packetPerSnip.append(np.asscalar(numPackets))
        self._logger.debug(
            "Packets per Snip is : {}".format(
                self._packetPerSnip))

    def _writeEncodedData(self, encodedData):
        """
        Write the encoded data on the channel to be read by snip

        :param encodedData: Encoded Data to be sent to snip
        """
        self._logger.debug("Writing the Input Encoder")

        dataPerChip = {}
        dataInjected = 0

        # Creating data buffer per chip
        for chipId, injectionPoints in self._dataOrder:
            if chipId not in dataPerChip:
                dataPerChip[chipId] = []
            dataPerChip[chipId].extend(encodedData[dataInjected:dataInjected + injectionPoints])
            dataInjected += injectionPoints

        dataSend = 0
        chipId = 0
        for idx, channel in enumerate(self._dataChannels):
            # Retrive chipId for snip from the stored dictionary
            if chipId != self._chipIdForSnip[idx]:
                chipId = self._chipIdForSnip[idx]
                dataSend = 0
            dataPerSnip = self._injectionPointsPerSnip[idx]
            channel.write(
                self._packetPerSnip[idx], dataPerChip[chipId][dataSend:dataSend + dataPerSnip])
            dataSend += dataPerSnip
            self._logger.debug(
                "Sending {} data to snip : {}".format(
                    dataPerSnip, idx))

    def _validateState(self, board: Graph):
        """
        Validates that incase of injcetion of bias current, all the other fields of
        CxCfg is 0 other than Mantisa.

        :param board: Configured board
        """
        for chip in board.n2Chips:
            for core in chip.n2Cores:
                for i in range(core.cxCfg.numNodes):
                    self._logger.debug(
                        "Validating : CxCfg for chip : {} core : {} and id : {} ".format(
                            chip, core, i))
                    if core.cxCfg[i].biasExp != 0 or \
                            core.cxCfg[i].cxProfileCfg != 0 or \
                            core.cxCfg[i].vthProfileCfg != 0:
                        raise ValueError(
                            "Chip : {} Core : {} CxCfg : {}"
                            " has non-zero non-modifiable attribute".format(
                                chip, core, i))

    def _defaultEncoder(self, data: np.ndarray):
        """
        Default Encoder function just flattens the nd array.
        'F' means to index the elements in column-major, Fortran-style order

        :param data: Data to be encoded
        """
        return np.ravel(data, 'F')

    def start(self, board: Graph, *args, **kwargs):
        """
        Sends the data size and packet size to be used by snip

        :param board: NxBoard
        """
        self._logger.debug("Starting the Input Encoder")
        self._logger.debug(
            "Packet per snip is : {} {}".format(self._packetPerSnip, len(
                self._dataSizeChannels)))
        self._logger.debug(
            "Length of metadata channels is : {}".format(len(
                self._metaDataChannels)))
        self._logger.debug(
            "Num range address per snip is : {}".format(
                self._metaData))
        for idx, channel in enumerate(self._dataSizeChannels):
            channel.write(1, [self._packetPerSnip[idx]])
        for channel in self._infoChannels:
            channel.write(1, [self.packetSize])
        for idx, channel in enumerate(self._metaDataChannels):
            channel.write(len(self._metaData[idx]), self._metaData[idx])

        self._compiled = True

    def encode(self, data: np.ndarray):
        """
        Encodes the data using default encoder or by user defined encoder
        and writes it to the channel

        :param data: Data to be encoded
        """
        self._logger.debug("Encoding the Input Encoder data")
        if not self._compiled:
            raise ValueError("InputGenerator is not yet compiled")
        if data.shape != self.shape:
            raise ValueError(
                "Shape of the Data to be Encoded {} doesn't match with the Input Encoder {}".format(
                    data.shape, self.shape))
        encodedData = self.encoderFunction(data)
        self._writeEncodedData(encodedData)

    def batchEncode(self, data: np.ndarray):
        """
        Encodes the batch data using default encoder or by user defined encoder
        and writes it to the channel

        :param data: ndarray, first dimension signifying the batch size
        """
        numBatches = data.shape[0]
        if numBatches > self.queueSize:
            raise ValueError(
                "Batch size cannot be greater than channel buffer")

        for batch_id, input in enumerate(data):
            self.encode(input)

    def _addInjectionPointsForChip(self, chipId, injectionPoints):
        """
        Helper function to add injectionPoints to chip
        :param chipId: Id of the chip
        :param injectionPoints: Num of Injection Points
        """
        if chipId not in self._injectionPointsPerChip:
            self._injectionPointsPerChip[chipId] = injectionPoints
            self._chips.append(chipId)
        else:
            self._injectionPointsPerChip[chipId] += injectionPoints

    def _calculateInjectionPointsForChip(self, addressMap: np.ndarray):
        """For the given chipId calculates the number of injection points"""
        if addressMap.size == 0:
            raise ValueError("Range Address Map for connected port is empty")
        chipId = addressMap[0][0]
        injectionPoints = 0
        for addr in addressMap:
            if chipId == addr[0]:
                injectionPoints += addr[3]
            else:
                self._addInjectionPointsForChip(chipId, injectionPoints)
                chipId = addr[0]
                injectionPoints = addr[3]
            # Storing the order in which data is mapped in form of chipId, numData
            self._dataOrder.append((addr[0], addr[3]))
        else:
            self._addInjectionPointsForChip(chipId, injectionPoints)

        self._logger.debug("Chips is : {}".format(self._chips))
        self._logger.debug(
            "Injection Points per Chip is : {}".format(
                self._injectionPointsPerChip))

    def partition(self, board: Graph):
        """
        Partition the injection ports into snips located on different chips
        and lmts.

        :param board: NxBoard
        """
        self._logger.debug("Partioning the Input Encoder")
        if len(self.ports) > 1:
            raise ValueError(
                "An Input Encoder should have single port of type input")
        numSnipsPerChip = self.numSnipsPerChip
        addressMap = self.ports[0].connectedPorts[0]().resourceMap
        if not isinstance(addressMap, RangeAddressWithVariableStrideStorage):
            addressMap.convertToStorageType(AddressesStorageType.RANGE_WITH_VARIABLE_STRIDE)
        self._rangeAddress = addressMap.map.addresses.rows
        self._calculateInjectionPointsForChip(addressMap.map.addresses.rows)

        for chipId in self._chips:
            if self._injectionPointsPerChip[chipId] % numSnipsPerChip == 0:
                self._injectionPointsPerSnip.extend(
                    [self._injectionPointsPerChip[chipId] // numSnipsPerChip] * numSnipsPerChip)
            elif self._injectionPointsPerChip[chipId] / numSnipsPerChip:
                self._injectionPointsPerSnip.extend(
                    [self._injectionPointsPerChip[chipId] // numSnipsPerChip] * (numSnipsPerChip - 1))
                self._injectionPointsPerSnip.append(
                    self._injectionPointsPerChip[chipId] //
                    numSnipsPerChip +
                    self._injectionPointsPerChip[chipId] %
                    numSnipsPerChip)
            else:
                raise ValueError(
                    "Total Injection Points is : {}".format(
                        self._injectionPointsPerChip[chipId]))
        self._logger.debug(
            "InjectionPointsPerChip is : {}".format(
                self._injectionPointsPerSnip))
        return self

    def map(self, board: Graph):
        """Maps the input Cx locations to range based Cx addresses"""
        self._logger.debug("Mapping the Input Encoder")
        # Placeholder function to do the range based cx address generation
        mapIdx = 0

        # Grouping range based address according to chip
        rangeAddress = self._rangeAddress.copy()
        rangeAddress = rangeAddress[rangeAddress[:, 0].argsort(kind='mergesort')]

        for idx, injectionPoints in enumerate(self._injectionPointsPerSnip):
            self._chipIdForSnip[idx] = rangeAddress[mapIdx][0]
            self._metaData.append([])
            toBeInjected = injectionPoints
            numRangeAddr = 0
            while toBeInjected:
                if toBeInjected >= rangeAddress[mapIdx][3]:
                    numRangeAddr += 1
                    toBeInjected -= rangeAddress[mapIdx][3]
                    self._metaData[idx].extend(rangeAddress[mapIdx].copy()[1:])
                    mapIdx += 1
                else:
                    numRangeAddr += 1
                    stride = rangeAddress[mapIdx][4]
                    self._metaData[idx].extend(rangeAddress[mapIdx].copy()[1:])
                    self._metaData[idx][-2] = toBeInjected
                    rangeAddress[mapIdx][2] += toBeInjected * stride
                    rangeAddress[mapIdx][3] = rangeAddress[mapIdx][3] - toBeInjected
                    toBeInjected = 0

        # Update processes to create new processes given the mapping is now
        # done
        self._createSnips()

        return self

    @staticmethod
    def load(path: str, board: Graph = None) -> 'AbstractComposable':
        """Loads the Input Generator"""
        with open(os.path.join(path, "ig"), "rb") as ig_file:
            result = pickle.load(ig_file)

        result.deserialize(path)

        with open(os.path.join(path, "ig_metadata"), "r") as m_file:
            metadata = json.load(m_file)

        result.logger = get_logger("NET.INE")

        result.dataChannels = [board.channels[idx]
                               for idx in metadata["dataChannelsIndices"]]
        result.infoChannels = [board.channels[idx]
                               for idx in metadata["infoChannelsIndices"]]
        result.dataSizeChannels = [board.channels[idx]
                                   for idx in metadata["dataSizeChannelsIndices"]]
        result.metaDataChannels = [board.channels[idx]
                                   for idx in metadata["metaDataChannelsIndices"]]

        return result

    def save(self, path: str):
        """
        Save the Input Generator within the given path

        :param path: Path to save the composable
        :returns: None
        """
        super().serialize(path)
        processes = self.processes
        ports = self.ports
        model = self.model
        logger = self._logger
        dataChannels = self._dataChannels
        infoChannels = self._infoChannels
        dataSizeChannels = self._dataSizeChannels
        metaDataChannels = self._metaDataChannels

        try:
            self.processes = None
            self.ports = None
            self.model = None
            self._logger = None

            dataChannelsIndices = [channel.index()
                                   for channel in self._dataChannels]
            infoChannelsIndices = [channel.index()
                                   for channel in self._infoChannels]
            dataSizeChannelsIndices = [channel.index()
                                       for channel in self._dataSizeChannels]
            metaDataChannelsIndices = [channel.index()
                                       for channel in self._metaDataChannels]

            metadata = {
                "dataChannelsIndices": dataChannelsIndices,
                "infoChannelsIndices": infoChannelsIndices,
                "dataSizeChannelsIndices": dataSizeChannelsIndices,
                "metaDataChannelsIndices": metaDataChannelsIndices
            }

            self._dataChannels = None
            self._infoChannels = None
            self._dataSizeChannels = None
            self._metaDataChannels = None

            with open(os.path.join(path, "ig_metadata"), "w") as m_file:
                json.dump(metadata, m_file)

            with open(os.path.join(path, "ig"), "wb") as ig_file:
                pickle.dump(self, ig_file)

        finally:
            self.processes = processes
            self.ports = ports
            self.model = model
            self._logger = logger
            self._dataChannels = dataChannels
            self._infoChannels = infoChannels
            self._dataSizeChannels = dataSizeChannels
            self._metaDataChannels = metaDataChannels

    def updatePorts(self, board):
        """Not Implemented"""
        return self

    def completeConnectivity(
            self,
            board: Graph,
            processAggregator: AbstractProcessAggregator):
        """
        Creates all channels and attaches them to embedded snip

        :param board: Configured board
        :param processAggregator: Aggregator process which holds the mapping of process to embedded snip
        :returns: self
        """
        snipIdx = 0

        for idx, process in enumerate(self.processes):
            key = process.getProcessKey()
            snip = processAggregator.getEmbeddedSnipForProcessKey(key)
            dataChannel = board.createChannel(
                'dataChannel_{}_{}'.format(
                    process.chipId,
                    process.lmtId),
                messageSize=self.packetSize,
                numElements=self.queueSize * self._packetPerSnip[idx]
            )
            self._dataChannels.append(dataChannel)

            # Incrementing the snip idx
            snipIdx += 1

            infoChannel = board.createChannel(
                'infoChannel_{}_{}'.format(
                    process.chipId,
                    process.lmtId),
                messageSize=4,
                numElements=1)
            infoChannel.connect(None, snip)
            self._infoChannels.append(infoChannel)

            dataSizeChannel = board.createChannel(
                'dataSizeChannel_{}_{}'.format(
                    process.chipId,
                    process.lmtId),
                messageSize=4,
                numElements=1)
            dataSizeChannel.connect(None, snip)
            self._logger.debug("Created Channel : {}".format(dataSizeChannel))
            self._dataSizeChannels.append(dataSizeChannel)

            metaDataChannel = board.createChannel('metaDataChannel_{}_{}'.format(
                process.chipId, process.lmtId), messageSize=4, numElements=16384)
            metaDataChannel.connect(None, snip)
            self._logger.debug("Created Channel : {}".format(metaDataChannel))
            self._metaDataChannels.append(metaDataChannel)

            # Connecting the channel
            dataChannel.connect(None, snip)
            infoChannel.connect(None, snip)
            dataSizeChannel.connect(None, snip)
            metaDataChannel.connect(None, snip)

        return self
