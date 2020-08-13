# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2020 Intel Corporation.
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

"""
SpikeInputGenerator is a layer which can be used to encode data and inject spikes
in neurocore.
"""

import json
import math
import pickle
import random
from collections import OrderedDict

import numpy as np
from typing import Callable
from nxsdk.composable.collections import Processes
from nxsdk.graph import channel
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


class SpikeInputGenerator(AbstractComposable):
    def __init__(self,
            name: str,
            packetSize: int = 1024,
            numSnipsPerChip: int = 1,
            queueSize: int = 64):
        """
        SpikeInputGenerator is layer which can encode and inject spikes into neurocore.
        Spike Input Generator performs following tasks:
        1. Input encoding to compress data in most suitable form for channel communication
        2. Writing input to channel
        3. Input decoding on host or embedded CPU
        4. Input injection to neuro core on host or embedded CPU via SNIP code

        :param name: Name of the Input Encoder
        :param packetSize: Size of one packet
        :param numSnipsPerChip: Number of snips per chip on which input layer is present
        :param queueSize: Number of inputs that can be queued without blocking):
        """

        super().__init__()
        self.packetSize = packetSize
        self.numLmts = numSnipsPerChip
        self.queueSize = queueSize
        self._logger = get_logger("NET.INE")
        self._dataChannels = []
        self.axonMap = {}
        self.decoderSnip = os.path.join(
            os.path.dirname(__file__),
            'templates',
            'inject_spike.c.template')
        self._portName = "output"
        self._addSnipPlaceholder()
        self._createOutputPort()
        # Setup the snip and channels

    def get_id_from_resource_map(self, id):
        return self.axonMap[id]

    def _addSnipPlaceholder(self):
        """
        As Snips can be only be attached post mapping phase, these processes are placeholders.
        Adds the inputEncoder process in spiking phase.
        """
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

    def _createOutputPort(self):
        """Create output port from which input will be injected"""
        outputPort = StateOutputPort(name=self._portName)
        self.addPort(outputPort)

    def _createSnips(self):
        """Create processes to decode and inject input data after mapping phase is done"""
        self._logger.debug("Creating the snips for Input Encoder")
        snipIdx = 0
        realProcesses = Processes()

        for process in self.processes:
            for chip in self.chips:
                for lmt in range(self.numLmts):
                    templateFile = os.path.basename(self.decoderSnip)
                    headerTemplateFile = templateFile.replace(".c", ".h")
                    cPath = os.path.dirname(os.path.realpath(
                        __file__)) + "/input_generator_{}_{}.c".format(chip, lmt)
                    headerFile = os.path.dirname(os.path.realpath(
                        __file__)) + "/input_generator_{}_{}.h".format(chip, lmt)

                    context = {
                        "chip": chip,
                        "lmt": lmt,
                        "packet_size": self.packetSize
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

    def completeConnectivity(
            self,
            board: Graph,
            processAggregator: AbstractProcessAggregator) -> 'AbstractComposable':
        snipIdx = 0

        for idx, process in enumerate(self.processes):
            key = process.getProcessKey()
            snip = processAggregator.getEmbeddedSnipForProcessKey(key)
            dataChannel = board.createChannel(
                'dataChannel_{}_{}'.format(
                    process.chipId,
                    process.lmtId),
                messageSize=self.packetSize * 4,
                numElements=self.queueSize * self.packetSize
            )
            self._dataChannels.append(dataChannel)

            # Incrementing the snip idx
            snipIdx += 1
            # Connecting the channel
            dataChannel.connect(None, snip)

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

        try:
            self.processes = None
            self.ports = None
            self.model = None
            self._logger = None

            dataChannelsIndices = [channel.index()
                                   for channel in self._dataChannels]

            metadata = {
                "dataChannelsIndices": dataChannelsIndices,
            }

            self._dataChannels = None

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

    def _createIdMap(self, addressMap):
        self.chips = sorted(set(addressMap.map.addresses.rows[:, 0]))
        id = 0
        for chip, core, axonid in addressMap.map.addresses.rows:
            self.axonMap[id] = (chip, core, axonid)
            id += 1

    def map(self, board: Graph) -> 'AbstractComposable':
        addressMap = self.ports[0].connectedPorts[0]().resourceMap
        self._createIdMap(addressMap)
        self._createSnips()
        return self

    def partition(self, board: Graph) -> 'AbstractComposable':
        return self

    def updatePorts(self, board: Graph) -> 'AbstractComposable':
        """Not Implemented"""
        return self

    def split(self, input_list, n_ways):
        length = len(input_list)
        return [input_list[i * (length // n_ways) + min(i, length %
                                                        n_ways):(i + 1) * (length // n_ways) + min(i + 1, length %
                                                                                                   n_ways)] for i in range(n_ways)]

    def encode(self, inputs):
        """
        Encode input into spikes
        Input is list of (id, t)
        """
        input_by_chip = {}
        channelIdx = 0
        # Group the spikes according to target chip
        for id, t in inputs:
            chip, core, axonid = self.get_id_from_resource_map(id)
            input_by_chip.setdefault(chip, []).append((core, axonid, t))

        # Sort the spikes per chip according to time, core, axons
        # Also group them by lmt which will be used to inject the spike
        for chip in input_by_chip.keys():
            input_by_chip[chip] = self.split(
                sorted(
                    input_by_chip[chip],
                    key=lambda inp: (
                        inp[2],
                        inp[0],
                        inp[1])),
                self.numLmts)

            for input_per_snip in input_by_chip[chip]:
                input_per_snip.append([0, 0, -1])

            # Iterating over list of input spikes per snip
            for snip in input_by_chip[chip]:
                # Key is time and Value is (core, axonid)
                input_per_snip_per_chip = OrderedDict()
                for core, axon, t in snip:
                    input_per_snip_per_chip.setdefault(
                        t, []).append((core, axon))

                input_to_be_sent = []
                for time, core_spike_list in input_per_snip_per_chip.items():
                    input_to_be_sent.append(time)
                    input_per_time_core = OrderedDict()
                    for core, axon in core_spike_list:
                        input_per_time_core.setdefault(core, []).append(axon)
                    input_to_be_sent.append(len(input_per_time_core))
                    for core, axonlist in input_per_time_core.items():
                        input_to_be_sent.append(core)
                        input_to_be_sent.append(len(axonlist))
                        input_to_be_sent.extend(axonlist)

                self._dataChannels[channelIdx].write(
                    math.ceil(len(input_to_be_sent) / self.packetSize),
                    input_to_be_sent)
                channelIdx += 1
