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
        self.axonMap = None
        self.decoderSnip = os.path.join(
            os.path.dirname(__file__),
            'templates',
            'inject_spike.c.template')
        self._portName = "output"
        self._addSnipPlaceholder()
        self._createOutputPort()
        # Setup the snip and channels

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
        self.axonMap = addressMap.map.addresses.rows.astype('uint64')

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
        sublen, rest = divmod(length, n_ways)
        return [input_list[i * sublen + min(i, rest):
                           (i + 1) * sublen + min(i + 1, rest)]
                for i in range(n_ways)]

    def encode(self, inputs):
        """
        Encode input into spikes
        Input is list of (id, t)
        """

        inputs_encoded = self.prepare_encoding(inputs)

        self.send_inputs(inputs_encoded)

    def prepare_encoding(self, inputs):

        inputs = np.array(inputs)

        input_addresses = self.axonMap[inputs[:, 0]]
        inputs_encoded = OrderedDict()
        # Group the spikes according to target chip.
        for chip in self.chips:
            chip_mask = input_addresses[:, 0] == chip
            core_axon_ids = input_addresses[chip_mask, 1:]
            timesteps = inputs[chip_mask, 1]
            inputs_per_chip = np.column_stack([core_axon_ids, timesteps])

            # Sort the spikes per chip according to time, core, axons.
            sort_idxs = np.lexsort((core_axon_ids[:, 0], timesteps))
            # Also group them by lmt which will be used to inject the spikes.
            inputs_per_cpu = self.split(inputs_per_chip[sort_idxs],
                                        self.numLmts)

            inputs_per_chip_encoded = OrderedDict()
            # Iterating over list of input spikes per snip.
            for lmt_id, input_per_cpu in enumerate(inputs_per_cpu):
                input_per_cpu_encoded = []
                for t in np.unique(input_per_cpu[:, 2]):
                    timestep_mask = input_per_cpu[:, 2] == t
                    core_axon_ids = input_per_cpu[timestep_mask, :2]

                    input_per_cpu_encoded.append(t)
                    core_axon_map = []
                    for core in np.unique(core_axon_ids[:, 0]):
                        core_mask = core_axon_ids[:, 0] == core
                        core_axon_map.append(
                            (core, list(core_axon_ids[core_mask, 1])))
                    input_per_cpu_encoded.append(len(core_axon_map))
                    for core, axon_ids in core_axon_map:
                        input_per_cpu_encoded.append(core)
                        input_per_cpu_encoded.append(len(axon_ids))
                        input_per_cpu_encoded.extend(axon_ids)
                inputs_per_chip_encoded[lmt_id] = input_per_cpu_encoded
            inputs_encoded[chip] = inputs_per_chip_encoded

        return inputs_encoded

    def send_inputs(self, inputs):
        channel_idx = 0
        for inputs_per_chip in inputs.values():
            for inputs_per_cpu in inputs_per_chip.values():
                num_packets = math.ceil(len(inputs_per_cpu) / self.packetSize)
                self._dataChannels[channel_idx].write(num_packets,
                                                      inputs_per_cpu)
                channel_idx += 1
