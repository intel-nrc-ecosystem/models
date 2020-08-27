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

"""A wrapper around NxModel to make it a composable"""

import os
from typing import List

import numpy as np
from jinja2 import Environment, FileSystemLoader
import atexit

from nxsdk import get_logger
from nxsdk.composable.abstract_composable import AbstractComposable
from nxsdk.composable.collections import Processes
from nxsdk.composable.interfaces.composable_enums import ResourceMapType
from nxsdk.composable.interfaces.process import Process
from nxsdk.composable.interfaces.process_aggregator_interface import AbstractProcessAggregator
from nxsdk.composable.port_impl import StateInputPort
from nxsdk.composable.resource_map import ResourceMapFactory
from nxsdk.graph.graph import Graph
from nxsdk.graph.monitor.probes import SpikeProbeCondition
from nxsdk.graph.processes.phase_enums import Phase
from nxsdk_modules_ncl.dnn.src.dnn_layers import ProbableStates, InputModes


class ComposableDNN(AbstractComposable):
    """A DNN that is composable. See nxsdk_modules_ncl.dnn.src.dnn_layers.NxModel which is the underlying DNN Model"""
    def __init__(self, model: 'NxModel', num_steps_per_img: int, enable_reset: bool = True):
        """
        Wraps a DNNModel and makes it composable

        :param model (nxsdk_modules_ncl.dnn.src.dnn_layers.NxModel): The underlying DNN Model created from NxTF Layers
        :param num_steps_per_img: Number of steps to run for each image
        :param enable_reset: Whether to reset states after ``num_steps_per_img``.
        """
        super().__init__()

        self._logger = get_logger("NET.DNN")

        self._build(model=model, num_steps_per_img=num_steps_per_img, enableReset=enable_reset)

    def _build(self, *args, **kwargs):
        """Builds the ports, probes and snips for the composable. This method is called from base class constructor"""
        # Stores a reference to the underlying model
        self._dnn = kwargs["model"]
        self._addPorts()
        self._addProcesses()
        self._num_steps_per_img = kwargs["num_steps_per_img"]
        self._enableReset = kwargs['enableReset']

    def _addPorts(self):
        """Adds ports to the composable"""
        # Create and add input port. This will be delegated to the input layer
        self.addPort(StateInputPort(name="input"))

    def _addProcesses(self):
        """Adds processes/snips associated with DNN Composable"""
        snipDir = os.path.join(os.path.dirname(__file__), '..', 'snips', 'reset_model_states')

        # Init snip to populate number of cores and reset interval
        init = Process(
            name='init',
            cFilePath=snipDir + "/snip_init.c",
            includeDir=snipDir,
            funcName='init_1',
            phase=Phase.EMBEDDED_INIT,
            lmtId=0)
        self.addProcess(init)

        # Todo : Profile and measure to see if spreading readout and/or reset across lmts helps.
        # Reset SNIP
        reset_snip = Process(
            name='reset',
            cFilePath=snipDir + "/snip_reset.c",
            includeDir=snipDir,
            guardName='do_reset',
            funcName='reset',
            phase=Phase.EMBEDDED_MGMT,
            lmtId=0)
        self.addProcess(reset_snip)

        readout_spike_activity_snip_dir = os.path.join(os.path.dirname(__file__),
                                                       '..', 'snips', 'readout_spike_activity')

        # This is an example of lazily creating a process. The C file does not exist yet and will
        # only be generated post map phase when output layer has been mapped to neurocores.

        # Class Readout SNIP
        readout_snip = Process(
            name='readout',
            cFilePath=readout_spike_activity_snip_dir + "/snip_class_readout.c",
            includeDir=readout_spike_activity_snip_dir,
            guardName='do_readout',
            funcName='readout',
            phase=Phase.EMBEDDED_MGMT,
            lmtId=0)
        self.addProcess(readout_snip)

    def partition(self, board: Graph) -> AbstractComposable:
        """Partition the dnn model. We ignore this step and delegate it to map which invokes compileModel"""
        return self

    def map(self, board: Graph) -> AbstractComposable:
        """Invoke partition and mapping of the dnn model"""
        self._dnn.compileModel(board)
        self._createSnips(board)
        self._createReadoutSnip()
        return self

    def updatePorts(self, board: Graph) -> AbstractComposable:
        """Updates resourceMap to input and output ports"""
        inputLayer = self._dnn.layers[0]

        if inputLayer.inputMode == InputModes.AEDAT:
            self.ports.input.resourceMap = ResourceMapFactory.createExplicit(
                ResourceMapType.INPUT_AXON, inputLayer.inputAxonResourceMap)
        else:
            # Return input compartments for multi-compartment neurons
            neuronSize = 2 if inputLayer.resetMode == 'soft' else 1
            cxResourceMap = inputLayer.cxResourceMap[::neuronSize]
            self.ports.input.resourceMap = ResourceMapFactory.createExplicit(
                ResourceMapType.COMPARTMENT, cxResourceMap)
        # self.ports.output.resourceMap = CompartmentResourceMap(self._dnn.layers[-1].cxResourceMap)
        return self

    def completeConnectivity(self, board: Graph, processAggregator: AbstractProcessAggregator) -> AbstractComposable:
        """Create channel to communicate data to init snip"""
        # Should pipe to resourceMap indices for output layer
        self._createInitializationChannel(board, processAggregator)
        self._createReadoutChannel(board, processAggregator)
        return self

    def _createSnips(self, board: Graph):
        """Create clones of reset and init snips based on number of chips used by input layer."""
        processes = Processes()
        for chip_id in range(board.numChips):
            # init snip
            initProcess = self.processes.init
            initProcessWithChipId = initProcess.clone(name=initProcess.name + str(chip_id),
                                                      params={'chipId': chip_id})
            processes.add(initProcessWithChipId)

            # reset snip
            resetProcess = self.processes.reset
            resetProcessWithChipId = resetProcess.clone(name=resetProcess.name + str(chip_id),
                                                      params={'chipId': chip_id})
            processes.add(resetProcessWithChipId)

        # Todo : Enable readout for output layers distributed across multiple chips.
        # readout
        chip_id = self._dnn.layers[-1].cxResourceMap[0, 0]
        assert len(np.unique(self._dnn.layers[-1].cxResourceMap[:, 0])) == 1

        readoutProcess = self.processes.readout
        readoutProcessWithChipId = readoutProcess.clone(name=readoutProcess.name,
                                                        params={'chipId': chip_id})
        processes.add(readoutProcessWithChipId)
        self.processes = processes

    def _createReadoutSnip(self):
        """Create readout snip for compartment of the output layer.

           The  voltage is readout when using an output layer with a softmax
           activation, otherwise, spikes are readout by creating spike counters
           at the lakemonts.
        """
        probeDt = 1
        probeStart = 100000000

        # Get the output layer from the spiking model
        output_layer = self._dnn.layers[-1]

        NUM_CLASSES = int(np.prod(output_layer.output_shape[1:]))

        # Return output compartments for multi-compartment neurons.
        neuronSize = 2 if output_layer.resetMode == 'soft' else 1
        offset = 1 if output_layer.resetMode == 'soft' else 0

        # Determine whether to read spikes or voltages based on activation.
        readSpikes = True
        if hasattr(output_layer, 'activation') and \
                output_layer.activation.__name__ == 'softmax':
            offset = 0
            readSpikes = False

        lmt_spike_counters = []

        if readSpikes:
            for i in range(NUM_CLASSES):
                spike_probe = output_layer[i * neuronSize + offset].probe(
                    state=ProbableStates.SPIKE,
                    probeCondition=SpikeProbeCondition(dt=probeDt, tStart=probeStart))
                lmt_spike_counters.append(spike_probe.counterId)
            cores = cxIds = np.zeros_like(lmt_spike_counters).tolist()
        else:
            rm = output_layer.cxResourceMap
            cores = rm[offset::neuronSize, 1].tolist()
            cxIds = rm[offset::neuronSize, 2].tolist()
            lmt_spike_counters = np.zeros_like(cxIds).tolist()

        # Now that lmt_spike_counters are known, generate the snip_class_readout.c
        self._generateReadOutSnipCFileFromJinjaTemplate(readSpikes=readSpikes,
                                                        num_classes=NUM_CLASSES,
                                                        lmt_output_spike_counter_ids=lmt_spike_counters,
                                                        cores=cores,
                                                        cxIds=cxIds)

    @staticmethod
    def _cleanup():
        readout_spike_activity_snip_dir = os.path.join(os.path.dirname(__file__),
                                                       '..', 'snips', 'readout_spike_activity')
        cFilePath = os.path.join(readout_spike_activity_snip_dir, "snip_class_readout.c")
        if os.path.exists(cFilePath):
            os.remove(cFilePath)

    def _generateReadOutSnipCFileFromJinjaTemplate(self,
                                                   readSpikes: bool,
                                                   num_classes: int,
                                                   lmt_output_spike_counter_ids: List[int],
                                                   cores: List[int],
                                                   cxIds: List[int]):
        atexit.register(ComposableDNN._cleanup)

        readout_spike_activity_snip_dir = os.path.join(os.path.dirname(__file__),
                                                       '..', 'snips', 'readout_spike_activity')

        context = {
            "READ_SPIKES": int(readSpikes),
            "NUM_CLASSES": num_classes,
            "NUM_STEPS_PER_IMG": self._num_steps_per_img,
            "LMT_OUTPUT_SPIKE_COUNTER_IDS": "{" + str(lmt_output_spike_counter_ids)[1:-1] + "}",
            "CORE_IDS": "{" + str(cores)[1:-1] + "}",
            "CX_IDS": "{" + str(cxIds)[1:-1] + "}"
        }

        env = Environment(loader=FileSystemLoader(os.path.join(readout_spike_activity_snip_dir, "templates")),
                          trim_blocks=True)

        c_template = env.get_template("snip_class_readout.c.template")
        c_contents = c_template.render(context)
        with open(os.path.join(readout_spike_activity_snip_dir, "snip_class_readout.c"), 'w') as cFile:
            cFile.write(c_contents)

    def _createInitializationChannel(self, board: Graph, processAggregator: AbstractProcessAggregator):
        """Creates a channel and connects it to init snip"""

        for chip_id in range(board.numChips):
            init_process = self.processes['init' + str(chip_id)]
            processKey = init_process.getProcessKey()
            snip_init_1 = processAggregator.getEmbeddedSnipForProcessKey(processKey)
            name = 'channel_init_ch{}_lmt0'.format(chip_id)
            setattr(self,
                    name,
                    board.createChannel(bytes(name, 'utf-8'), "int", 3))

            getattr(self, name).connect(None, snip_init_1)

    def _createReadoutChannel(self, board: Graph, processAggregator: AbstractProcessAggregator):
        """Create a readout channel to read the classification values from spike counters"""
        readout_process = self.processes.readout
        processKey = readout_process.getProcessKey()
        snip_readout = processAggregator.getEmbeddedSnipForProcessKey(processKey)
        self.readout_channel = board.createChannel(bytes('readout', 'utf-8'), "int", numElements=100000)
        self.readout_channel.connect(snip_readout, None)

    @staticmethod
    def load(path: str, board: Graph = None) -> 'AbstractComposable':
        """Not Implemented"""
        raise NotImplementedError

    def save(self, path: str):
        """Not Implemented"""
        raise NotImplementedError

    def start(self, board: Graph, *args, **kwargs):
        """Writes initial configuration settings (num_cores_per_chip, num_steps_per_img, enableReset) to init channel"""
        num_cores_per_chip = [board.n2Chips[i].numCores for i in range(board.numChips)]
        for chip_id in range(board.numChips):
            name = 'channel_init_ch{}_lmt0'.format(chip_id)
            getattr(self, name).write(3, [num_cores_per_chip[chip_id], self._num_steps_per_img, self._enableReset])
