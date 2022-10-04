# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
# 
# Copyright © 2021-2022 Intel Corporation.
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
import warnings
import inspect
import numpy as np
from nxsdk.graph.processes.phase_enums import Phase
from nxsdk.arch.n2a.n2board import N2Board

def getChipCoreAndCxId(layer):
    """Returns the physical chip core and compartment id of a compartment group layer.

    Args:
        layer : layer compartment group

    Returns:
        chip_ids, core_ids and cx_ids
    """
    core_ids = []
    cx_ids = []
    chip_ids = []
    for id in layer.nodeIds:
        _, chip_id, core_id, cx_id, _, _ = layer.net.resourceMap.compartment(id)
        chip_ids.append(chip_id)
        core_ids.append(core_id)
        cx_ids.append(cx_id)
    return np.array(chip_ids), np.array(core_ids), np.array(cx_ids)

class Header():
    """Manages header constants aggregation and file creation for snip processes.
    The constants are readable/writeable with constant key like a dictionary. It
    internally maintained in a dictionary object.
    """
    def __init__(self, filename='array_sizes.h'):
        """Initializer.

        Args:
            filename (str, optional): name of the header file to be created. 
                Defaults to 'array_sizes.h'.
        """
        self.name = filename
        self.define = {}

    def __getitem__(self, key):
        """Get header constant value/s.

        Args:
            key (string): header constant name.

        Returns:
            header constant value.
        """
        return self.define[key]

    def __setitem__(self, key, value):
        """Set header constant value/s. Issues warning if same constant is overwritten.
        Redefinition is generally not intended.

        Args:
            key (string): header constant name.
            value : header constant value/s.
        """
        if key in self.define:
            warnings.warn('Key {} is being overwritten to {}. It had a value of {}. Hope you know what you are doing.'.format(key, value, self.define[key]))
        self.define[key] = value

    @staticmethod
    def format(value):
        """Formats the header object to appropriate style of C type definition.

        Args:
            value (scalar, string, bool, dictionary and array): header constant to be formatted.

        Returns:
            formatted header constant
        """
        if isinstance(value, str):
            return '"{}"'.format(value)
        if isinstance(value, bool):
            return 'true' if value is True else 'false'
        elif isinstance(value, dict):
            assert False, 'Not implemented for dictionary type'
        elif hasattr(value, '__len__'): # should cover list and numpy array
            return '{{{}}}'.format(', '.join([str(v) for v in value]))
        else: # assume scalar value
            return value

    def __str__(self):
        """Creates a C/C++ string representation of the header.

        Returns:
            string: C/C++ string representation of the header.
        """
        header_string = ''
        for key, value in self.define.items():
            header_string += '#define {} {}\n'.format(key, self.format(value))
        return header_string

    def write(self):
        """Generate the header file as defined by filename member.
        """
        # # Sometimes file is not written properly. So delete and rewrite it
        # os.system('rm {}'.format(snip_dir + '/' + self.name))
        # if 'NUM_TIME_STEPS' not in self.define.keys():
        #     warnings.warn('NUM_TIME_STEPS missing in header. Execution may hang!')
        with open(snip_dir + '/' + self.name, 'w') as f:
            f.write('/* Temporary generated file for snip process definitions before compilation */\n')
            f.write(self.__str__())

        # os.system('ls {}'.format(snip_dir + '/' + self.name))

class IOManagement():
    """IO process handler for management phase snips. All the management processes are
    derived from this class. It maintains a static table of boards, snips, chips and lmts
    for which management snips are created. 
    """
    boards = [] # this is probably always going to be 1
    snips = []
    chips = []
    lmts  = []
    def __init__(self):
        self.board = None
    
    def snip(self, chip, lmt):
        """Snip creation method checks for pre-exisiting snips when requested and creates one if it is not available.

        Args:
            chip (int): chip ID.
            lmt (int): lakemnont ID.

        Returns:
            snip process.
        """
        for i in range(len(IOManagement.snips)): # look if snips is alerady there
            if IOManagement.boards[i]==self.board.id and IOManagement.chips[i]==chip and IOManagement.lmts[i]==lmt:
                # print('Found existing snip process. Using it.')
                return IOManagement.snips[i]
    
        # if snip is not alrady created, create one
        snip_process = self.board.createSnip(
                phase = Phase.EMBEDDED_MGMT,
                name = 'snip_{}_{}'.format(chip, lmt),
                includeDir = snip_dir,
                cFilePath = snip_dir + '/iomgmt.c',
                funcName = 'run_mgmt',
                guardName = 'do_run_mgmt',
                lmtId = lmt,
                chipId = chip,
            )
        IOManagement.boards.append(self.board.id)
        IOManagement.snips.append(snip_process)
        IOManagement.chips.append(chip)
        IOManagement.lmts.append(lmt)

        return snip_process

class IOSpiking():
    """IO process handler for spiking phase snips. All the management processes are
    derived from this class. It maintains a static table of boards, snips, chips and lmts
    for which management snips are created. 
    """
    boards = [] # this is probably always going to be 1
    snips = []
    chips = []
    lmts  = []
    def __init__(self):
        self.board = None
    
    def snip(self, chip, lmt):
        """Snip creation method checks for pre-exisiting snips when requested and creates one if it is not available.

        Args:
            chip (int): chip ID.
            lmt (int): lakemnont ID.

        Returns:
            snip process.
        """
        for i in range(len(IOSpiking.snips)): # look if snips is alerady there
            if IOSpiking.boards[i]==self.board.id and IOSpiking.chips[i]==chip and IOSpiking.lmts[i]==lmt:
                # print('Found existing snip process. Using it.')
                return IOSpiking.snips[i]
    
        # if snip is not alrady created, create one
        snip_process = self.board.createSnip(
                phase = Phase.EMBEDDED_SPIKING,
                name = 'spike_snip_{}_{}'.format(chip, lmt),
                includeDir = snip_dir,
                cFilePath = snip_dir + '/iospiking.c',
                funcName = 'run_spiking',
                guardName = 'do_run_spiking',
                lmtId = lmt,
                chipId = chip,
            )
        IOSpiking.boards.append(self.board.id)
        IOSpiking.snips.append(snip_process)
        IOSpiking.chips.append(chip)
        IOSpiking.lmts.append(lmt)

        return snip_process

class BiasInput(IOManagement):
    """IO process for bias input injection. For now, it assumes the input layer
    is entirely on chip 0.
    """
    def __init__(
            self, board, input_layer, comp_per_core, header, interval, 
            packet_size=2048, num_lmts = 3,
            filename=None,
        ):
        """Initializer.

        Args:
            board : Board object for IO process.
            input_layer : Input layer compartment group.
            comp_per_core : Input compartments per core.
            header : Header management object.
            interval (int): Interval for bias input injection.
            packet_size (int): size of communication packet.
            num_lmts (int, optional): Number of lakemonts to use for bias injection. Defaults to 3.
            filename (string, optional): If filename is provided, the layer's physical
                core ids and compartment ids are read from the file. Otherwise they are
                populated from compiled network. Defaults to None.
        """
        super(BiasInput, self).__init__()
        self.board = board
        self.layer = input_layer
        self.header = header
        self.num_lmts = num_lmts
        self.num_chips = 1 # assume input fits in one chip and is always placed on chip 0
        self.interval = interval
        self.num_packed_elements = packet_size
        self.num_neurons_core = int(np.ceil(comp_per_core))
        self.num_cores = int(np.ceil(input_layer.numNodes/comp_per_core))
        self.channels = self._create_channels()
        if filename is None:
            _, self.core_ids, self.cx_ids = getChipCoreAndCxId(self.layer)
        else:
            self.core_ids, self.cx_ids = self.load(filename)

        # check if core_ids and cx_ids are in proper order
        assert (self.core_ids - np.arange(len(self.core_ids))//self.num_neurons_core).sum() == 0, 'Input core IDs are not in proper order to be decoded from lmt.'
        assert (self.cx_ids - np.arange(len(self.cx_ids))%self.num_neurons_core).sum() == 0, 'Input compartment IDs are not in proper order to be decoded from lmt.'

        # print(self.core_ids)
        # print(self.cx_ids)
        # print(self.num_neurons_core)
        # print(self.num_cores)
        # print(self.num_packed_elements)

        self.header['USE_BIAS_INPUT'] = True
        self.header['NUM_BIAS_INPUT_SNIPS'] = self.num_lmts
        self.header['NUM_INPUTS'] = self.layer.numNodes
        self.header['INPUT_NUM_PACKED'] = self.num_packed_elements
        self.header['INPUT_NUM_PACKETS'] = self.num_packets
        self.header['NUM_INPUT_CORES'] = self.num_cores
        self.header['INPUT_NEURONS_PER_CORE'] = self.num_neurons_core
        self.header['BIAS_INPUT_INTERVAL'] = self.interval
         
    def _create_channels(self):
        channels = [[None] * self.num_lmts for _ in range(self.num_chips)]
        for chip in range(self.num_chips):
            for lmt in range(self.num_lmts):
                channels[chip][lmt] = self.board.createChannel(
                        name = 'bias_input_{}_{}'.format(chip, lmt).encode(),
                        messageSize = 4*16, # it is efficient to read 64 bytes of message at once
                        numElements = self.num_packed_elements // 4 // 16,
                        # slack = 32, # add extra slack to avoid hanging
                    )
                channels[chip][lmt].connect(None, self.snip(chip, lmt))
                
        return channels

    @property
    def num_packets(self):
        """Number of packets to send per lakemont used.
        """
        return int(np.ceil(self.layer.numNodes / self.num_packed_elements / self.num_lmts))

    def _pack_data(self, input):
        input = input.transpose([1, 0, 2]) # order in s2l is WHC
        
        packed_data = np.zeros(self.num_packed_elements * self.num_packets * self.num_lmts).astype(np.int32)
        packed_data[:self.layer.numNodes] = input.flatten()
        assert packed_data.min() >= -128 and packed_data.max() < 128, \
            'Input data expected to be 8 bit signed. Found min value of {} and max valueof {}'.format(
                packed_data.min() >= -128, packed_data.max() >= -128
            )

        lmt_packed_data = [None for _ in range(self.num_lmts)]
        for lmt in range(self.num_lmts):
            lmt_data = packed_data[lmt::self.num_lmts]

            lmt_packed_data[lmt] = np.left_shift(np.bitwise_and(lmt_data[3::4], 0xFF), 24) + \
                                   np.left_shift(np.bitwise_and(lmt_data[2::4], 0xFF), 16) + \
                                   np.left_shift(np.bitwise_and(lmt_data[1::4], 0xFF),  8) + \
                                                 np.bitwise_and(lmt_data[0::4], 0xFF)

            lmt_packed_data[lmt] = lmt_packed_data[lmt].reshape(self.num_packets, -1)
        
        return [lmt_packed_data]

    def send(self, image):
        """Send input data to chip

        Args:
            image : Input image data
        """
        packed_data = self._pack_data(image)
        for chip in range(self.num_chips):
            for lmt in range(self.num_lmts):
                for packet in range(self.num_packets):
                    self.channels[chip][lmt].write(self.channels[chip][lmt].numElements, packed_data[chip][lmt][packet])

    def save(self, filename):
        """Save the physical core and compartment ids to file for loading later.
        The file is saved as a npz data.

        Args:
            filename (string): Name of the saved file. 
        """
        np.savez(temp_dir + '/' + filename + '.npz', core_ids=self.core_ids, cx_ids=self.cx_ids)

    def load(self, filename):
        """Load the physical core and compartment ids from saved npz file.

        Args:
            filename (string): Name of the saved file. 

        Returns:
            core_ids and cx_ids
        """
        data = np.load(temp_dir + '/' + filename + '.npz')
        return data['core_ids'], data['cx_ids']


class VoltageOutput(IOManagement):
    """IO process for reading layer voltage periodically.
    """
    def __init__(self, board, output_layer, header, num_samples, offset=0, interval=1, lmt=2, filename=None):
        """Initializer

        Args:
            board : Board object for IO process.
            output_layer : Output layer compartment group.
            header : Header management object.
            num_samples (int): Number of samples to read.
            offset (int, optional): Number of time steps to skip before reading data. Defaults to 0.
            interval (int, optional): Inverval for voltage readout. Defaults to 1.
            lmt (int, optional): The lakemont id to use. Defaults to 2.
            filename (string, optional): If filename is provided, the layer's physical
                core ids and compartment ids are read from the file. Otherwise they are
                populated from compiled network. Defaults to None.
        """
        super(VoltageOutput, self).__init__()
        self.board = board
        self.layer = output_layer
        self.header = header
        self.offset = offset
        self.interval = interval # read data every interval ticks, but start reading at an offset
        self.num_samples = num_samples
        self.skip_samples = self.offset // self.interval
        if self.skip_samples > 0:
            print(f'Read offset ({self.offset}) is greater than the read interval ({self.interval}).')
            print(f'{self.skip_samples} readout will be skipped for valid data.')
            # object_name = f'{self=}'.split('=')[0]
            print('To reveive full data, explicitly set {object_name}.skip_samples = 0')
        
        self.header['USE_VOLTAGE_OUTPUT'] = True
        self.header['NUM_OUTPUTS'] = self.layer.numNodes
        
        if filename is None:
            self.chip_ids, self.core_ids, self.cx_ids = getChipCoreAndCxId(self.layer)
        else:
            self.chip_ids, self.core_ids, self.cx_ids = self.load(filename)

        # print(self.chip_ids)
        # print(self.core_ids)
        # print(self.cx_ids)

        self.header['OUTPUT_CORE_IDS'] = self.core_ids
        self.header['OUTPUT_CX_IDS'] = self.cx_ids
        self.header['VOLTAGE_OUTPUT_OFFSET'] = self.offset
        self.header['VOLTAGE_OUTPUT_INTERVAL'] = self.interval

        self.chip_id = np.min(self.chip_ids) # assume all neurons in layer are in same chip
        self.lmt_id = lmt
        self.channel = self._create_channel()

    def _create_channel(self):
        channel = self.board.createChannel(
                'voltage_output_{}_{}'.format(self.chip_id, self.lmt_id).encode(),
                messageSize = 4, # 4 bytes of message
                numElements = self.layer.numNodes * self.num_samples,
                slack = 32, # add extra slack to avoid hanging
            )
        channel.connect(self.snip(self.chip_id, self.lmt_id), None)

        return channel

    def receive(self):
        """Receive output voltage data.

        Returns:
            Output voltage data.
        """
        results = self.channel.read(self.num_samples * self.layer.numNodes)
        results = np.array(results).reshape((self.num_samples, -1))
        return results[self.skip_samples:] # skip invalid data

    def save(self, filename):
        """Save the physical core and compartment ids to file for loading later.
        The file is saved as a npz data.

        Args:
            filename (string): Name of the saved file. 
        """
        np.savez(temp_dir + '/' + filename + '.npz', chip_ids=self.chip_ids, core_ids=self.core_ids, cx_ids=self.cx_ids)

    def load(self, filename):
        """Load the physical core and compartment ids from saved npz file.

        Args:
            filename (string): Name of the saved file. 

        Returns:
            chip_ids, core_ids and cx_ids
        """
        data = np.load(temp_dir + '/' + filename + '.npz')
        return data['chip_ids'], data['core_ids'], data['cx_ids']

class LayerReset(IOManagement):
    """IO process to orchestrate layerwise reset. Layers 0, 1, 2, ... are
    reset at time 0, 1, 2, ... and periodically after that. Layer-0 is input layer.
    """
    def __init__(self, board, corenums, header, interval, lmt=2):
        """Initializer

        Args:
            board : Board object for IO process.
            corenums : array indicating start and end cores of layers.
            header : Header management object.
            interval (int): Inverval for voltage readout
            lmt (int, optional): The lakemont id to use. Defaults to 2.
        """
        super(LayerReset, self).__init__()
        self.board = board
        self.header = header
        self.interval = interval
        self.lmt_id = lmt

        core_st = np.ceil(corenums)[:-1].astype(int)
        core_en = np.ceil(corenums)[1: ].astype(int)
        cores = []
        for l in range(len(core_st)):
            cores.append(np.arange(core_st[l], core_en[l]))
        
        self.cores = np.hstack(cores)
        self.cores = self.cores % 128
        self.core_start = core_st % 128
        self.chip_start = core_st // 128
        self.chip_end = core_en // 128
        self.num_cores_in_layer = core_en - core_st

        # print('')
        # print('Cores  :', self.cores)
        # print('Chip St:', self.chip_start)
        # print('Chip En:', self.chip_end)
        # print('Core St:', self.core_start)
        # print('N cores:', self.num_cores_in_layer)

        self.header['USE_LAYER_RESET'] = True
        self.header['NUM_LAYERS'] = len(core_st)
        self.header['NUM_RESET_CORES'] = len(self.cores)
        self.header['LAYER_RESET_INTERVAL'] = self.interval
        self.header['LAYER_RESET_LMT'] = self.lmt_id
        self.header['LAYER_RESET_CORES'] = self.cores
        self.header['LAYER_CHIP_START'] = self.chip_start
        self.header['LAYER_CHIP_END'] = self.chip_end
        self.header['LAYER_CORE_START'] = self.core_start
        self.header['NUM_CORES_IN_LAYER'] = self.num_cores_in_layer
        
        self.snips = self._setup_snips()

    def _setup_snips(self):
        snips = []
        for chip in range(self.chip_end.max() + 1):
            snips.append(self.snip(chip, self.lmt_id))
        return snips


class ProfileTime(IOSpiking):
    """IO process for getting execution time log. Light weight than time probes.
    """
    def __init__(self, board, header, num_profile_steps):
        """Initializer

        Args:
            board : Board object for IO process.
            header : Header management object.
            num_profile_steps (int): Number of time steps to log the timing data.
        """
        self.num_profile_steps = num_profile_steps
        self.board = board
        self.header = header

        self.header['PROFILE_TIME'] = True
        self.header['NUM_PROFILE_TIME_STEPS'] = self.num_profile_steps

        self.channel = self._create_channel()

    def _create_channel(self):
        chip_id = 0
        lmt_id = 0
        channel = self.board.createChannel(
                'time_log_{}_{}'.format(chip_id, lmt_id).encode(),
                messageSize = 4, # 4 bytes of message
                numElements = self.num_profile_steps,
                slack = 32, # add extra slack to avoid hanging
            )
        channel.connect(self.snip(chip_id, lmt_id), None)

        return channel

    def receive(self):
        """Receive timing data.

        Returns:
            Timing data in microseconds.
        """
        results = self.channel.read(self.num_profile_steps)
        results = np.array(results).flatten() / 400 # clock frequency is 400Mhz
        return results # results are in microseconds

snip_dir = os.path.abspath(os.path.dirname(inspect.getfile(IOManagement)) + "/ioSnips")
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)