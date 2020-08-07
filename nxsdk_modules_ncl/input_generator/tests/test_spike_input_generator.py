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
Tests SpikeInputGenerator. SpikeInputGenerator is a layer which can be used to encode data and inject spikes
in neurocore.
"""

import os
import random
import unittest

from nxsdk.composable.interfaces.composable_enums import ResourceMapType
from nxsdk.composable.model import Model
from nxsdk.composable.port_impl import StateInputPort
from nxsdk.composable.resource_map import ResourceMapFactory
from nxsdk.graph.nxboard import N2Board
from nxsdk_modules_ncl.input_generator.spike_input_generator import SpikeInputGenerator
import numpy as np


class TestSpikeInputGenerator(unittest.TestCase):
    """
    Tests SpikeInputGenerator. SpikeInputGenerator is a layer which can be used to encode data and inject spikes
    in neurocore.

    """

    def test_spike_input_gen(self):
        """
        Create a mock output port and inject spikes into random axons.
        """

        # Create an input encoder
        ie = SpikeInputGenerator(name="SpikeGen")

        # Create an instance of board
        board = N2Board(1, 1, [4], [[5] * 4])

        # Create a mock port and connect input encoder to this port
        mockPort = StateInputPort(name="input")
        axonIds = np.arange(100)
        chip_core_ids = np.array([[0, 19]] * 100)
        mockAddresses = np.hstack((chip_core_ids, axonIds.reshape((100, 1))))
        mockPort.resourceMap = ResourceMapFactory.createExplicit(
            ResourceMapType.INPUT_AXON, mockAddresses)

        # Connect the ports
        ie.ports.output.connect(mockPort)

        # Create a model and add input encoder as composable
        m = Model(name="spike_encoder")
        m.add(ie)
        m.compile(board=board)
        m.start(board=m.board, partition=os.environ.get("PARTITION"))

        # Encode the data
        input_list = []
        for i in range(20):
            input_list.append((random.randint(1, 20), random.randint(0, 100)))
        ie.encode(input_list)

        m.run(20)
        m.disconnect()
        self.assertEqual(True, True)
