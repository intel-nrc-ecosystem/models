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

"""Unit test for Input Encoder"""

import os

from nxsdk.composable.interfaces.composable_enums import ResourceMapType
from nxsdk.composable.port_impl import StateInputPort
from nxsdk.composable.resource_map import CompartmentResourceMap, ResourceMapFactory
import unittest
from nxsdk.composable.model import Model
from nxsdk_modules_ncl.input_generator.input_generator import InputGenerator
from nxsdk.graph.nxboard import N2Board
import numpy as np


class TestInputEncoder(unittest.TestCase):
    """Unit test for Input Encoder"""
    def test_input_encoder(self):
        """Test running a compilation pipeline with input encoder and mocked port"""
        # Create an input encoder
        shape = (32, 32, 1, 1)
        ie = InputGenerator(shape)
        a = np.arange(0, 1024*5, 5)
        a = a.reshape(*shape)

        # Create an instance of board
        board = N2Board(1, 1, [4], [[5]*4])

        # Create a mock port and connect input encoder to this port
        mockPort = StateInputPort(name="input")
        cxIds = np.arange(shape[0]*shape[1])
        chip_core_ids = np.array([[0, 19]]*shape[0]*shape[1])
        mockAddresses = np.hstack((chip_core_ids, cxIds.reshape((shape[0]*shape[1],1))))
        mockPort.resourceMap = ResourceMapFactory.createExplicit(ResourceMapType.COMPARTMENT, mockAddresses)

        # Connect the ports
        ie.ports.output.connect(mockPort)

        # Create a model and add input encoder as composable
        m = Model(name="video_encoder")
        m.add(ie)
        m.compile(board=board)
        m.start(board=m.board, partition=os.environ.get("PARTITION"))

        # Encode the data
        ie.encode(a)

        m.run(1)
        m.disconnect()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
