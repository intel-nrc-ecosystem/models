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

os.environ['SLURM'] = '1'

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

        n2Core = board.n2Chips[0].n2Cores[0]

        n2Core.cxProfileCfg[0].configure(decayV=int(2 ** 12 - 1),
                                         decayU=int(2 ** 12 - 1))

        n2Core.cxMetaState[0].configure(phase0=2)

        n2Core.vthProfileCfg[0].staticCfg.configure(vth=1024)
        n2Core.numUpdates.configure(numUpdates=1)
        n2Core.cxCfg[0].configure(
            bias=0,
            biasExp=0,
            vthProfile=0,
            cxProfile=0)

        n2Core.synapseMap[0].synapsePtr = 0
        n2Core.synapseMap[0].synapseLen = 1
        n2Core.synapseMap[0].discreteMapEntry.configure()
        n2Core.synapses[0].CIdx = 0
        n2Core.synapses[0].Wgt = 255
        n2Core.synapses[0].synFmtId = 1
        n2Core.synapseFmt[1].wgtBits = 7
        n2Core.synapseFmt[1].numSynapses = 63
        n2Core.synapseFmt[1].idxBits = 1
        n2Core.synapseFmt[1].compression = 3
        n2Core.synapseFmt[1].fanoutType = 1

        mon = board.monitor
        vProbes = mon.probe(n2Core.cxState, [0], 'v')[0]

        # Create a mock port and connect input encoder to this port
        mockPort = StateInputPort(name="input")
        num_axons = 100
        dst_core_id = 4
        chip_id = 0
        axonIds = np.arange(num_axons)
        chip_core_ids = np.array([[chip_id, dst_core_id]] * num_axons)
        mockAddresses = np.hstack((chip_core_ids, np.expand_dims(axonIds, 1)))
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
        num_timesteps = 200
        input_list = [(0, t) for t in range(1, num_timesteps)]
        ie.encode(input_list)

        m.run(num_timesteps)
        vProbes.plot()
        import matplotlib.pyplot as plt
        plt.show()

        m.disconnect()
        self.assertGreater(np.max(vProbes.data), 0)
