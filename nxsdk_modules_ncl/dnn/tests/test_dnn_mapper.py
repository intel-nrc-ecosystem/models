#
# Copyright © 2018 Intel Corporation.
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

"""Test mapping DNNs onto Loihi via NxCore."""

import unittest

from test import support
import numpy as np

from nxsdk.arch.n2a.n2board import N2Board
from nxsdk_modules_ncl.dnn.src.data_structures import Layer, SynFmt, Partition, \
    SynapseGroup, SynEntry, OutputAxonGroup, InputAxonGroup, CompartmentGroup
from nxsdk_modules_ncl.dnn.src.dnn_mapper import DnnMapper
from nxsdk_modules_ncl.dnn.src.optimization import getDummyLayer


class TestDnnMapper(unittest.TestCase):
    """Test DNN mapper."""

    def test_init(self):
        """Check that object can be constructed."""

        c = DnnMapper(N2Board(0))
        self.assertTrue(isinstance(c, DnnMapper))

    def test_mapInputAxons_1(self):
        """Check mapping of InputAxonGroup with all input nodes having
        multiplicity 1."""

        verbose = False

        layer = Layer(layerId=0, layerType='', compartmentKwargs={},
                      connectionKwargs={"weightExponent": 2},
                      coreIdMap=np.array([]), multiplicityMap=np.array([]),
                      postLayer=None)

        sf = SynFmt(
            synFmtId=0,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=8,
            compression=0,
            signMode=1
        )

        se0 = SynEntry(prefixOffset=0, idxs=np.array([0, 1, 2]),
                       weights=np.array([6, 7, 8]), synFmt=sf)

        se1 = SynEntry(prefixOffset=10, idxs=np.array([0, 1]),
                       weights=np.array([16, 17]), synFmt=sf)

        se2 = SynEntry(prefixOffset=20, idxs=np.array([0, 1, 2]),
                       weights=np.array([26, 27, 28]), synFmt=sf)

        se3 = SynEntry(prefixOffset=30, idxs=np.array([0, 1]),
                       weights=np.array([36, 37]), synFmt=sf)

        sg = SynapseGroup(
            groupId=0,
            synEntries=[[se0, se1], [se1, se2, se3], [se3]]
        )

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)

        p.addSynFmt(sf)

        p.addSynapseGroup(sg)

        iag = InputAxonGroup(
            srcNodeIds=np.array([10, 11, 12]),
            multiplicity=np.array([1, 1, 1]),
            synGroup=sg,
            cxBase=2,
            parentPartition=p
        )

        p.addInputAxonGroup(iag)
        layer.addPartition(p)

        numSyn = 14
        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, numSyn)[0]

        c._mapSynapses(p, core)  # pylint: disable=protected-access
        c._mapInputAxons(p, core)  # pylint: disable=protected-access

        # Validate input axon
        sMap = core.synapseMap[0]
        self.assertEqual(sMap.synapsePtr, 0)
        self.assertEqual(sMap.synapseLen, [5, 7, 2])
        self.assertEqual(sMap.popSize, 3)
        self.assertEqual(sMap.population32MapEntry.ptr, 0)
        self.assertEqual(sMap.population32MapEntry.length, 3)
        self.assertEqual(sMap.population32MapEntry.cxBase, 2)

        # Validate inAxMap
        self.assertTrue(np.array_equal(c._inAxMap[iag.id],
                                       np.array([[0, 0, 4],
                                                 [0, 0, 4],
                                                 [0, 0, 4]], int)))

        if verbose:
            c.printCore(core, inputAxons=True)

    def test_mapInputAxons_2(self):
        """Check mapping of InputAxonGroup with some input nodes having 1
        and >1 multiplicity."""

        verbose = False

        layer = Layer(layerId=0, layerType='', compartmentKwargs={},
                      connectionKwargs={"weightExponent": 2},
                      coreIdMap=np.array([]), multiplicityMap=np.array([]),
                      postLayer=None)

        sf = SynFmt(
            synFmtId=0,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=8,
            compression=0,
            signMode=1
        )

        se0 = SynEntry(prefixOffset=0, idxs=np.array([0, 1, 2]),
                       weights=np.array([6, 7, 8]), synFmt=sf)

        se1 = SynEntry(prefixOffset=10, idxs=np.array([0, 1]),
                       weights=np.array([16, 17]), synFmt=sf)

        se2 = SynEntry(prefixOffset=20, idxs=np.array([0, 1, 2]),
                       weights=np.array([26, 27, 28]), synFmt=sf)

        se3 = SynEntry(prefixOffset=30, idxs=np.array([0, 1]),
                       weights=np.array([36, 37]), synFmt=sf)

        sg = SynapseGroup(
            groupId=0,
            synEntries=[[se0, se1], [se1, se2, se3], [se3]]
        )

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)

        p.addSynFmt(sf)

        p.addSynapseGroup(sg)

        iag = InputAxonGroup(
            srcNodeIds=np.array([10, 11, 12]),
            multiplicity=np.array([1, 2, 1]),
            synGroup=sg,
            cxBase=2,
            parentPartition=p
        )

        p.addInputAxonGroup(iag)
        layer.addPartition(p)

        numSyn = 14
        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, numSyn)[0]

        c._mapSynapses(p, core)  # pylint: disable=protected-access
        c._mapInputAxons(p, core)  # pylint: disable=protected-access

        # Validate shared input axon
        sMapShared = core.synapseMap[0]
        self.assertEqual(sMapShared.synapsePtr, 0)
        self.assertEqual(sMapShared.synapseLen, [5, 7, 2])
        self.assertEqual(sMapShared.popSize, 3)
        self.assertEqual(sMapShared.population32MapEntry.ptr, 0)
        self.assertEqual(sMapShared.population32MapEntry.length, 3)
        self.assertEqual(sMapShared.population32MapEntry.cxBase, 2)

        # Validate discrete input axon
        sMapDiscrete = core.synapseMap[1]
        self.assertEqual(sMapDiscrete.synapsePtr, 5)
        self.assertEqual(sMapDiscrete.synapseLen, 7)
        self.assertEqual(sMapDiscrete.discreteMapEntry.ptr, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.length, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.cxBase, 2)

        # Validate inAxMap
        self.assertTrue(np.array_equal(c._inAxMap[iag.id],
                                       np.array([[0, 0, 4],
                                                 [1, 0, 4],
                                                 [0, 0, 4]], int)))

        if verbose:
            c.printCore(core, inputAxons=True)

    def test_mapInputAxons_3(self):
        """Check mapping of InputAxonGroup with all input nodes having
        multiplicity >1."""

        verbose = False

        layer = Layer(layerId=0, layerType='', compartmentKwargs={},
                      connectionKwargs={"weightExponent": 2},
                      coreIdMap=np.array([]), multiplicityMap=np.array([]),
                      postLayer=None)

        sf = SynFmt(
            synFmtId=0,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=8,
            compression=0,
            signMode=1
        )

        se0 = SynEntry(prefixOffset=0, idxs=np.array([0, 1, 2]),
                       weights=np.array([6, 7, 8]), synFmt=sf)

        se1 = SynEntry(prefixOffset=10, idxs=np.array([0, 1]),
                       weights=np.array([16, 17]), synFmt=sf)

        se2 = SynEntry(prefixOffset=20, idxs=np.array([0, 1, 2]),
                       weights=np.array([26, 27, 28]), synFmt=sf)

        se3 = SynEntry(prefixOffset=30, idxs=np.array([0, 1]),
                       weights=np.array([36, 37]), synFmt=sf)

        sg = SynapseGroup(
            groupId=0,
            synEntries=[[se0, se1], [se1, se2, se3], [se3]]
        )

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)

        p.addSynFmt(sf)

        p.addSynapseGroup(sg)

        iag = InputAxonGroup(
            srcNodeIds=np.array([10, 11, 12]),
            multiplicity=np.array([2, 3, 4]),
            synGroup=sg,
            cxBase=2,
            parentPartition=p
        )

        p.addInputAxonGroup(iag)
        layer.addPartition(p)

        numSyn = 14
        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, numSyn)[0]

        c._mapSynapses(p, core)  # pylint: disable=protected-access
        c._mapInputAxons(p, core)  # pylint: disable=protected-access

        # Validate discrete input axon 0
        sMapDiscrete = core.synapseMap[0]
        self.assertEqual(sMapDiscrete.synapsePtr, 0)
        self.assertEqual(sMapDiscrete.synapseLen, 5)
        self.assertEqual(sMapDiscrete.discreteMapEntry.ptr, 0)
        self.assertEqual(sMapDiscrete.discreteMapEntry.length, 2)
        self.assertEqual(sMapDiscrete.discreteMapEntry.cxBase, 2)

        sMapDiscrete = core.synapseMap[1]
        self.assertEqual(sMapDiscrete.synapsePtr, 5)
        self.assertEqual(sMapDiscrete.synapseLen, 7)
        self.assertEqual(sMapDiscrete.discreteMapEntry.ptr, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.length, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.cxBase, 2)

        sMapDiscrete = core.synapseMap[2]
        self.assertEqual(sMapDiscrete.synapsePtr, 12)
        self.assertEqual(sMapDiscrete.synapseLen, 2)
        self.assertEqual(sMapDiscrete.discreteMapEntry.ptr, 6)
        self.assertEqual(sMapDiscrete.discreteMapEntry.length, 1)
        self.assertEqual(sMapDiscrete.discreteMapEntry.cxBase, 2)

        # Validate inAxMap
        self.assertTrue(np.array_equal(c._inAxMap[iag.id],
                                       np.array([[0, 0, 4],
                                                 [1, 0, 4],
                                                 [2, 0, 4]], int)))

        if verbose:
            c.printCore(core, inputAxons=True)

    def test_mapInputAxons_4(self):
        """Check mapping of multiple InputAxonGroups."""

        verbose = False

        layer = Layer(layerId=0, layerType='', compartmentKwargs={},
                      connectionKwargs={"weightExponent": 2},
                      coreIdMap=np.array([]), multiplicityMap=np.array([]),
                      postLayer=None)

        sf = SynFmt(
            synFmtId=0,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=8,
            compression=0,
            signMode=1
        )

        se0 = SynEntry(prefixOffset=0, idxs=np.array([0, 1, 2]),
                       weights=np.array([6, 7, 8]), synFmt=sf)

        se1 = SynEntry(prefixOffset=10, idxs=np.array([0, 1]),
                       weights=np.array([16, 17]), synFmt=sf)

        se2 = SynEntry(prefixOffset=20, idxs=np.array([0, 1, 2]),
                       weights=np.array([26, 27, 28]), synFmt=sf)

        se3 = SynEntry(prefixOffset=30, idxs=np.array([0, 1]),
                       weights=np.array([36, 37]), synFmt=sf)

        sg0 = SynapseGroup(
            groupId=0,
            synEntries=[[se0, se1], [se1, se2, se3], [se3]]
        )

        sg1 = SynapseGroup(
            groupId=1,
            synEntries=[[se0, se1], [se2], [se1, se2, se3]]
        )

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)

        p.addSynFmt(sf)

        p.addSynapseGroup(sg0)
        p.addSynapseGroup(sg1)

        iag0 = InputAxonGroup(
            srcNodeIds=np.array([10, 11, 12]),
            multiplicity=np.array([1, 1, 1]),
            synGroup=sg0,
            cxBase=1,
            parentPartition=p
        )

        iag1 = InputAxonGroup(
            srcNodeIds=np.array([10, 11, 12]),
            multiplicity=np.array([1, 2, 1]),
            synGroup=sg0,
            cxBase=2,
            parentPartition=p
        )

        iag2 = InputAxonGroup(
            srcNodeIds=np.array([10, 11, 12]),
            multiplicity=np.array([1, 2, 3]),
            synGroup=sg1,
            cxBase=3,
            parentPartition=p
        )

        p.addInputAxonGroup(iag0)
        p.addInputAxonGroup(iag1)
        p.addInputAxonGroup(iag2)
        layer.addPartition(p)

        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, p.numSyn)[0]

        c._mapSynapses(p, core)  # pylint: disable=protected-access
        c._mapInputAxons(p, core)  # pylint: disable=protected-access

        # Validate axons of iag0
        sMapShared = core.synapseMap[0]
        self.assertEqual(sMapShared.synapsePtr, 0)
        self.assertEqual(sMapShared.synapseLen, [5, 7, 2])
        self.assertEqual(sMapShared.popSize, 3)
        self.assertEqual(sMapShared.population32MapEntry.ptr, 0)
        self.assertEqual(sMapShared.population32MapEntry.length, 3)
        self.assertEqual(sMapShared.population32MapEntry.cxBase, 1)

        # Validate axons of iag1
        sMapShared = core.synapseMap[1]
        self.assertEqual(sMapShared.synapsePtr, 0)
        self.assertEqual(sMapShared.synapseLen, [5, 7, 2])
        self.assertEqual(sMapShared.popSize, 3)
        self.assertEqual(sMapShared.population32MapEntry.ptr, 0)
        self.assertEqual(sMapShared.population32MapEntry.length, 3)
        self.assertEqual(sMapShared.population32MapEntry.cxBase, 2)

        sMapDiscrete = core.synapseMap[2]
        self.assertEqual(sMapDiscrete.synapsePtr, 5)
        self.assertEqual(sMapDiscrete.synapseLen, 7)
        self.assertEqual(sMapDiscrete.discreteMapEntry.ptr, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.length, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.cxBase, 2)

        # Validate axons of iag2
        sMapShared = core.synapseMap[3]
        self.assertEqual(sMapShared.synapsePtr, 14)
        self.assertEqual(sMapShared.synapseLen, [5, 3, 7])
        self.assertEqual(sMapShared.popSize, 3)
        self.assertEqual(sMapShared.population32MapEntry.ptr, 9)
        self.assertEqual(sMapShared.population32MapEntry.length, 3)
        self.assertEqual(sMapShared.population32MapEntry.cxBase, 3)

        sMapDiscrete = core.synapseMap[4]
        self.assertEqual(sMapDiscrete.synapsePtr, 19)
        self.assertEqual(sMapDiscrete.synapseLen, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.ptr, 12)
        self.assertEqual(sMapDiscrete.discreteMapEntry.length, 1)
        self.assertEqual(sMapDiscrete.discreteMapEntry.cxBase, 3)

        sMapDiscrete = core.synapseMap[5]
        self.assertEqual(sMapDiscrete.synapsePtr, 22)
        self.assertEqual(sMapDiscrete.synapseLen, 7)
        self.assertEqual(sMapDiscrete.discreteMapEntry.ptr, 15)
        self.assertEqual(sMapDiscrete.discreteMapEntry.length, 3)
        self.assertEqual(sMapDiscrete.discreteMapEntry.cxBase, 3)

        # Validate inAxMap
        self.assertTrue(np.array_equal(c._inAxMap[iag0.id],
                                       np.array([[0, 0, 4],
                                                 [0, 0, 4],
                                                 [0, 0, 4]], int)))
        self.assertTrue(np.array_equal(c._inAxMap[iag1.id],
                                       np.array([[1, 0, 4],
                                                 [2, 0, 4],
                                                 [1, 0, 4]], int)))
        self.assertTrue(np.array_equal(c._inAxMap[iag2.id],
                                       np.array([[3, 0, 4],
                                                 [4, 0, 4],
                                                 [5, 0, 4]], int)))

        if verbose:
            c.printCore(core, inputAxons=True)

    def test_mapSynapses_1(self):
        """Check allocation of synFmts."""

        verbose = False

        layer = Layer(layerId=0, layerType='', compartmentKwargs={},
                      connectionKwargs={"weightExponent": 2},
                      coreIdMap=np.array([]), multiplicityMap=np.array([]),
                      postLayer=None)

        sf0 = SynFmt(
            synFmtId=0,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=8,
            compression=0,
            signMode=2
        )

        sf1 = SynFmt(
            synFmtId=1,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=6,
            compression=0,
            signMode=3
        )

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)

        p.addSynFmt(sf0)
        p.addSynFmt(sf1)

        layer.addPartition(p)

        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, 0)[0]

        c._mapSynapses(p, core)  # pylint: disable=protected-access

        if verbose:
            c.printCore(core, synFmts=True)

        # Validate number of synFmts
        self.assertEqual(len(core.synapseFmt.modified), 4)

        # Validate duplication of synFmts by checking unique fanoutTypes
        self.assertEqual(core.synapseFmt[1].fanoutType, 2)
        self.assertEqual(core.synapseFmt[2].fanoutType, 2)
        self.assertEqual(core.synapseFmt[3].fanoutType, 3)
        self.assertEqual(core.synapseFmt[4].fanoutType, 3)

        # Validate mapping of numWgtBits -> wgtBits
        self.assertEqual(core.synapseFmt[1].wgtBits, 7)
        self.assertEqual(core.synapseFmt[3].wgtBits, 6)

        # Validate wgtExp
        self.assertEqual(core.synapseFmt[1].wgtExp, 2)

    def test_mapSynapses_2(self):
        """Check switching between synFmt for successive synEntries if
        synFmtId does not change."""

        verbose = False

        layer = Layer(layerId=0, layerType='', compartmentKwargs={},
                      connectionKwargs={"weightExponent": 2},
                      coreIdMap=np.array([]), multiplicityMap=np.array([]),
                      postLayer=None)

        sf0 = SynFmt(
            synFmtId=0,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=8,
            compression=0,
            signMode=1
        )

        sf1 = SynFmt(
            synFmtId=1,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=7,
            numSkipBits=0,
            numWgtBits=8,
            compression=1,
            signMode=1
        )

        se00 = SynEntry(prefixOffset=0, idxs=np.array([0, 1, 2]),
                        weights=np.array([6, 7, 8]), synFmt=sf0)

        se01 = SynEntry(prefixOffset=10, idxs=np.array([0, 1]),
                        weights=np.array([16, 17]), synFmt=sf0)

        se02 = SynEntry(prefixOffset=20, idxs=np.array([0, 1, 2]),
                        weights=np.array([26, 27, 28]), synFmt=sf0)

        se10 = SynEntry(prefixOffset=30, idxs=np.array([0, 1]),
                        weights=np.array([36, 37]), synFmt=sf1)

        sg = SynapseGroup(
            groupId=0,
            synEntries=[[se00, se01, se02], [se10]]
        )

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)

        p.addSynFmt(sf0)
        p.addSynFmt(sf1)

        p.addSynapseGroup(sg)

        layer.addPartition(p)

        numSyn = 10
        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, numSyn)[0]

        c._mapSynapses(p, core)  # pylint: disable=protected-access

        if verbose:
            c.printCore(core, synapses=True)

        cIdx = [0, 1, 2, 10, 11, 20, 21, 22, 30, 31]
        wgts = [6, 7, 8, 16, 17, 26, 27, 28, 36, 37]
        synFmtIds = [1, 1, 1, 2, 2, 1, 1, 1, 3, 3]

        for i in range(numSyn):
            self.assertEqual(core.synapses[i].CIdx, cIdx[i])
            self.assertEqual(core.synapses[i].Wgt, wgts[i])
            self.assertEqual(core.synapses[i].synFmtId, synFmtIds[i])

    def test_mapSynapses_3(self):
        """Check generation of synGrpMap."""

        verbose = False

        layer = Layer(layerId=0, layerType='', compartmentKwargs={},
                      connectionKwargs={"weightExponent": 2},
                      coreIdMap=np.array([]), multiplicityMap=np.array([]),
                      postLayer=None)

        sf = SynFmt(
            synFmtId=0,
            cIdxOffset=0,
            cIdxMult=2,
            numIdxBits=6,
            numSkipBits=0,
            numWgtBits=8,
            compression=0,
            signMode=1
        )

        se0 = SynEntry(prefixOffset=0, idxs=np.array([0, 1, 2]),
                       weights=np.array([6, 7, 8]), synFmt=sf)

        se1 = SynEntry(prefixOffset=10, idxs=np.array([0, 1]),
                       weights=np.array([16, 17]), synFmt=sf)

        se2 = SynEntry(prefixOffset=20, idxs=np.array([0, 1, 2]),
                       weights=np.array([26, 27, 28]), synFmt=sf)

        se3 = SynEntry(prefixOffset=30, idxs=np.array([0, 1]),
                       weights=np.array([36, 37]), synFmt=sf)

        sg0 = SynapseGroup(
            groupId=0,
            synEntries=[[se0, se1, se2], [se3]]
        )

        sg1 = SynapseGroup(
            groupId=1,
            synEntries=[[se0], [se1], [se2, se3]]
        )

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)

        p.addSynFmt(sf)

        p.addSynapseGroup(sg0)
        p.addSynapseGroup(sg1)

        layer.addPartition(p)

        numSyn = 10 * 2
        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, numSyn)[0]

        c._mapSynapses(p, core)  # pylint: disable=protected-access

        if verbose:
            print(c._synGrpMap)

        self.assertEqual(len(c._synGrpMap), 5)
        self.assertEqual(c._synGrpMap[(sg0, 0)], (0, 8))
        self.assertEqual(c._synGrpMap[(sg0, 1)], (8, 2))
        self.assertEqual(c._synGrpMap[(sg1, 0)], (10, 3))
        self.assertEqual(c._synGrpMap[(sg1, 1)], (13, 2))
        self.assertEqual(c._synGrpMap[(sg1, 2)], (15, 5))

    def test_mapCompartments_1(self):
        """Check mapping of CompartmentGroups without soma traces."""

        verbose = False

        c = CompartmentGroup(cxIds=np.arange(15), biasMant=np.concatenate(
            [range(10, 0, -1), range(5, 0, -1)]), biasExp=np.zeros(15, int),
                             relToAbsDestCxIdxMap=np.array([], int))

        layer = Layer(
            layerId=0, layerType='',
            compartmentKwargs={
                "compartmentCurrentDecay": 4096,
                "compartmentVoltageDecay": 10,
                "refractoryDelay": 30,
                "vThMant": 40,
                "enableSomaTrace": 0,
                "threshOp": 0},
            connectionKwargs={},
            coreIdMap=np.array([]),
            multiplicityMap=np.array([]),
            postLayer=None)

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)
        p.addCompartmentGroup(c)

        layer.addPartition(p)

        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, 0)[0]

        c._mapCompartments(p, core)  # pylint: disable=protected-access

        if verbose:
            c.printCore(core, compartments=True)

        self.assertEqual(len(core.cxCfg.modified), 15)

        self.assertEqual(core.numUpdates.numUpdates, 4)

        self.assertEqual(len(core.cxMetaState.modified), 4)
        self.assertEqual(core.cxMetaState[3].phase2, 2)
        self.assertEqual(core.cxMetaState[3].somaOp2, 0)
        self.assertEqual(core.cxMetaState[3].phase3, 2)
        self.assertEqual(core.cxMetaState[3].somaOp3, 0)
        self.assertEqual(core.cxMetaState[4].phase2, 0)
        self.assertEqual(core.cxMetaState[4].somaOp2, 0)

        self.assertEqual(core.dendriteSharedCfg[0].dsOffset, 1)
        self.assertEqual(core.dendriteSharedCfg[0].dmOffsets, 0)

        self.assertEqual(core.cxProfileCfg[0].refractDelay, 30)
        self.assertEqual(core.cxProfileCfg[0].decayU, 4095)
        self.assertEqual(core.cxProfileCfg[0].decayV, 10)

        self.assertEqual(core.vthProfileCfg[0].staticCfg.vth, 40)

    def test_mapCompartments_2(self):
        """Check mapping of CompartmentGroups with soma traces."""

        verbose = False

        c = CompartmentGroup(cxIds=np.arange(15), biasMant=np.concatenate(
            [range(10, 0, -1), range(5, 0, -1)]), biasExp=np.zeros(15, int),
                             relToAbsDestCxIdxMap=np.array([], int))

        layer = Layer(layerId=0, layerType='',
                      compartmentKwargs={"compartmentCurrentDecay": 4096,
                                         "compartmentVoltageDecay": 10,
                                         "refractoryDelay": 30,
                                         "vThMant": 40,
                                         "enableSomaTrace": 1,
                                         "threshOp": 0},
                      connectionKwargs={}, coreIdMap=np.array([]),
                      multiplicityMap=np.array([]), postLayer=None)

        p = Partition(partitionId=0, chipCounter=0, sizeInterleaved=-1,
                      parentLayer=layer)
        p.addCompartmentGroup(c)

        layer.addPartition(p)

        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, 0)[0]

        c._mapCompartments(p, core)  # pylint: disable=protected-access

        if verbose:
            c.printCore(core, compartments=True)

        self.assertEqual(len(core.somaState.modified), 15)
        self.assertEqual(core.somaState[0].vth, 40)
        self.assertEqual(core.vthProfileCfg[0].dynamicCfg.enableHomeostasis, 1)
        self.assertEqual(core.vthProfileCfg[0].dynamicCfg.aMax, 127)

    def test_mapOutputAxons(self):
        """Check mapping of OutputAxonGroups."""

        verbose = False

        # Create destination layer
        lDst = Layer(layerId=0, layerType='foo', compartmentKwargs={},
                     connectionKwargs={}, coreIdMap=np.array([]),
                     multiplicityMap=np.array([]),
                     postLayer=getDummyLayer((1,)))

        # Create destination partition with InputAxonGroups
        pDst = Partition(partitionId=0, chipCounter=10, sizeInterleaved=-1,
                         parentLayer=lDst)

        sgDst = SynapseGroup(
            groupId=0,
            synEntries=[[]]
        )

        iag0 = InputAxonGroup(
            srcNodeIds=np.array([10, 11, 12]),
            multiplicity=np.array([1, 1, 1]),
            synGroup=sgDst,
            cxBase=1,
            parentPartition=pDst
        )

        iag1 = InputAxonGroup(
            srcNodeIds=np.array([20, 21, 22]),
            multiplicity=np.array([1, 2, 1]),
            synGroup=sgDst,
            cxBase=2,
            parentPartition=pDst
        )

        iag2 = InputAxonGroup(
            srcNodeIds=np.array([30, 31, 32]),
            multiplicity=np.array([2, 3, 4]),
            synGroup=sgDst,
            cxBase=3,
            parentPartition=pDst
        )

        pDst.addInputAxonGroup(iag0)
        pDst.addInputAxonGroup(iag1)
        pDst.addInputAxonGroup(iag2)
        lDst.addPartition(pDst)

        # Create source layer
        lSrc = Layer(layerId=0, layerType='foo', compartmentKwargs={},
                     connectionKwargs={}, coreIdMap=np.array([]),
                     multiplicityMap=np.array([]), postLayer=lDst)

        # Create source partition with output axons
        pSrc = Partition(partitionId=1, chipCounter=8, sizeInterleaved=-1,
                         parentLayer=lSrc)

        oag0 = OutputAxonGroup(
            cxIds=np.array([0, 1, 2]),
            multiplicity=np.array([1, 1, 1]),
            relSrcIds=np.array([0, 1, 2]),
            inAxGrp=iag0,
            parentPartition=pSrc
        )

        oag1 = OutputAxonGroup(
            cxIds=np.array([3, 4, 5]),
            multiplicity=np.array([1, 2, 1]),
            relSrcIds=np.array([0, 1, 2]),
            inAxGrp=iag1,
            parentPartition=pSrc
        )

        oag2 = OutputAxonGroup(
            cxIds=np.array([6, 7, 8]),
            multiplicity=np.array([2, 3, 4]),
            relSrcIds=np.array([0, 1, 2]),
            inAxGrp=iag2,
            parentPartition=pSrc
        )

        pSrc.addOutputAxonGroup(oag0)
        pSrc.addOutputAxonGroup(oag1)
        pSrc.addOutputAxonGroup(oag2)
        lSrc.addPartition(pSrc)

        # Initialize board and mapper
        board = N2Board(0)
        c = DnnMapper(board)
        core = board.allocateCores(1, 0)[0]

        # Create fake inAxMap as if it was created by mapInputAxons
        c._inAxMap[iag0.id] = np.array([[10, 10, 1],
                                       [10, 10, 1],
                                       [10, 10, 1]], int)
        c._inAxMap[iag1.id] = np.array([[11, 10, 1],
                                       [12, 10, 1],
                                       [11, 10, 1]], int)
        c._inAxMap[iag2.id] = np.array([[13, 10, 1],
                                       [14, 10, 1],
                                       [15, 10, 1]], int)

        # First call: nothing will happen.
        c._mapOutputAxons(pDst, core)  # pylint: disable=protected-access
        c._mapOutputAxons(pSrc, core)  # pylint: disable=protected-access

        # Validate output axons
        # OutputAxonGroup 0
        self.assertEqual(core.axons[0].srcCxId, 0)
        self.assertEqual(core.axons[0].srcRelCxId, 0)
        self.assertEqual(core.axons[0].dstChipId, 10)
        self.assertEqual(core.axons[0].dstCoreId, 1)
        self.assertEqual(core.axons[0].dstSynMapId, 10)
        self.assertEqual(core.axons[0].axonType, 2)  # pop32
        self.assertEqual(core.axons[0].popId, 0)

        self.assertEqual(core.axons[1].srcCxId, 1)
        self.assertEqual(core.axons[1].srcRelCxId, 1)
        self.assertEqual(core.axons[1].dstChipId, 10)
        self.assertEqual(core.axons[1].dstCoreId, 1)
        self.assertEqual(core.axons[1].dstSynMapId, 10)
        self.assertEqual(core.axons[1].axonType, 2)  # pop32
        self.assertEqual(core.axons[1].popId, 1)

        self.assertEqual(core.axons[2].srcCxId, 2)
        self.assertEqual(core.axons[2].srcRelCxId, 2)
        self.assertEqual(core.axons[2].dstChipId, 10)
        self.assertEqual(core.axons[2].dstCoreId, 1)
        self.assertEqual(core.axons[2].dstSynMapId, 10)
        self.assertEqual(core.axons[2].axonType, 2)  # pop32
        self.assertEqual(core.axons[2].popId, 2)

        # OutputAxonGroup 1
        self.assertEqual(core.axons[3].srcCxId, 3)
        self.assertEqual(core.axons[3].srcRelCxId, 0)
        self.assertEqual(core.axons[3].dstChipId, 10)
        self.assertEqual(core.axons[3].dstCoreId, 1)
        self.assertEqual(core.axons[3].dstSynMapId, 11)
        self.assertEqual(core.axons[3].axonType, 2)  # pop32
        self.assertEqual(core.axons[3].popId, 3)

        self.assertEqual(core.axons[4].srcCxId, 4)
        self.assertEqual(core.axons[4].dstChipId, 10)
        self.assertEqual(core.axons[4].dstCoreId, 1)
        self.assertEqual(core.axons[4].dstSynMapId, 12)
        self.assertEqual(core.axons[4].axonType, 0)  # discrete

        self.assertEqual(core.axons[5].srcCxId, 5)
        self.assertEqual(core.axons[5].srcRelCxId, 2)
        self.assertEqual(core.axons[5].dstChipId, 10)
        self.assertEqual(core.axons[5].dstCoreId, 1)
        self.assertEqual(core.axons[5].dstSynMapId, 11)
        self.assertEqual(core.axons[5].axonType, 2)  # pop32
        self.assertEqual(core.axons[5].popId, 4)

        # OutputAxonGroup 1
        self.assertEqual(core.axons[6].srcCxId, 6)
        self.assertEqual(core.axons[6].dstChipId, 10)
        self.assertEqual(core.axons[6].dstCoreId, 1)
        self.assertEqual(core.axons[6].dstSynMapId, 13)
        self.assertEqual(core.axons[6].axonType, 0)  # discrete

        self.assertEqual(core.axons[7].srcCxId, 7)
        self.assertEqual(core.axons[7].dstChipId, 10)
        self.assertEqual(core.axons[7].dstCoreId, 1)
        self.assertEqual(core.axons[7].dstSynMapId, 14)
        self.assertEqual(core.axons[7].axonType, 0)  # discrete

        self.assertEqual(core.axons[8].srcCxId, 8)
        self.assertEqual(core.axons[8].dstChipId, 10)
        self.assertEqual(core.axons[8].dstCoreId, 1)
        self.assertEqual(core.axons[8].dstSynMapId, 15)
        self.assertEqual(core.axons[8].axonType, 0)  # discrete

        if verbose:
            c.printCore(core, outputAxons=True)


def main():
    support.run_unittest(TestDnnMapper)


if __name__ == '__main__':
    main()
