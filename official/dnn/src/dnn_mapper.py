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

from collections import OrderedDict
from typing import TYPE_CHECKING

import math
import numpy as np

from nxsdk.arch.n2a.compiler.synapsegen.synapse_compiler import \
    N2ASynapseCompiler as SynCompiler
from nxsdk.arch.n2a.compiler.tracecfggen.tracecfggen import TraceCfgGen

if TYPE_CHECKING:
    from nxsdk_modules.dnn.src.data_structures import Partition, Layer
    from nxsdk.arch.n2a.graph.n2acore import N2ACore


# ToDo: Check if NodeSet configuration can be vectorized.
class DnnMapper:
    """Map a Deep Neural Network onto Loihi."""

    def __init__(self, n2Board):
        """Initialize ``DnnMapper``.

        :param N2Board n2Board: The N2Board in which to allocate resources.
        """

        self._board = n2Board
        self._inAxMap = OrderedDict()
        self._synGrpMap = OrderedDict()
        self._isSomaTraceEnabled = False

    def _mapInputAxons(self, partition, core):
        """Map input axons to NeuroCore.

        Input axons are mapped in two stages because the InputAxonGroups may
        contain shared and discrete axons that share the same SynapseGroup.

        First, a shared dummy input axon is created for each
        InputAxonGroup/SynapseGroup, and the synapses are manually compiled
        with the SynapseCompiler. This process stores the synapses for each
        input node of a SynapseGroup sequentially and in order in SynapseMem.

        Second, the actual shared and discrete axons are created for each
        SynapseGroup which overwrites the previous dummy axons. Then, the
        synapseMemPtr and synapseMemLen of each axon are set manually to
        point to the right set of SynapseMem entries.

        :param Partition partition: Layer partition.
        :param N2ACore core: Core to map to.
        """

        # Create dummy shared axons just to compile synGrps into one
        # contiguous block of memory
        synLensPerAxGrp = []
        for i, synGrp in enumerate(partition.synapseGroups):
            popSize = len(synGrp.synEntries)
            synLens = [self._synGrpMap[(synGrp, j)][1] for j in range(popSize)]
            synPtr, _ = self._synGrpMap[(synGrp, 0)]
            core.synapseMap[i].synapsePtr = synPtr
            core.synapseMap[i].synapseLen = synLens
            core.synapseMap[i].popSize = popSize
            core.synapseMap[i].population32MapEntry.configure()
            synLensPerAxGrp.append(synLens)

        # Compile synapses to obtain HW addresses. The encodeSynapse method
        # returns a list of all synMemLen for each sequential input node.
        synapseMapping = SynCompiler().encodeSynapses(core)

        synMapping = OrderedDict()
        synMemPtr = 0
        for synGrpId in synapseMapping:
            sm = synapseMapping[synGrpId]
            popSize = len(sm)
            synMemLens = [sm[j].size for j in range(popSize)]
            maxSynMemLen = max(synMemLens)
            synMapping[synGrpId] = (synMemPtr, maxSynMemLen, synMemLens)
            synMemPtr += popSize * maxSynMemLen

        # Create actual axons.
        synMapPtr = 0  # Gets incremented after each input axon
        for i, inAxGrp in enumerate(partition.inputAxonGroups):
            # inAxMap maps each node in inAxGrp to its HW address (by default
            # this is the address of the shared axon per inAxGrp).
            inAxMap = np.ones((inAxGrp.numNodes, 3), int)
            inAxMap[:, 0] = synMapPtr
            inAxMap[:, 1] = core.parent.id
            inAxMap[:, 2] = core.id
            synGrpId = inAxGrp.synGroup.id
            popSize = inAxGrp.numNodes
            synMemPtr = synMapping[synGrpId][0]
            maxSynMemLen = synMapping[synGrpId][1]
            synMemLen = synMapping[synGrpId][2]

            # Create shared input axon if there's a node with multiplicity == 1
            if np.any(inAxGrp.multiplicity == 1):
                synPtr, _ = self._synGrpMap[(inAxGrp.synGroup, 0)]
                core.synapseMap[synMapPtr].synapsePtr = synPtr
                core.synapseMap[synMapPtr].synapseLen = \
                    synLensPerAxGrp[synGrpId]
                core.synapseMap[synMapPtr].popSize = popSize
                core.synapseMap[synMapPtr].population32MapEntry.configure(
                    ptr=synMemPtr,
                    length=maxSynMemLen,
                    cxBase=inAxGrp.cxBase)
                synMapPtr += 1

            # Create discrete axons if there are nodes with multiplicity > 1.
            discreteIdx = np.where(inAxGrp.multiplicity > 1)[0]
            for j in discreteIdx:
                # Overwrite default shared axon HW address by discrete axon
                # address.
                inAxMap[j, 0] = synMapPtr
                # Get synMemLen for j-th input node.
                synPtr, synLen = self._synGrpMap[(inAxGrp.synGroup, j)]
                core.synapseMap[synMapPtr].synapsePtr = synPtr
                core.synapseMap[synMapPtr].synapseLen = synLen
                core.synapseMap[synMapPtr].discreteMapEntry.configure(
                    ptr=synMemPtr + j*maxSynMemLen,
                    length=synMemLen[j],
                    cxBase=inAxGrp.cxBase)
                synMapPtr += 1

            # Build inAxGrp -> inAxMap map for output axon creation on
            # previous layer.
            self._inAxMap[inAxGrp.id] = inAxMap

        # Prevent that synapses get compiled again
        core.synapseMap.synMapIds = set()

    def _mapSynapses(self, partition, core):
        """Map synapses to NeuroCore.

        :param Partition partition: Layer partition.
        :param N2ACore core: Core to map to.
        """

        sharedCfgs = partition.layer.connectionKwargs

        # Build synFmts
        for i, sf in enumerate(partition.synFmts):
            wgtBits = sf.numWgtBits if sf.numWgtBits != 8 else 7
            dlyBits = sf.numDlyBits

            # Check if synFmt is used for soft-reset mode
            if partition.resetMode == 'soft' and sf.softReset:
                wgtExp = sharedCfgs["weightExpSR"]
            else:
                wgtExp = sharedCfgs["weightExponent"]

            # Original synFmt
            core.synapseFmt[1 + i * 2 + 0].configure(
                compression=int(sf.compression),
                numSynapses=63,
                wgtExp=wgtExp,
                wgtBits=wgtBits,
                dlyBits=dlyBits,
                skipBits=sf.numSkipBits,
                idxBits=sf.numIdxBits - 5,
                cIdxOffset=sf.cIdxOffset,
                cIdxMult=sf.cIdxMult,
                fanoutType=sf.signMode)
            # Copy of synFmt to enforce synEntry switching
            core.synapseFmt[1 + i * 2 + 1].configure(
                compression=int(sf.compression),
                numSynapses=63,
                wgtExp=wgtExp,
                wgtBits=wgtBits,
                dlyBits=dlyBits,
                skipBits=sf.numSkipBits,
                idxBits=sf.numIdxBits - 5,
                cIdxOffset=sf.cIdxOffset,
                cIdxMult=sf.cIdxMult,
                fanoutType=sf.signMode)

        # Build synapses
        j = 0
        for synGrp in partition.synapseGroups:
            for col, synEntriesPerCol in enumerate(synGrp.synEntries):
                prevSynFmtId = -1
                synPtr = j

                for synEntry in synEntriesPerCol:
                    # Different synFmts are spaced by 2 to leave room for copy
                    synFmtId = synEntry.synFmtId * 2

                    # Enforce new synEntry by jumping to copy if synFmtId
                    # does not change
                    if prevSynFmtId == synFmtId:
                        synFmtId += 1
                    prevSynFmtId = synFmtId

                    # Store synapses of synEntry
                    for i in range(synEntry.numSyn):
                        synCfgKwargs = {
                            'synFmtId': 1 + synFmtId,
                            'CIdx': synEntry.prefixOffset + synEntry.idxs[i],
                            'Wgt': synEntry.weights[i]}

                        if synEntry.delays is not None:
                            synCfgKwargs.update({'Dly': synEntry.delays[i]})

                        core.synapses[j].configure(**synCfgKwargs)

                        j += 1

                # Build (synGrp, col)->(synPtr, synLen) map for axon creation
                synLen = j - synPtr
                self._synGrpMap[(synGrp, col)] = (synPtr, synLen)

    def _mapCompartments(self, partition, core):
        """Map compartments to NeuroCore.

        :param Partition partition: Layer partition.
        :param N2ACore core: Core to map to.
        """

        sharedCfgs = partition.layer.compartmentKwargs
        cxGrp = partition.compartmentGroup

        assert len(cxGrp.cxIds) == len(set(cxGrp.cxIds)), \
            "CompartmentGroup in partition {} contains duplicates.".format(
                partition.id)

        # Configure discrete states
        # Multi-compartment neurons profiles are interleaved by neuronSize
        neuronSize = 2 if partition.resetMode == 'soft' else 1
        maxCxId = max(cxGrp.cxIds)
        for cxId in range(maxCxId + 1):
            bias = cxGrp.biasMant[cxId]
            biasExp = cxGrp.biasExp[cxId]
            profile = (cxId + 2) % neuronSize
            core.cxCfg[cxId].configure(
                bias=bias,
                biasExp=biasExp,
                cxProfile=profile,
                vthProfile=0
            )

        # Configure number of compartments to update
        numCxGroups = int(math.ceil((maxCxId + 1) / 4))
        core.numUpdates[0].configure(numUpdates=numCxGroups)

        # Configure compartments to start out in IDLE phase
        somaOp = 3 if sharedCfgs['enableSomaTrace'] else 0
        for i in range(numCxGroups):
            core.cxMetaState[i].configure(
                phase0=2, phase1=2, phase2=2, phase3=2,
                somaOp0=somaOp, somaOp1=somaOp, somaOp2=somaOp, somaOp3=somaOp)

        # Configure dendritic accumulators to max delay of 8 to support 1024
        # compartments
        core.dendriteAccumCfg.delayBits = 3

        # Configure decay constants
        decayU = sharedCfgs["compartmentCurrentDecay"]
        decayV = sharedCfgs["compartmentVoltageDecay"]
        core.dendriteSharedCfg[0].configure(
            dsOffset=int(decayU == 4096),
            dmOffsets=int(decayV == 4096),
            negVmLimit=23,
            posVmLimit=7,
        )

        cxProfile0Kwargs = {'decayU': decayU if decayU < 4096 else 4095,
                            'decayV': decayV if decayV < 4096 else 4095,
                            'refractDelay': sharedCfgs["refractoryDelay"],
                            'threshOp': sharedCfgs['threshOp']}

        if partition.resetMode == 'soft':
            cxProfile1Kwargs = cxProfile0Kwargs.copy()
            cxProfile0Kwargs.update({'threshOp': 2,
                                     'stackOut': 1,
                                     'refractDelay': 0})
            cxProfile1Kwargs.update({'decayU': 4095,
                                     'decayV': 4095,
                                     'bapAction': 0,
                                     'threshOp': 0,
                                     'stackIn': 2,
                                     'joinOp': 6,
                                     'refractDelay': 0})
            core.cxProfileCfg[1].configure(**cxProfile1Kwargs)

        core.cxProfileCfg[0].configure(**cxProfile0Kwargs)

        # Voltage Threshold
        vThMant = sharedCfgs["vThMant"]

        # Enable somaTrace for spike recording (debug only)
        if sharedCfgs["enableSomaTrace"] == 1:
            self._isSomaTraceEnabled = True
            # Soma traces are only active when homeostasis is enabled
            core.vthProfileCfg[0].dynamicCfg.configure(
                enableHomeostasis=1,
                beta=0,
                aMin=0,
                aMax=127
            )

            # Set trace update interval to 1 to see spikes every time step
            core.dendriteTimeState[0].tepoch = 1

            # Configure soma traces to decay immediately
            tcg = TraceCfgGen()
            tc = tcg.genTraceCfg(tau=1,
                                 spikeLevelInt=127,
                                 spikeLevelFrac=0)
            tc.writeToRegister(core.somaTraceCfg[0])

        # Configure voltage threshold
        if sharedCfgs["enableSomaTrace"] == 0:
            # Threshold can be stored in shared location when homeostasis is
            # off.
            core.vthProfileCfg[0].staticCfg.configure(
                vth=vThMant, useSomaVth=0)
        else:
            # Threshold must be stored discretely for each compartment when
            # homeostasis is on
            for cxId in range(maxCxId+1):
                core.somaState[cxId].configure(vth=vThMant)

    def _mapOutputAxons(self, partition, core):
        """Map output axons to NeuroCore.

        :param Partition partition: Layer partition.
        :param N2ACore core: Core to map to.
        """

        if partition.layer.numOutputAxons == 0:
            return

        cxIdsToPopMap = {}
        i = 0
        for outAxGrp in partition.outputAxonGroups:
            inAxMap = self._inAxMap[outAxGrp.inAxGrpId]
            for j in range(outAxGrp.numNodes):
                relSrcId = outAxGrp.relSrcIds[j]
                if outAxGrp.multiplicity[j] == 1:
                    cxHash = hash((outAxGrp.cxIds.tobytes(), j))
                    if cxHash in cxIdsToPopMap:
                        popId = cxIdsToPopMap[cxHash]
                    else:
                        cxIdsToPopMap[cxHash] = i
                        popId = i
                        i += 1

                    core.createPop32Axon(
                        popId=popId,
                        srcCxId=outAxGrp.cxIds[j],
                        srcRelCxId=relSrcId,
                        dstChipId=inAxMap[relSrcId, 1],
                        dstCoreId=inAxMap[relSrcId, 2],
                        dstSynMapId=inAxMap[relSrcId, 0])

                else:
                    if relSrcId >= len(inAxMap):
                        relSrcId = j
                    core.createDiscreteAxon(
                        srcCxId=outAxGrp.cxIds[j],
                        dstChipId=inAxMap[relSrcId, 1],
                        dstCoreId=inAxMap[relSrcId, 2],
                        dstSynMapId=inAxMap[relSrcId, 0])

    # Todo: Merge _mapcomplexCompartments into _mapCompartments
    @staticmethod
    def _mapComplexCompartments(partition, core):
        """Map complex partition compartments to NeuroCore.

        :param Partition partition: Layer partition.
        :param N2ACore core: Core to map to.
        """

        sharedCfgs = partition.layer.compartmentKwargs
        connectionCfgs = partition.layer.connectionKwargs

        cxGrp = partition.compartmentGroup

        assert len(cxGrp.cxIds) == len(set(cxGrp.cxIds)), \
            "CompartmentGroup in partition {} contains duplicates.".format(
                partition.id)

        # Configure discrete states
        maxCxId = 0
        for i, cxId in enumerate(cxGrp.cxIds):
            maxCxId = max([cxId, maxCxId])
            # Inh Compartment
            if partition.isInhibitory:
                core.cxCfg[cxId].configure(
                    bias=0,
                    biasExp=0,
                    cxProfile=0,
                    vthProfile=0
                )
            # Soma
            elif ((cxId + 1) % 2) == 0:
                core.cxCfg[cxId].configure(
                    bias=cxGrp.biasMant[i],
                    biasExp=cxGrp.biasExp[i],
                    cxProfile=1,
                    vthProfile=1
                )
            # Dendrite
            elif ((cxId + 2) % 2) == 0:
                core.cxCfg[cxId].configure(
                    bias=cxGrp.biasMant[i],
                    biasExp=cxGrp.biasExp[i],
                    cxProfile=0,
                    vthProfile=0
                )
            else:
                assert False, "Invalid Compartment"

        # Configure number of compartments to update
        numCxGroups = int(math.ceil((maxCxId + 1) / 4))
        core.numUpdates[0].configure(numUpdates=numCxGroups)

        # Configure compartments to start out in IDLE phase
        somaOp = 3 if sharedCfgs['enableSomaTrace'] else 0
        for i in range(numCxGroups):
            core.cxMetaState[i].configure(
                phase0=2, phase1=2, phase2=2, phase3=2,
                somaOp0=somaOp, somaOp1=somaOp, somaOp2=somaOp, somaOp3=somaOp)

        # Configure dendritic accumulators
        core.dendriteAccumCfg.delayBits = connectionCfgs['numDelayBits']

        if partition.isInhibitory:
            # Configure Inter-neuron compartment
            decayU = sharedCfgs["compartmentCurrentDecayInh"]
            decayV = sharedCfgs["compartmentVoltageDecayInh"]
            vThMant = sharedCfgs['vThMantInh']
            refractoryDelay = sharedCfgs["refractoryDelayInh"]
            core.cxProfileCfg[0].configure(
                decayU=decayU if decayU < 4096 else 4095,
                decayV=decayV if decayV < 4096 else 4095,
                refractDelay=refractoryDelay,
                bapAction=0,
                stackIn=0, stackOut=0
            )
            core.vthProfileCfg[0].staticCfg.configure(
                vth=vThMant)
        else:
            # Configure dendrite compartment
            decayU = sharedCfgs["compartmentCurrentDecayDendrite"]
            decayV = sharedCfgs["compartmentVoltageDecayDendrite"]
            vThMant = sharedCfgs['vThMantDendrite']
            refractoryDelay = sharedCfgs["refractoryDelayDendrite"]
            core.dendriteSharedCfg[0].configure(
                dsOffset=int(decayU == 4096),
                dmOffsets=int(decayV == 4096)
            )
            core.cxProfileCfg[0].configure(
                decayU=decayU if decayU < 4096 else 4095,
                decayV=decayV if decayV < 4096 else 4095,
                refractDelay=refractoryDelay,
                stackOut=1
            )
            core.vthProfileCfg[0].staticCfg.configure(
                vth=vThMant)

            # Configure soma compartment
            decayU = sharedCfgs["compartmentCurrentDecaySoma"]
            decayV = sharedCfgs["compartmentVoltageDecaySoma"]
            vThMant = sharedCfgs['vThMantSoma']
            refractoryDelay = sharedCfgs["refractoryDelaySoma"]
            core.cxProfileCfg[1].configure(
                decayU=decayU if decayU < 4096 else 4095,
                decayV=decayV,
                refractDelay=refractoryDelay,
                stackIn=2, joinOp=1,
                bapAction=sharedCfgs['bapAction']
            )

            core.vthProfileCfg[1].staticCfg.configure(
                vth=vThMant)

    def _printInputAxons(self, core):
        """Print NodeSets related to input axons.

        :param N2ACore core: Core.
        """

        print("Input axons:")
        inputAxons = [inAxMap[:, 0] for inAxMap in self._inAxMap.values()
                      if inAxMap[0, 2] == core.id]
        if len(inputAxons):
            for inputAxon in inputAxons:
                for i in set(inputAxon):
                    sMap = core.synapseMap[i]
                    if sMap.axonType == 2:
                        print("{}    {}".format(
                            i, sMap.population32MapEntry), end="")
                        print(" synapsePtr={}, synapseLen={}, popSize={}"
                              "".format(sMap.synapsePtr, sMap.synapseLen,
                                        sMap.popSize))
                    else:
                        print("{}    {}".format(
                            i, sMap.discreteMapEntry), end="")
                        print(" synapsePtr={}, synapseLen={}".format(
                            sMap.synapsePtr, sMap.synapseLen))

    @staticmethod
    def _printSynapses(core):
        """Print NodeSets related to synapses.

        :param N2ACore core: Core.
        """

        print("Synapse formats:")
        nodes = list(core.synapseFmt.modified)
        nodes.sort()
        for i in nodes:
            print("{}    {}".format(i, core.synapseFmt[i]))

        print("Synapses:")
        for i in range(core.synapses.numNodes):
            print("{}    {}".format(i, core.synapses[i]))

    def _printCompartments(self, core):
        """Print NodeSets related to compartments.

        :param N2ACore core: Core.
        """

        print("Compartments:")
        print("  CxCfg:")
        nodes = list(core.cxCfg.modified)
        nodes.sort()
        for i in nodes:
            print("{}    {}".format(i, core.cxCfg[i]))

        print("  CxMetaState:")
        nodes = list(core.cxMetaState.modified)
        nodes.sort()
        for i in nodes:
            print("{}    {}".format(i, core.cxMetaState[i]))

        print("  NumUpdates:")
        nodes = list(core.numUpdates.modified)
        nodes.sort()
        for i in nodes:
            print("{}    {}".format(i, core.numUpdates[i]))

        print("  DendriteSharedCfg:")
        nodes = list(core.dendriteSharedCfg.modified)
        nodes.sort()
        for i in nodes:
            print("{}    {}".format(i, core.dendriteSharedCfg[i]))

        print("  CxProfileCfg:")
        nodes = list(core.cxProfileCfg.modified)
        nodes.sort()
        for i in nodes:
            print("{}    {}".format(i, core.cxProfileCfg[i]))

        print("  VthProfileCfg:")
        nodes = list(core.vthProfileCfg.modified)
        nodes.sort()
        for i in nodes:
            if self._isSomaTraceEnabled:
                print("{}    {}".format(i, core.vthProfileCfg[i].dynamicCfg))
            else:
                print("{}    {}".format(i, core.vthProfileCfg[i].staticCfg))

        if self._isSomaTraceEnabled:
            print("  SomeState:")
            nodes = list(core.somaState.modified)
            nodes.sort()
            for i in nodes:
                print("{}    {}".format(i, core.somaState[i]))

            print("  SomaTraceCfg:")
            nodes = list(core.somaTraceCfg.modified)
            nodes.sort()
            for i in nodes:
                print("{}    {}".format(i, core.somaTraceCfg[i]))

            print("  DendriteTimeState:")
            nodes = list(core.dendriteTimeState.modified)
            nodes.sort()
            for i in nodes:
                print("{}    {}".format(i, core.dendriteTimeState[i]))

    @staticmethod
    def _printOutputAxons(core):
        """Print NodeSets related to output axons.

        :param N2ACore core: Core.
        """

        print("Output axons:")
        for i, a in enumerate(core.axons):
            print("{}    {}".format(i, a))

    def printCore(self, core, **kwargs):
        """Print NodeSets in a core.

        :param N2ACore core: Core.
        :key bool inputAxons: Prints NodeSets related to input axons.
        :key bool synapses: Prints NodeSets related to synapses.
        :key bool compartments: Prints NodeSets related to compartments.
        :key bool outputAxons: Prints NodeSets related to output axons.
        """

        if "inputAxons" in kwargs and kwargs["inputAxons"] is True:
            self._printInputAxons(core)

        if "synFmts" in kwargs and kwargs["synFmts"] is True:
            self._printSynapses(core)

        if "synapses" in kwargs and kwargs["synapses"] is True:
            self._printSynapses(core)

        if "compartments" in kwargs and kwargs["compartments"] is True:
            self._printCompartments(core)

        if "outputAxons" in kwargs and kwargs["outputAxons"] is True:
            self._printOutputAxons(core)

    def map(self, layer):
        """Map a partitioned layer to the NxCore interface.

        :param Layer layer: Mappable layer object that describes how the DNN is
            partitioned across neuro cores.
        """

        # Map partition to core.
        for partition in layer.partitions:
            # Allocate a new core for partition
            core = self._board.allocateCores(1, partition.numSyn)[0]

            # Skip synapses / input axons for input layers.
            if layer.numInputAxons > 0:
                # Skip synapses / input axons for complex input partitions
                if len(partition.synapseGroups) > 0:
                    self._mapSynapses(partition, core)
                    self._mapInputAxons(partition, core)
            # Skip output axons for output layer.
            if layer.numOutputAxons > 0:
                self._mapOutputAxons(partition, core)
            if 'Complex' in layer.type:
                self._mapComplexCompartments(partition, core)
            else:
                self._mapCompartments(partition, core)

            partition.chipId = core.parent.id
            partition.coreId = core.id

        layer.setMapped()
