###############################################################
# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2018-2021 Intel Corporation.

# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.

# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
###############################################################
from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import random
import warnings

import nxsdk.api.enums.api_enums as enums
import nxsdk.api.n2a as nx
from nxsdk.graph.processes.phase_enums import Phase


class Probable(Sequence):
    """Entity that supports probing and retrieving u,v,s state variables more intuitively than NxSDK."""

    def __init__(self):
        self._sprobe = None
        self._vprobe = None
        self._uprobe = None
        self._probable = None
        self._population = None
        super().__init__()

    def __getitem__(self, item):
        return self._population[item]

    def __len__(self):
        return len(self._population)

    @property
    def vprobe(self):
        """Container for probed voltage data."""
        return self._vprobe.data

    @property
    def sprobe(self):
        """Container for probed spikes data."""
        return self._sprobe.data

    @property
    def uprobe(self):
        """Container for probed current data."""
        return self._uprobe.data

    def probe(self, *params, t_start=1, index=None):
        """Setup one or several probes, allowed probes are 's', 'v' and 'u'.

        :param params: sequence of strings from 's', 'v' and 'u'.
        :param t_start: timestep when the probe will start recording.
        :param index: compartment index from which to probe.

        :return: probed data as single NxNet probe object containing some or all from spikes, current and voltage
        records.
        :rtype: NxNet probe obj
        """
        probe_dict = dict(v=(nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.IntervalProbeCondition(tStart=t_start)),
                          s=(nx.ProbeParameter.SPIKE, nx.SpikeProbeCondition(tStart=t_start)),
                          u=(nx.ProbeParameter.COMPARTMENT_CURRENT, nx.IntervalProbeCondition(tStart=t_start)))
        probable = self._probable[index] if index is not None else self._probable
        probe = None
        for p in list(params):
            assert p in ['v', 's', 'u'], "Only s, v or u can be probed!"
            probe = probable.probe(*probe_dict[p])[0]
            setattr(self, '_' + p + 'probe', probe)
        if len(list(params)) == 1:
            return probe

    def plot_probe(self, *params, figsize=(15, 8), xspacing=None, yspacing=None, subplot=None, fontsize=20,
                   ylabel=None, linestyle='-', index=None, **kwargs):
        """
        :param params: parameters to plot, allowed values are 's', 'v', 'u'
        :param figsize: tuple specifying the size of the figure
        :param subplot: subplot where to plot.
        :param index: element index from which to plot.
        """
        for p in params:
            probe = getattr(self, p + 'probe')[index] if index is not None else getattr(self, p + 'probe')
            condition = type(probe) is list or probe.ndim == 1
            plt.subplot(*subplot) if subplot else plt.figure(figsize=figsize)
            axs = plt.gca()
            if p == 's':
                data = np.where(np.asarray(probe) == 1) if condition else [np.asarray(row == 1).nonzero()[0] for row
                                                                           in probe]
                plt.eventplot(data,
                              orientation="horizontal",
                              label=p + 'probe' + (str(index) if index is not None else ''),
                              colors=kwargs['linecolor'] if 'linecolor' in kwargs else None,
                              lineoffsets=0 if condition else np.arange(len(probe))
                              )
                plt.xlim(0, len(probe) + 2 if condition else len(probe[0]) + 2)
            else:
                if condition:
                    xdata = range(len(probe))
                    plt.step(xdata, probe, label=p + 'probe' + (str(index) if index is not None else ''),
                             color=kwargs['linecolor'] if 'linecolor' in kwargs else None,
                             linestyle=linestyle)
                    plt.plot(xdata, probe, '.'  # , label=p + 'probe'
                             )
                    plt.xlim(0, len(probe) * 1.1)
                else:
                    xdata = range((len(probe[0])))
                    for idx, ydata in enumerate(probe):
                        plt.step(xdata, ydata, label=p + 'probe' + (str(index) if index is not None else ''),
                                 linestyle=linestyle,
                                 color=kwargs['linecolor'][idx] if 'linecolor' in kwargs else None,
                                 )
                    plt.xlim(0, len(probe[0]) * 1.1)
            if yspacing is not None:
                axs.yaxis.set_major_locator(ticker.MultipleLocator(yspacing))
            if xspacing is not None:
                axs.xaxis.set_major_locator(ticker.MultipleLocator(xspacing))
            plt.ylabel(ylabel if ylabel else p, fontsize=fontsize)
            plt.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.xlabel('Timestep', fontsize=fontsize)
        plt.legend()


class OptimizationSolver:
    """Base class from which to derive diverse neuromorphic optimization solvers."""

    def __init__(self,
                 problem,
                 x0=None,
                 nkwargs=None,
                 snnkwargs=None,
                 lmtkwargs=None,
                 *args, **kwargs):
        """
        :param problem: specification of the problem to be optimized.
        :param x0: vector of initial states of the problem variables.
        :param nkwargs: parameters to be passed to the neuron creator.
        :param snnkwargs: parameters to be passed to the SNN creator.
        :param lmtkwargs: parameters for LMT configuration, to be passed to the snips creator.
        """
        self.problem = problem
        self._x0 = x0
        self._nkwargs = nkwargs if nkwargs else {}
        self._snnkwargs = snnkwargs if snnkwargs else {}
        self._lmtkwargs = lmtkwargs if lmtkwargs else {}
        self._collected_solutions = []
        self.snn = None

    @property
    def x0(self):
        """Initial value for the state variables. """
        return self._x0

    @x0.setter
    def x0(self, val):
        self._x0 = val

    @property
    def collected_solutions(self):
        """Container to collect the solutions to the problem."""
        return self._collected_solutions

    def solve(self, seed=None, runtime=None, target_cost=None):
        """Run the SNN to solve the problem it encodes.

        :param seed: seed for python's RNG.
        :param runtime: maximum number of timesteps to search for a solution.
        :param target_cost: if an optimal solution is not needed or possible, target_costs set's the number of
        satisfying variables that cause the reporting neuron to spike.
        :return: Solution to the problem if one is found.
        """
        self.snn.set_cost(target_cost)
        solution = self.snn.run(seed=None, runtime=None)
        if self._check_solution(solution):
            self._collected_solutions.append(solution)
            return solution
        else:
            return None

    def _check_solution(self, solution):
        """Verify that a solution is consistent, complete and optimal."""
        pass

    def _build(self):
        """Creates the SNN that represents and solves the problem."""
        snn = None
        return snn

    def _randomize_v_init(self):
        """Set the initial voltage of the neurons across the SNN to different random values."""
        pass


class Qubo:
    def __init__(self, q_mtx):
        """A Quadratic Unconstrained Binary Optimization (QUBO) problem.

        :param q_mtx: squared Q matrix defining the QUBO problem over a binary vector x as: minimize x^T*Q*x.
        """
        self.q_mtx = q_mtx
        m, n = self.q_mtx.shape
        assert m == n, "QUBO matrix should be a square matrix."
        self.num_variables = m
        self.domain_size = 1

class Csp:
    def __init__(self, in_problem):
        """A constraint satisfaction problem represented by the tuple (variables, domains, constraints).

        constraints can be specified as a list of tuples with each tuple specifying two variables and a relation (X,
        Y, Relation). However, if the relation is the same for all constraints, then constraints can be specified as
        a tuple where the first element is the list of tuples specifying the pairs of variables affected by the
        relation and the second element is the relation that applies to all.
        A relation is a binary matrix where 1's indicate the allowed mutual values between the two variables in the
        binary constraint. Any non-bianary constraint can be expresed as a set of binary constraints, this is left to
        the user.

        :param in_problem: csp problem definition as a (variables, domains, constraints) tuple.
        """
        allowed_1 = type(in_problem[2]) is list and type(in_problem[2][0]) is tuple
        allowed_2 = type(in_problem[2]) is tuple and len(in_problem[2]) == 2
        assert allowed_1 or allowed_2, "Wrong specification for constraints."
        self._problem = in_problem

    @property
    def variables(self):
        return set(range(self._problem[0])) if isinstance(self._problem[0], int) else self._problem[0]

    @property
    def domains(self):
        return set(range(self._problem[1])) if isinstance(self._problem[1], int) else self._problem[1]

    @property
    def constraints(self):
        return self._problem[2]

    @property
    def num_variables(self):
        return len(self.variables)

    @property
    def domain_size(self):
        return self._problem[1]


class CspSolver(OptimizationSolver):
    """SNN constraint satisfaction solver, encode a CSP as an SNN and find a solution via stochastic dynamics.

    :param node_constraints:
    :param cost_variables:
    :param args:
    :param options:
    """

    def __init__(self,
                 node_constraints=None,
                 cost_variables=None,
                 *args,
                 **options):
        super().__init__(*args, **options)
        self.__channel_size = 100
        self.seeds_sequence = None
        self.online_currents = None
        self._node_constraints = node_constraints
        self.cost_variables = cost_variables
        self.csp = Csp(self.problem) if self.problem is not None else None
        self._board = None
        self.solving_times = None
        self.is_compiled = False
        self.lfsr_offset = 0
        self.snn = None

    @property
    def node_constraints(self):
        return self._node_constraints

    @node_constraints.setter
    def node_constraints(self, val):
        self._node_constraints = val
        self._build()

    @property
    def _channel_size(self):
        return self.__channel_size

    @_channel_size.setter
    def _channel_size(self, value):
        self.__channel_size = value

    def _build(self):
        """Build the actual SNN.
        """
        self.snn = SnnBuilder(self.csp.num_variables, self.csp.domain_size,
                              constraints=self.csp.constraints,
                              node_constraints=self.node_constraints,
                              mckwargs=self._nkwargs,
                              **self._snnkwargs)

    def _check_solution(self, solution):
        violations = []
        var_pairs = None
        if isinstance(self.csp, Qubo):
            return True
        else:
            tt = type(self.csp.constraints)
            if tt is tuple:
                var_pairs = self.csp.constraints[0]
                relation = self.csp.constraints[1]
                for x, y in var_pairs:
                    if relation[solution[x], solution[y]] == 0:
                        violations.append((x, y))
            elif tt is list:
                for x, y, r in self.csp.constraints:
                    if r[solution[x], solution[y]] == 0:
                        violations.append((x, y))
            if len(violations) == 0:
                print("Solution is valid")
                return True
            else:
                return False

    def solve(self, seed=1, runtime=100, target_cost=None, vr_low=-20, vr_high=1, randomize_lfsr_seeds=False,
              randomize_vinit=True, partition=None, _do_snips_setup=True, set_random_initial_state=False,
              keep_going=False):
        self.is_compiled = False
        self._build()
        self.solving_times = []
        random.seed(seed)
        if _do_snips_setup:
            self._summation_lmt_axon = self.snn.integrator.connect_sumation_neuron_to_lmt(runtime)
        if target_cost is not None:
            self.snn.set_cost(target_cost)
        if not self.is_compiled:
            self._board = self.snn.main_net.compiler.compile(self.snn.main_net)
            self.is_compiled = True
        # Randomise initial voltage for all base compartments on is_multicompartment neurons
        if randomize_vinit:
            self.snn.randomize_v(vr_low, vr_high, self._board)
        else:
            _overwrite_c_snip(lfsr_offset=0)
        if set_random_initial_state:
            self._sequential_initial_voltage(self._board)
        if _do_snips_setup:
            self._setup_snips(self._board)
            self._create_notification_channels(self._board)
        compartments3_ids = []
        # Get hardware mapping of top compartments
        for compartment in self.snn.principal_population_c3:
            (boardId, chipId, coreId, cxId, cxProfileCfgId,
             vthProfileCfgId) = self.snn.main_net.resourceMap.compartment(compartment.nodeId)
            compartments3_ids.append((chipId, coreId, cxId))
        solving_times, extracted_state = self.run(runtime=runtime, partition=partition,
                                                  compartments3_ids=compartments3_ids, keep_going=keep_going,
                                                  _do_snips_setup=_do_snips_setup, hack_lfsr=randomize_lfsr_seeds)
        return extracted_state

    def randomize_lfsr_seeds(self, board, _hack_lfsr=False, partition=None):
        # Randomise LFSR seed across all cores in board
        self.seeds_sequence = []
        for chip in board.n2Chips:
            for core in chip.n2CoresAsList:
                for dendrite in core.dendriteRandom:
                    dendrite.word = _draw_random_seed()
                    self.seeds_sequence.append(dendrite.word)
        if _hack_lfsr:
            self._hack_lfsr_func(partition=partition)
        pass

    def run(self, runtime, _do_snips_setup=True, partition=None, keep_going=False,
            hack_lfsr=False, compartments3_ids=None):
        """Run simulation for the number of timesteps given by runtime.

        Handles random seeding, random voltage and on-chip vs off-chip validation.
        """
        extracted_state = None
        if _do_snips_setup:
            solving_times, extracted_state = self.run_async_mode(runtime, partition, hack_lfsr=hack_lfsr,
                                                                 keep_going=keep_going,
                                                                 compartments3_ids=compartments3_ids)
            return solving_times, extracted_state
        else:
            self._board.run(runtime, partition=partition)
            return

    def run_async_mode(self, runtime, partition, hack_lfsr=False, compartments3_ids=None, keep_going=False):
        extracted_state = None
        self._channel_size = runtime
        # Run simulation in async mode
        print("\nsimulation starts in async mode \n")
        # Keep listening to channel for notification of solution (t_sol) or runtime exhaustion (-1)
        self._board.run(runtime, aSync=True, partition=partition)
        solving_times = self._notification_channel.read(1)[0]
        self._board.pause()
        print("\nChannel was read by superhost \n");
        # notify user whether a solution was found
        if solving_times == -1:
            print('\nNo solution was found during runtime \n')
            self.extract_net_state_from_loihi(compartments3_ids)
            extracted_state = self._online_read_delayed_state()
            self._board.finishRun()
            self._board.disconnect()
            print("disconnected from Loihi")
        else:
            if hack_lfsr:
                solving_times -= self.lfsr_offset  # this is the offset from hacking the LFSR before the actual run
            print('\n Entering loop to read soma traces at timestep %d\n' % solving_times)
            # read solution and save to self.online_solution
            self.extract_net_state_from_loihi(compartments3_ids)
            extracted_state = self._online_read_delayed_state()
            while (keep_going or not self._check_solution(extracted_state)) and solving_times != -1 and \
                    solving_times < \
                    runtime:
                warnings.warn("Solver reported a false positive!")
                self._acknowledgement_channel.write(1, [0])
                self._acknowledgement_channel.write(1, [0])
                self._notification_channel.read(1)[0]
                print("\n superhost wrote acknowledgement to channel to continue run [Inner]. \n")
                self._board.run(runtime - solving_times, aSync=True, partition=partition)
                solving_times = self._notification_channel.read(1)[0]
                self._board.pause()
                print("\n Channel was read by superhost [inner] \n", solving_times)
                self.extract_net_state_from_loihi(compartments3_ids)
                extracted_state = self._online_read_delayed_state()
                if keep_going:
                    print(extracted_state)
                print(extracted_state)
            print("\nSuperhost will write acknowledgement to channel to finish run [outer].\n")
            self._acknowledgement_channel.write(1, [1])
            if not self._check_solution(extracted_state):
                print("No Solution found during runtime.")
            else:
                print("Problem Solved During Runtime")
            print("will disconnect...")
            self._board.disconnect()
            print("disconnected from Loihi")
            self._collected_solutions.append(extracted_state)
            self.solving_times.append(solving_times)
        return solving_times, extracted_state

    def extract_net_state_from_loihi(self, compartments3_ids):
        """Extract network state from Loihi, encodes the solution if CSP was solved, last network state otherwise."""
        online_currents = []
        self._board.sync = True
        for chipId, coreId, cxId in compartments3_ids:
            self._board.n2Chips[chipId].n2Cores[coreId].cxState[cxId].fetch()
            online_currents.append(self._board.n2Chips[chipId].n2Cores[coreId].cxState[cxId].u)
        print("\nSuperhost read compartments current from SNN\n")
        self.online_currents = np.asarray(online_currents)

    def _create_notification_channels(self, board):
        """Create channels for communication between LMT and superhost.

        Args:
            board: board object from compiler.
        """
        # Create channel for detecting spike in lmt comming from summation neuron
        notification_channel = board.createChannel(b'nxsummlmt', "int", numElements=self._channel_size, slack=1
                                                   )
        # connect channel from lmt to superhost which is receiving spike count from lmt register
        notification_channel.connect(self.management_snip, None)

        # Create channel for acknowledging superhost reception of spike time
        acknowledgement_channel = board.createChannel(b'nxstacknow', "int", numElements=self._channel_size, slack=1
                                                      )
        # create notification_channel from lmt to superhost which is receiving spike count from lmt register
        acknowledgement_channel.connect(None, self.management_snip)

        self._notification_channel = notification_channel
        self._acknowledgement_channel = acknowledgement_channel

    def _online_read_delayed_state(self):
        """Extract delayed network state from the value of soma traces, refractory counters and phases.
        """
        # initialize array to the states_per_variable number, it exceeds the indexes by 1, encoding non-defined state.
        state = np.full(self.csp.num_variables, -1)
        currents = self.online_currents
        currents = currents.reshape(self.csp.num_variables, self.csp.domain_size)
        state = currents.argmax(1)
        state[~currents.any(1)] = -1
        self.online_solution = state.astype(int)
        return self.online_solution

    @property
    def sprobes(self):
        sprobes = []
        self.snn.principal_population.probe()
        for domain in self.snn.variables.values():
            for compartment in domain:
                sprobes.append(compartment.sprobe)
        return np.asarray(sprobes)

    @property
    def vprobes(self):
        vprobes = []
        for domain in self.snn.variables.values():
            for compartment in domain:
                vprobes.append(compartment.vprobe)
        return np.asarray(vprobes)

    @property
    def szprobes(self):
        szprobes = []
        for domain in self.snn.variables.values():
            for compartment in domain:
                szprobes.append(compartment.szprobe)
        return np.asarray(szprobes)

    def probe_net(self, *pvars):
        for domain in self.snn.variables.values():
            for compartment in domain:
                compartment.probe(*pvars)
        vars_integrator = (var for var in pvars if var in ['v', 's', 'u'])
        self.snn.integrator.summation_neuron.probe(*vars_integrator)

    def _random_initial_voltage(self, board, interval):
        """ Generate a 24-bit binary random voltage between rest and threshold voltages.

        Args:
            board:  board object obtained from the compiler.
            interval: 2-tuple giving the interval from which random initial voltage will be drawn, values are
                        implicitly multiplied by 2**6, e.g., interval=(0, vThMant) will result in (0*2**6,vThMant*2**6).
        """
        vThMant = self.snn.principal_population[0].vThMant
        # create array of random initial voltages for the whole board
        b_1_tot = self.snn.principal_population[0].b_1_tot
        self.v_initial = random.sample(range(int(vThMant * interval[0] * 2 ** 6) - b_1_tot,
                                             int(vThMant * interval[1] * 2 ** 6) - b_1_tot),
                                       # avoid neurons spiking at first timestep as this may cause synchronization
                                       self.snn.size)

        # get ids for base principal compartments
        compartment_ids = []
        for compartment in self.snn.principal_population:
            (boardId, chipId, coreId, cxId, cxProfileCfgId, vthProfileCfgId) = \
                self.snn.main_net.resourceMap.compartment(compartment.nodeId)
            compartment_ids.append((chipId, coreId, cxId))
        # set v_initial for all base neurons
        count = 0
        for chipId, coreId, cxId in compartment_ids:
            board.n2Chips[chipId].n2Cores[coreId].cxState[cxId].v = self.v_initial[count]
            count += 1
        return

    def _setup_snips(self, board):
        """Setup snips to notify solution, stop simulation and read network state.

        Args:
            board: board object from compiler.
        """
        # Define the process which sends information from lmt counter register to superhost about summation spikes.
        # This snip will run on chip 0 lmt 0
        runSpikingProcess = board.createSnip(name="spikingProcess",
                                             cFilePath=os.path.dirname(os.path.realpath(__file__)) + "/spiking.c",
                                             includeDir=os.path.dirname(os.path.realpath(__file__)),
                                             funcName="run_spiking",
                                             guardName="do_spiking",
                                             phase=Phase.EMBEDDED_SPIKING,
                                             chipId=0,
                                             lmtId=0)
        self.management_snip = runSpikingProcess
        self._create_notification_channels(board)

    @property
    def _num_principal_cores(self):
        return int(np.ceil(self.snn.size * 4 / self.snn.compartments_per_core))

    def _hack_lfsr_func(self, partition):
        self._lfsr_initialization_random_times = np.random.randint(1, 1000, self._num_principal_cores)
        self.lfsr_offset = np.asarray(self._lfsr_initialization_random_times).sum()
        _overwrite_c_snip(lfsr_offset=self.lfsr_offset)
        change_bias = True
        change_decays = True
        # Get ids for compartments
        board = self._board
        net = self.snn.main_net
        population = self.snn.principal_population
        pre_time = 0
        compartment_ids = []
        p_cores_ids = []
        for compartment in population:
            (boardId, chipId, coreId, cxId, cxProfileCfgId, vthProfileCfgId) = \
                net.resourceMap.compartment(compartment.nodeId)
            compartment_ids.append((chipId, coreId, cxId))
            p_cores_ids.append(coreId)
        p_cores_ids = set(p_cores_ids)

        summation_ids = []
        iterable = [self.snn.integrator.summation_neuron] if self.snn.integrator.summation_neuron.numNodes == 1 else \
            self.snn.integrator.summation_neuron
        for compartment in iterable:
            (boardId, chipId, coreId, cxId, cxProfileCfgId, vthProfileCfgId) = \
                net.resourceMap.compartment(compartment.nodeId)
            summation_ids.append(coreId)

        # a core can not have two types of neurons.
        assert len(set(summation_ids) & set(compartment_ids)) == 0

        # summation, hades and principal cores should account for all the only cores
        assert len(set(summation_ids) | set(p_cores_ids)) == sum([chip.numCores for chip in
                                                                  board.n2Chips])

        # Store numUpdates, bias and decayV for all compartmentas and cores
        #  Iterate over all compartments and store bias and decayV
        num_updates, bias, bias_exp, decay_v, decay_u = dict(), dict(), dict(), dict(), dict()
        if change_bias:
            for chipId, coreId, cxId in compartment_ids:
                # num_updates.append(board.n2Chips[chipId].n2Cores[coreId].numUpdates.numUpdates)
                bias[(chipId, coreId, cxId)] = board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].bias
                bias_exp[(chipId, coreId, cxId)] = board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].biasExp
                board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].bias = 1
                board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].biasExp = 0
                assert board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].bias == 1
                assert board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].biasExp == 0

        # Iterate over all cores, store numUpdates, and set numUpdates to 0
        for chipId, chip in enumerate(board.n2Chips):
            for coreId, core in enumerate(chip.n2CoresAsList):
                if coreId not in summation_ids:
                    num_updates[(chipId, coreId)] = core.numUpdates.numUpdates
                    if change_decays:
                        for cc in range(32):
                            decay_v[(chipId, coreId, cc)] = core.cxProfileCfg[cc].decayV
                            decay_u[(chipId, coreId, cc)] = core.cxProfileCfg[cc].decayU
                            core.cxProfileCfg[cc].configure(decayV=4095, decayU=4095)
                            assert core.cxProfileCfg[cc].decayV == 4095
                            assert core.cxProfileCfg[cc].decayU == 4095
                    core.numUpdates.numUpdates = 0
                    assert core.numUpdates.numUpdates == 0

        #  Iterate over each core and randomly advance each core's LFSR
        count = 0
        for chip in board.n2Chips:
            for coreId, core in enumerate(chip.n2CoresAsList):
                if coreId not in summation_ids:
                    print("\n Advancing LFSR for core:", core.id)
                    core.numUpdates.numUpdates = 1  # TODO: only 4 compartments need to be backed up
                    assert core.numUpdates.numUpdates == 1
                    rand_time = self._lfsr_initialization_random_times[count]
                    count += 1
                    board.run(rand_time, aSync=True, partition=partition)
                    board.sync = True
                    core.numUpdates.numUpdates = 0
                    assert core.numUpdates.numUpdates == 0
                    pre_time += rand_time

        # Restore numUpdates
        for chipId, chip in enumerate(board.n2Chips):
            for coreId, core in enumerate(chip.n2CoresAsList):
                if coreId not in summation_ids:
                    core.numUpdates.numUpdates = num_updates[(chipId, coreId)]
                    assert core.numUpdates.numUpdates != 0

        # Restore decays
        if change_decays:
            for chipId, chip in enumerate(board.n2Chips):
                for coreId, core in enumerate(chip.n2CoresAsList):
                    if coreId not in summation_ids:
                        for cc in range(32):
                            core.cxProfileCfg[cc].configure(decayV=decay_v[(chipId, coreId, cc)], decayU=decay_u[(
                                chipId, coreId, cc)])
                    else:
                        print("DEBUG, Ignoring", core.id)

        # Restore biases
        if change_bias:
            for chipId, coreId, cxId in compartment_ids:
                board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].bias = bias[(chipId, coreId, cxId)]
                board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].biasExp = bias_exp[(chipId, coreId, cxId)]
                assert board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].bias != 1
                assert board.n2Chips[chipId].n2Cores[coreId].cxCfg[cxId].biasExp != 0
        board.pause()
        return pre_time

    def _sequential_initial_voltage(self, board):
        """ Set initial voltage to force an initial random valid CSP state.

        Args:
            board:  board object obtained from the compiler.
        """
        # if self.v_initial is None:
        v_Th = self.snn.principal_population[0].vThMant * 2 ** 6
        # create array of random initial voltages for the whole board
        mask = np.zeros((self.csp.num_variables, self.csp.domain_size), dtype=int)
        mask[:, 0] = 1
        for i in range(2):
            for row in mask:
                np.random.shuffle(row)
        mask[np.where(mask == 1)] = v_Th - np.random.randint(v_Th // 2, v_Th, self.csp.num_variables)
        self.v_initial = mask.ravel()
        # get ids for base principal compartments
        compartment_ids = []
        for compartment in self.snn.principal_population:
            (boardId, chipId, coreId, cxId, cxProfileCfgId, vthProfileCfgId) = \
                self.snn.main_net.resourceMap.compartment(compartment.nodeId)
            compartment_ids.append((chipId, coreId, cxId))
        # set v_initial for all base neurons
        count = 0
        for chipId, coreId, cxId in compartment_ids:
            board.n2Chips[chipId].n2Cores[coreId].cxState[cxId].v = self.v_initial[count]
            count += 1
        return


class QuboSolver(CspSolver):
    def __init__(self,
                 q,
                 q_weight_scaling=1,
                 *args,
                 **options
                 ):
        self.q = q
        self._q_weight_scaling = q_weight_scaling
        self.qubo = Qubo(q)
        self.biases = np.repeat(-self.qubo.q_mtx.diagonal(), 1)
        self.constraints = None
        super().__init__(problem=None, *args, **options)
        if 'b' in self._nkwargs.keys():
            warnings.warn("Updating bias according to QUBO matrix")
        self._node_constraints = None
        self.cost_variables = None
        self.csp = self.qubo
        self._board = None
        self.solving_times = None
        self.is_compiled = False
        self.lfsr_offset = 0
        self.snn = None

    @property
    def q_weight_scaling(self):
        return self._q_weight_scaling

    def _build(self):
        """Build the actual SNN.
        """
        self.snn = SnnBuilder(self.qubo.num_variables,
                              self.qubo.domain_size, constraints=None, q_mtx=self.qubo.q_mtx,
                              q_weight_scaling=self.q_weight_scaling,
                              mckwargs=self._nkwargs,
                              **self._snnkwargs)
        for idx, neuron in enumerate(self.snn.principal_population):
            neuron.biasMant = self.biases[idx]


class SnnBuilder:
    def __init__(self,
                 num_vars, dom_size,
                 is_multicompartment=True,
                 wta_type='lateral',
                 self_excitation=False,
                 w_wta_exc=None,
                 w_wta_inh=None,
                 w_self_exc=None,
                 constraints=None,
                 q_mtx=None,
                 w_exp=None,
                 box_duration=None,
                 mckwargs=None,
                 node_constraints=None,
                 do_snips_setup=True,
                 cost_variables=None,
                 sigmaargs=None,
                 compartments_per_core=1024,
                 w_constraints_inh=None,
                 q_weight_scaling=None,
                 multicompartments_per_summation_neuron=None
                 ):
        self.multicompartments_per_summation_neuron = multicompartments_per_summation_neuron if \
            multicompartments_per_summation_neuron is not None else (2048 if num_vars > 16**2 else 4096)
        self.q_weight_scaling = q_weight_scaling
        self.principal_adjacency_mtx = None
        self.v_initial = None
        self._w_constraints_inh = w_constraints_inh
        self.compartments_per_core = compartments_per_core
        self.sigmaargs = sigmaargs if sigmaargs else {}
        self.node_constraints = node_constraints
        self.cost_variables = cost_variables
        self.num_vars = num_vars
        self.dom_size = dom_size
        self.size = num_vars * dom_size
        self.mckwargs = mckwargs if mckwargs else {}
        self.is_multicompartment = is_multicompartment
        self._box_duration = box_duration
        self._w_wta_exc = w_wta_exc
        self._w_wta_inh = w_wta_inh
        self.constraints = constraints
        self.q_mtx = q_mtx
        self._w_self_exc = w_self_exc
        self.self_excitation = self_excitation
        self.wta_type = wta_type
        self._w_exp = w_exp
        self.main_net = nx.NxNet()
        self._multicompartment_pop = []
        self._setup(do_snips_setup)

    def _setup(self, _do_snips_setup):
        # Create neurons
        self._create_principal_population()
        # Setup connectivity
        self._apply_principal_network_connections()
        self._setup_online_validator(_do_snips_setup)

    @property
    def box_duration(self):
        """Set duration of box post-synaptic potential.

        :return: duration of box post-synaptic potential.
        :rtype: int
        """
        if self._box_duration is None:
            return 6
        else:
            return self._box_duration

    @box_duration.setter
    def box_duration(self, value):
        """Allow frontend definition of box_duration.

        :param int value: value for the duration of rectangular postsynaptic response.
        """
        self._box_duration = int(value)

    @box_duration.deleter
    def box_duration(self):
        """Delete box_duration parameter."""
        del self._box_duration

    @property
    def _noise_amplitude(self):
        """Compute the amplitude of compartments noise as the difference between max and min values.

        :return: noise amplitude.
        :rtype: int
        """
        if 'enable_noise' in self.mckwargs.keys():
            return 255 * 2 ** (-7. + self.mckwargs['noise_at_multicompartment']['noiseExpAtCompartment'])
        else:
            return 0

    @property
    def w_wta_exc(self):
        """Get internal excitatory weight for WTAs.

        :return: WTAs excitatory weight.
        :rtype: int
        """
        w = 0 if self._w_wta_exc is None else self._w_wta_exc
        # assert abs(w) > 2*self._noise_amplitude/2**(6+self.w_ij_exp) or w==0
        return abs(w)

    @w_wta_exc.setter
    def w_wta_exc(self, value):
        """Set value of excitatory weight for WTAs. The value will be forced to be positive.

        :param int value: WTAs excitatory weight.
        """
        self._w_wta_exc = abs(value)

    @property
    def w_wta_inh(self):
        """Get internal inhibitory weight for WTAs.

        :return: WTAs inhibitory weight.
        :rtype: int
        """
        w = -100 if self._w_wta_inh is None else self._w_wta_inh
        assert abs(w) > 2 * self._noise_amplitude / 2 ** (6 + self.w_ij_exp) or w == 0
        return -abs(w)

    @w_wta_inh.setter
    def w_wta_inh(self, value):
        """Set value of inhibitory weight for WTAs. The value will be forced to be negative.

        :param int value: WTAs inhibitory weight.
        """
        assert abs(value) > 2 * self._noise_amplitude / 2 ** (6 + self.w_ij_exp) or value == 0
        self._w_wta_inh = -abs(value)

    @property
    def w_constraints_inh(self):
        """Get inhibitory weight for implementing constraints/competition between WTAs.

        :return: inhibitory weight for all-different constraints between WTAs.
        :rtype: int
        """
        w = -100 if self._w_constraints_inh is None else self._w_constraints_inh
        assert abs(w) > 2 * self._noise_amplitude / 2 ** (6 + self.w_ij_exp) or w == 0
        return -abs(w)

    @w_constraints_inh.setter
    def w_constraints_inh(self, value):
        """Set value of inhibitory weight for implementing constraints/competition between WTAs.

        The value will be forced to be negative.

        :param int value: inhibitory weight for all-different constraints between WTAs.
        """
        assert abs(value) > 2 * self._noise_amplitude / 2 ** (6 + self.w_ij_exp) or value == 0
        self._w_constraints_inh = -abs(value)

    @property
    def w_self_exc(self):
        """Get weight for self-excitation in WTAs principal neurons.

        :return: self-excitation weight for principal neurons.
        :rtype: int
        """
        w = 0 if self._w_self_exc is None else self._w_self_exc
        return abs(w)

    @w_self_exc.setter
    def w_self_exc(self, value):
        """Set value of self-excitation weights in WTAs principal neurons. The value will be forced to be positive.

        :param int value: WTAs self-excitation weight.
        """
        self._w_self_exc = abs(value)

    @property
    def w_ij_exp(self):
        """Get weight exponent for inter-variable and inter-domain connections between is_multicompartment neurons.

        :return: weight exponent for inter-variable and inter-domain connections.
        :rtype: int
        """
        return self._w_exp if self._w_exp is not None else 6

    @w_ij_exp.setter
    def w_ij_exp(self, value):
        """Set weight exponent for inter-variable and inter-domain connections between is_multicompartment neurons.

        :param int value: weight exponent for inter-variable and inter-domain connections.
        """
        self._w_exp = value

    @property
    def w_max_inh(self):
        """Get weight mantisa for inter-variable and inter-domain connections between is_multicompartment neurons.

        :return: weight mantissa for for inter-variable and inter-domain connections. The value is possitive and the
        sign
        will be assigned by the relevant methods according to its use for inhibition or excitation.
        :rtype: int
        """
        return max([abs(i) for i in [self.w_wta_inh, self.w_constraints_inh]]) * 2 ** (6 + self.w_ij_exp)

    @property
    def w_min_inh(self):
        """Get weight mantisa for inter-variable and inter-domain connections between is_multicompartment neurons.

        :return: weight mantissa for for inter-variable and inter-domain connections. The value is possitive and the
        sign
        will be assigned by the relevant methods according to its use for inhibition or excitation.
        :rtype: int
        """
        return min([abs(i) for i in [self.w_wta_inh, self.w_constraints_inh]]) * 2 ** (6 + self.w_ij_exp)

    @property
    def _logical_core_id(self):
        """Progressively use next core as resources are exhausted.

        For on-chip evaluation the first core is used for summation and one_hot_enforcement neurons.

        :return: logical core ID for allocating the compartment being processed.
        :rtype: int
        """
        if self.is_multicompartment:
            return self.main_net.numCompartments // self.compartments_per_core + 1
        else:
            return self.main_net.numCompartments // self.compartments_per_core

    @property
    def _cx_prototype_map(self):
        pm = CspPrototypeMap(self.num_vars, self.dom_size, clamped_values=None, wta_type='lateral')
        return pm.prototype_map

    def _create_principal_population(self):
        """ Create group of single- or 4- compartment neurons to be segmented in variables, domains and cores.

         The used prototypes depend on the wta motif, this is taken care of by the principal_prototype_map.
         """
        if self.is_multicompartment:
            for neuron in range(self.size):
                mc = MultiCompartment(main_net=self.main_net, is_clamped=bool(self.principal_prototype_map[neuron]),
                                      w_min=self.w_min_inh, box_duration=self.box_duration,
                                      logical_core_id=self._logical_core_id,
                                      **self.mckwargs)
                self._multicompartment_pop.append(mc)

            # Create container groups for accessing compartments in multicompartments across whole network
            self.principal_population = self.main_net.createCompartmentGroup()
            self.principal_population_c1 = self.main_net.createCompartmentGroup()
            self.principal_population_c2 = self.main_net.createCompartmentGroup()
            self.principal_population_c3 = self.main_net.createCompartmentGroup()

            for idx, population in enumerate([self.principal_population, self.principal_population_c1,
                                              self.principal_population_c2, self.principal_population_c3]):
                population.addCompartments([multicompartment.compartment_group[idx] for multicompartment in
                                            self._multicompartment_pop])

    @property
    def inter_domain_prototype_inh(self):
        prototype = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                                           weight=self.w_wta_inh,
                                           weightExponent=self.w_ij_exp,
                                           delay=self.box_duration,
                                           postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX)
        return prototype

    @property
    def inter_variable_prototype(self):
        prototype = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                                           weight=self.w_constraints_inh,
                                           weightExponent=self.w_ij_exp,
                                           delay=self.box_duration,
                                           postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX)
        return prototype

    @property
    def self_excitation_prototype(self):
        prototype = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                           weight=self.w_self_exc,
                                           weightExponent=self.w_ij_exp,
                                           delay=self.box_duration,
                                           postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX
                                           )
        return prototype

    @property
    def principal_prototype_map(self):
        pmap = CspPrototypeMap(self.num_vars, self.dom_size,
                               clamped_values=self.node_constraints,
                               wta_type=self.wta_type)
        return pmap.prototype_map


    @property
    def _num_summation_neurons(self):
        """Get number of summation neurons to be used.

        It is larger than 1 when multiple summation neurons are necessary due to maximum fan-out/fan-in per core.

        :return: number of summation neurons to be used for on-chip validation.
        :rtype: int
        """
        if self.size > self.multicompartments_per_summation_neuron:
            num_summations=np.ceil(self.size / self.multicompartments_per_summation_neuron) + 1
            return num_summations.astype(int)
        else:
            return 1

    def _apply_principal_network_connections(self):
        """Use full connectivity matrix to create all synapses inside NxNet object."""
        dummy = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,  # todo delete
                                       weight=0,
                                       weightExponent=self.w_ij_exp)
        self.principal_adjacency_mtx = CspAdjacencyMatrix(self.num_vars, self.dom_size, self.constraints)
        mapping = self.principal_adjacency_mtx.adjacency_mtx
        prototype_list = [self.inter_domain_prototype_inh,
                          self.inter_variable_prototype]
        # 0, 1, 2, 3 map to wta, and constraints respectively
        wtas_mtx = (mapping == 1).astype(int)
        cons_mtx = (mapping == 3).astype(int)
        if self.constraints is not None:
            for mtx, proto in zip([wtas_mtx, cons_mtx], prototype_list):
                # Apply connections
                self.principal_population.connect(self.principal_population, prototype=proto, connectionMask=mtx)
        elif self.q_mtx is not None:
            qc = np.eye(1)
            q_mtx_no_diag = np.copy(self.q_mtx)
            np.fill_diagonal(q_mtx_no_diag, 0)
            q = np.kron(q_mtx_no_diag, qc)
            adjacency_weights = wtas_mtx * self.w_wta_inh  - q * self.q_weight_scaling
            prototype = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                                               delay=self.box_duration,
                                               postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX)
            # Apply  connections
            self.principal_population.connect(self.principal_population,
                                              prototype=prototype,
                                              weight=adjacency_weights,
                                              )
            self.principal_adjacency_mtx = adjacency_weights
        return

    def _setup_online_validator(self, _do_snips_setup):
        sigma_threshold = self.num_vars if self.cost_variables is None else len(self.cost_variables)
        self.integrator = SummationNeuron(sigma_threshold, _do_snips_setup,
                                          multicompartment_population=self._multicompartment_pop,
                                          multicompartments_per_summation_neuron=self.multicompartments_per_summation_neuron,
                                          size_of_principal_pop=self.size,
                                          states_per_variable=self.dom_size,
                                          _logical_core_id=self._logical_core_id,
                                          _num_summation_neurons=self._num_summation_neurons,
                                          **self.sigmaargs
                                          )

    def randomize_v(self, vr_low, vr_high, board):
        """ Generate a 24-bit binary random voltage between rest and threshold voltages.

        Args:
            :param board: board object obtained from the compiler.
            :param vr_low: low limit from which random initial voltage will be drawn, values are
                        implicitly multiplied by 2**6, e.g., interval=(0, vThMant) will result in (0*2**6,vThMant*2**6).
            :param vr_high: top limit from which random initial voltage will be drawn, values are
                        implicitly multiplied by 2**6, e.g., interval=(0, vThMant) will result in (0*2**6,vThMant*2**6).
        """
        vThMant = self.principal_population[0].vThMant
        # create array of random initial voltages for the whole board
        b_1_tot = self._multicompartment_pop[0].b_1_tot
        self.v_initial = random.sample(range(int(vThMant * vr_low * 2 ** 6) - b_1_tot,
                                             int(vThMant * vr_high * 2 ** 6) - b_1_tot),
                                       self.size)

        # get ids for base principal compartments
        compartment_ids = []
        for compartment in self.principal_population:
            (boardId, chipId, coreId, cxId, cxProfileCfgId, vthProfileCfgId) = \
                self.main_net.resourceMap.compartment(compartment.nodeId)
            compartment_ids.append((chipId, coreId, cxId))
        # set v_initial for all base neurons
        count = 0
        for chipId, coreId, cxId in compartment_ids:
            board.n2Chips[chipId].n2Cores[coreId].cxState[cxId].v = self.v_initial[count]
            count += 1
        return
        pass

    def set_cost(self, target_cost):
        # todo implement for multiple summation neurons too
        self.integrator.summation_neuron.vThMant = target_cost * 2 - 1


class CspPrototypeMap:
    """Create prototype maps to build the principal population of winner-take-alls.

    If the wta motif is 'lateral' all neurons use the same prototype, if it is 'aux' and additional prototype
    for the inhibitory interneuron needs to be created and the map assigns it to the appropriate neurons using
    their index.
    """

    def __init__(self, number_of_variables, states_per_variable, clamped_values=None, wta_type='lateral'):
        self.wta_type = wta_type
        self.number_of_variables = number_of_variables
        self.size_of_principal_pop = number_of_variables * states_per_variable
        self.states_per_variable = states_per_variable
        self.clamped_values = clamped_values
        self.prototype_map = None
        self._create_prototype_maps()
        if self.clamped_values:
            self._set_clamped_values()

    def _create_prototype_maps(self):
        """Create prototype maps to build the principal population of winner-take-alls.

        If the wta motif is 'lateral' all neurons use the same prototype, if it is 'aux' and additional prototype
        for the inhibitory interneuron needs to be created and the map assigns it to the appropriate neurons using
        their index.
        """
        if self.wta_type == 'lateral':
            proto = np.zeros(self.size_of_principal_pop, dtype=int)
        elif self.wta_type == 'aux':
            # map prototypes so that for every states_per_variable number of neurons an inhibitory auxiliary neuron
            # appears.
            proto = np.tile(np.repeat([0, 1], [self.states_per_variable, 1]), self.number_of_variables)
        else:
            raise ValueError("wta_type should be \'lateral\' or \'aux\'")
        self.prototype_map = proto.tolist()

    def _set_clamped_values(self):
        """Set the variables that have a pre-defined value to be active on that value.

        Here clamped_values are fixed and predetermined values for particular variables, These influence how the
        constraints are mapped to synapses.

        args:
            clamped_values: an array of the form [[list of variable ids],[list of values taken by those variables]]
        """
        for idx, val in self.clamped_values:
            if self.wta_type == 'lateral':
                # set all inactive
                self.prototype_map[idx * self.states_per_variable:(idx + 1) * self.states_per_variable] = \
                    [1] * self.states_per_variable
                # set clamped neuron to active
                if isinstance(val, list) or isinstance(val, tuple) or isinstance(val, set):
                    for v in val:
                        self.prototype_map[idx * self.states_per_variable + v] = 0
                else:
                    self.prototype_map[idx * self.states_per_variable + val] = 0
            elif self.wta_type == 'aux':
                doms = self.states_per_variable + 1
                # set all inactive
                self.prototype_map[idx * doms:(idx + 1) * doms - 1] = \
                    [2] * doms  # exclude first from next element + aux
                # set clamped active
                self.prototype_map[idx * doms + val] = 0


class CspAdjacencyMatrix:
    """Represent the connectivity of a neural CSP problem solver, encode WTAs and constraints."""

    def __init__(self, num_variables=1, domain_size=1, constraints=None):
        self.num_variables = num_variables
        self.domain_size = domain_size
        self.constraints = constraints
        self.size = self.num_variables * self.domain_size
        self.adjacency_mtx = None
        self._build()

    def _build(self):
        """Generate the actual adjacency matrix."""
        # Create WTA matrix
        diag = 2
        mtx = np.ones((self.domain_size, self.domain_size))
        np.fill_diagonal(mtx, diag)
        wta_mtx = np.kron(np.eye(self.num_variables), mtx)
        # Create constraints matrix and integrate it with the WTA matrix
        if self.constraints and len(self.constraints) > 0:
            if type(self.constraints) is tuple:
                m = np.zeros((self.num_variables, self.num_variables))
                # Apply a relation per constraints
                m[tuple(zip(*self.constraints[0]))] = 1
                cids = np.logical_or(m, m.T).astype(int)
                relation = np.logical_not(self.constraints[1]) * 3
                cs_mtx = np.kron(cids, relation)
                self.adjacency_mtx = wta_mtx + cs_mtx
            elif type(self.constraints) is list:
                # Apply a single relation for all constraints
                for i, j, r in self.constraints:
                    doms = self.domain_size
                    wta_mtx[i * doms:i * doms + doms, j * doms:j * doms + doms] = np.logical_not(r) * 3
                    i, j = j, i
                    wta_mtx[i * doms:i * doms + doms, j * doms:j * doms + doms] = np.logical_not(r) * 3
                self.adjacency_mtx = wta_mtx
        else:
            self.adjacency_mtx = wta_mtx


class SummationNeuron(Probable):  # Todo there are side effects with Logical core id so this has to be created last
    def __init__(self, sigma_threshold, _do_snips_setup=True, multicompartment_population=None,
                 _num_summation_neurons=1,
                 multicompartments_per_summation_neuron=4096, states_per_variable=1, size_of_principal_pop=1,
                 _logical_core_id=0, runtime=None):
        super().__init__()
        self.sigma_threshold = sigma_threshold
        self.main_net = multicompartment_population[0].compartment_group.net if multicompartment_population else \
            nx.NxNet()
        self.multicompartment_population = multicompartment_population
        self._num_summation_neurons = _num_summation_neurons
        self.multicompartments_per_summation_neuron = multicompartments_per_summation_neuron
        self.states_per_variable = states_per_variable
        self.size_of_principal_pop = size_of_principal_pop
        self._logical_core_id = _logical_core_id
        self.runtime = runtime
        # configure
        self._setup_summation_neuron()
        self._probable = self.summation_neuron
        self._population = self.summation_neuron

    def _setup_summation_neuron(self):
        """Create summation neuron and configure connections with principal neurons."""
        self._create_summation_neuron()
        if self.multicompartment_population:
            self._wire_sumation_neuron()
        return

    def _create_summation_neuron(self):
        """Create a summation neuron which integrates spikes from all is_multicompartment neurons and computes cost
        function.

        Set bias to max value to give maximum range to u and vth to this same value so that the neuron fires by default.
        """
        if self._num_summation_neurons == 1:
            prototype_summation = nx.CompartmentPrototype(biasMant=0,
                                                          biasExp=0,
                                                          vThMant=self.sigma_threshold * 2 - 1,
                                                          logicalCoreId=0,
                                                          enableNoise=0,
                                                          compartmentVoltageDecay=2 ** 12,
                                                          compartmentCurrentDecay=2 ** 12,
                                                          functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
            self.summation_neuron = self.main_net.createCompartment(prototype_summation)
        else:
            prototypes_summation = []
            for i in range(self._num_summation_neurons - 1):
                prototypes_summation.append(nx.CompartmentPrototype(biasMant=0,
                                                                    biasExp=0,
                                                                    vThMant=(self.multicompartments_per_summation_neuron // self.states_per_variable) * 2 - 1,
                                                                    logicalCoreId=self._logical_core_id + 1 + i,
                                                                    enableNoise=0,
                                                                    compartmentVoltageDecay=2 ** 12,
                                                                    compartmentCurrentDecay=2 ** 12,
                                                                    functionalState=
                                                                    nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE))
            if self.size_of_principal_pop % self.multicompartments_per_summation_neuron != 0:
                prototypes_summation[-1].vThMant = 2 * (self.size_of_principal_pop %
                                                        self.multicompartments_per_summation_neuron) // self.states_per_variable - 1
            prototype_last_summation = nx.CompartmentPrototype(biasMant=0,
                                                               biasExp=0,
                                                               vThMant=(self._num_summation_neurons-1) * 2 - 1,
                                                               logicalCoreId=0,
                                                               enableNoise=0,
                                                               compartmentVoltageDecay=2 ** 12,
                                                               compartmentCurrentDecay=2 ** 12,
                                                               functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
            prototypes_summation.append(prototype_last_summation)
            pmap = np.arange(len(prototypes_summation), dtype=int)
            self.summation_neuron = self.main_net.createCompartmentGroup(
                size=self._num_summation_neurons,
                prototype=prototypes_summation,
                prototypeMap=pmap)

    def _wire_sumation_neuron(self):
        """Integrate local satisfiability information from second and third compartments of is_multicompartment neuron.

        The w_mc_to_summation weight will make the u variable of summation neuron a proxy of the real energy function.
        However, there is a limit on representability given by the max value that u can take in discrete steps of
        wgtMant*2**(6-wgtExp), so if the number of neurons in the principal network is larger than the number of steps
        in u the mapping will not be accurate.

        The summation neuron also receives inhibitory input from the one-hot-enforcement neuron.
        """
        w_mc_to_summation = 2
        w_sum_to_integration = 2
        exc_to_sumation_prototype = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                           weight=w_mc_to_summation,
                                                           weightExponent=0
                                                           )

        summation_to_sumation_prototype = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                                 weight=w_sum_to_integration,
                                                                 weightExponent=0,
                                                                 )
        if self._num_summation_neurons == 1:
            for neuron in self.multicompartment_population:
                neuron.compartment_group[2].connect(self.summation_neuron,
                                                    prototype=exc_to_sumation_prototype)
            return
        else:
            for idx, neuron in enumerate(self.multicompartment_population):
                neuron.compartment_group[2].connect(
                    self.summation_neuron[idx // self.multicompartments_per_summation_neuron],
                    prototype=exc_to_sumation_prototype)
            for compartment in range(self._num_summation_neurons - 1):
                self.summation_neuron[compartment].connect(self.summation_neuron[-1],
                                                           prototype=summation_to_sumation_prototype)
            return

    def connect_sumation_neuron_to_lmt(self, runtime):
        """Setup axon from summation neuron to LMT but do not read from host.

        setting the probe creates an axon from summation to LMT but setting tStart large avoids reading from host, which
        will be done using a bespoken SNIP.
        """
        prob_condition = nx.SpikeProbeCondition(tStart=runtime * 2000000)
        if self._num_summation_neurons == 1:
            summation_probes = self.summation_neuron.probe(nx.ProbeParameter.SPIKE, prob_condition)
        else:
            summation_probes = self.summation_neuron[-1].probe(nx.ProbeParameter.SPIKE, prob_condition)
        return summation_probes

    def run(self, runtime=200):
        self.main_net.run(runtime)
        self.main_net.disconnect()


class MultiCompartment(Probable):
    def __init__(self, box_duration=6,
                 main_net=None,
                 noise_at_multicompartment=None,
                 enable_noise=True,
                 enable_activity_traces=False,
                 bias_to_fire=6,
                 randomized_seeds=False,
                 randomize_v_init=False,
                 interval_for_v_init_randomization=(-20., 1.),
                 v_min_exp=5,
                 v_th_1_mant=None,
                 readout_autapse_delay=2,
                 logical_core_id=-1,
                 is_clamped=False,
                 nkwargs=None,
                 w_min=None,
                 num_dendritic_accumulators=8,
                 num_delay_bits=3
                 ):
        super().__init__()
        assert bias_to_fire >= box_duration or bias_to_fire == -1
        self.main_net = main_net if main_net else nx.NxNet()
        self.w_min = w_min if w_min else -20 * 2 ** 9
        self.nkwargs = nkwargs
        self.box_duration = box_duration
        self._num_delay_bits = num_delay_bits
        self._num_dendritic_acumulators = num_dendritic_accumulators
        self._readout_autapse_delay = readout_autapse_delay
        self.bias_to_fire = bias_to_fire
        self._v_decay = 0
        self._u_decay = 0
        self._vMinExp = v_min_exp
        self._enable_noise = enable_noise
        self._noise_at_multicompartment = noise_at_multicompartment
        self._interval_for_v_init_randomization = interval_for_v_init_randomization
        self._vth_1_mant = v_th_1_mant
        self._randomized_seeds = randomized_seeds
        self._randomize_v = randomize_v_init
        self.enable_activity_traces = enable_activity_traces
        self._logical_core_id = logical_core_id
        self._is_clamped = is_clamped
        # Create actual multicompartment
        self.compartment_group = None
        self._build_compartment_group()
        self._probable = self.compartment_group
        self._population = self.compartment_group

    def _build_compartment_group(self):
        """Create actual multicompartment in the NxNet."""
        if self.enable_activity_traces:
            self._configure_activity_traces()
        if self._is_clamped:
            self.compartment_group = self._create_multicompartment_clamp_inactive()
        else:
            self.compartment_group = self._create_multicompartment_neuron()

    @property
    def enable_noise(self):
        """Whether noise is to be enabled or not.

        :rtype: bool
        """
        return self._enable_noise

    @enable_noise.setter
    def enable_noise(self, value):
        """Set value of enable_noise.

        :param bool value: whether noise is to be enable or not.
        """
        if not isinstance(value, bool):
            raise Exception("enable_noise should be True or False.")
        self._enable_noise = value

    @property
    def noise_at_multicompartment(self):
        """ Get noise parameters for is_multicompartment neuron. If user did not define them return default.

        :return: Dictionary of mantissa and exponent parameters for noise intensity in compartment voltage.
        :rtype: dict
        """
        if self._noise_at_multicompartment is not None:
            assert self._noise_at_multicompartment["mantissa"] == 0, \
                "In its current form the solver can not work with nonzero noise mantissa."

            return {'noiseMantAtCompartment': self._noise_at_multicompartment['mantissa'],
                    'noiseExpAtCompartment': self._noise_at_multicompartment['exponent']}
        elif self.enable_noise:
            return {'noiseMantAtCompartment': 0,
                    'noiseExpAtCompartment': 6}
        else:
            return {'noiseMantAtCompartment': 0,
                    'noiseExpAtCompartment': 0}

    @noise_at_multicompartment.setter
    def noise_at_multicompartment(self, values):
        """Set is_multicompartment noise parameters.

        :param dict values: dictionary for noise configuration, it has the form: {'mantissa': int, 'exponent': int}
        and should follow the intervals allowed by NxNet.
        :rtype: dict
        """
        if not isinstance(values, dict):
            raise ("""
             Please provide input of the form: 
             {'mantissa':int, 'exponent':int}
             """)
        self._noise_at_multicompartment = values

    @property
    def noise_exp_at_multicompartment(self):
        """Get noise exponent for compartment voltage of is_multicompartment neurons.

        :return: exponent for noise intensity in compartment voltage of is_multicompartment neurons.
        :rtype: int
        """
        return self.noise_at_multicompartment['noiseExpAtCompartment']

    @noise_exp_at_multicompartment.setter
    def noise_exp_at_multicompartment(self, value):
        """Set noise exponent for compartment voltage of is_multicompartment neurons.

        :param int value: exponent for noise intensity in compartment voltage of is_multicompartment neurons.
        """
        self._noise_at_multicompartment = {'mantissa': self.noise_at_multicompartment['noiseMantAtCompartment'],
                                           'exponent': value}

    @property
    def noise_mant_at_multicompartment(self):
        """Get noise mantissa for compartment voltage of is_multicompartment neurons.

        :return: mantissa for noise intensity in compartment voltage of is_multicompartment neurons.
        :rtype: int
        """
        return self.noise_at_multicompartment['noiseMantAtCompartment']

    @noise_mant_at_multicompartment.setter
    def noise_mant_at_multicompartment(self, value):
        """Set noise mantissa for compartment voltage of is_multicompartment neurons.

        :param int value: mantissa for noise intensity in compartment voltage of is_multicompartment neurons.
        """
        assert value == 0, "In its current form the solver can not work with nonzero noise mantissa."
        self._noise_at_multicompartment = {'mantissa': value,
                                           'exponent': self.noise_at_multicompartment['noiseExpAtCompartment']}

    @noise_at_multicompartment.deleter
    def noise_at_multicompartment(self):
        """Delete is_multicompartment noise parameters."""
        del self._noise_at_multicompartment

    @property
    def noise_amplitude(self):
        """Compute the amplitude of compartments noise as the difference between max and min values.

        :return: noise amplitude.
        :rtype: int
        """
        if self.enable_noise:
            na = [3, 7, 15, 31, 63, 127, 254, 255, 510, 1020, 2040, 4080, 8160, 16320, 32640, 65280]
            nm = self.noise_at_multicompartment['noiseMantAtCompartment']
            ne = self.noise_at_multicompartment['noiseExpAtCompartment']
            return na[ne] + nm * 2 ** (ne - 1)
            # return int(255 * 2 ** (-7. + self.noise_at_multicompartment['noiseExpAtCompartment']))
        else:
            return 0

    @property
    def v_min_exp(self):
        """Get minimum voltage at which compartment voltage saturates for is_multicompartment neurons.

        :return: lower bound for is_multicompartment voltage.
        :rtype: int
        """
        return self._vMinExp

    @v_min_exp.setter
    def v_min_exp(self, value):
        """Set minimum voltage at which compartment voltage saturates for is_multicompartment neurons.

        :param int value: lower bound for is_multicompartment voltage.
        """
        self._vMinExp = value

    @property
    def vth_1_mant(self):
        """Get mantissa for threshold voltage of middle compartment of is_multicompartment neurons.

        :return: voltage threshold mantissa for middle compartment.
        """
        if self._vth_1_mant:
            assert self._vth_1_mant * 2 ** 6 > self.noise_amplitude
            return self._vth_1_mant
        else:
            if self.nkwargs and 'vth_1_mant' in self.nkwargs.keys():
                v_th_mant_1 = self.nkwargs['vth_1_mant']
            else:
                v_th_mant_1 = ((2 ** 13 / 2 - 1) * 2 ** 3) / 2 ** 6
            assert v_th_mant_1 * 2 ** 6 > self.noise_amplitude
            return v_th_mant_1

    @vth_1_mant.setter
    def vth_1_mant(self, value):
        """Set mantissa for threshold voltage of middle compartment of is_multicompartment neurons.

        :param int value: voltage threshold mantissa for middle compartment.
        """
        assert value * 2 ** 6 > self.noise_amplitude
        self._vth_1_mant = value

    @property
    def _vth_1_tot(self):
        """Get actual value of threshold voltage of middle compartment of is_multicompartment neurons.

        :return: actual value of threshold voltage of middle compartment.
        :rtype: int
        """
        return self.vth_1_mant * 2 ** 6

    @property
    def _vth_2_mant(self):
        """Get mantissa of threshold voltage for top compartment of is_multicompartment neurons.
        :rtype: int
        :return: Mantissa of threshold voltage for top compartment.
        """
        return (254 / 2) * 2 ** 6 - self.noise_amplitude

    @property
    def _vth_2_tot(self):
        """Get actual value of threshold voltage for top compartment of is_multicompartment neurons.
        :rtype: int
        :return: value for threshold voltage of top compartment.
        """
        return self._vth_2_mant * 2 ** 6

    @property
    def _vth_3_mant(self):
        """Get mantissa for the threshold of the top compartment in is_multicompartment neurons.

         Total threshold should be less than the bias current, as well as noise should not cause firing.

         :return: mantissa for the threshold of the top compartment.
         :rtype: int
         """
        # vth_3_mant = ((2 * self._b_3_tot -abs(self.w_min) #+ self.multicompartment_prototype0.bias
        #                ) / 2) / 2 ** 6
        vth_3_mant = int((self._b_3_tot + self.b_1_tot) / (2 ** self._b_3_exp)) - self.noise_amplitude / 2 ** 6 - 4
        assert abs(self.w_min) == 0 or abs(self.w_min) > (self._b_3_tot + self.b_1_tot) + self.noise_amplitude - \
               vth_3_mant * 2 ** 6, \
            "Weights should be larger so that noise does not cause interference with online validation"
        return vth_3_mant

    @property
    def _vth_3_tot(self):
        """Get actual value for the threshold of the top compartment in is_multicompartment neurons.

         Total threshold should be less than the bias current, as well as noise should not cause firing.

         :return: actual value for the threshold of the top compartment.
         :rtype: int
         """
        return self._vth_3_mant * 2 ** 6

    @property
    def _b_1_mant(self):
        """Get mantissa of bias current of middle compartment in is_multicompartment neurons.

        :return: bias current mantissa for middle compartment.
        :rtype: int
        """
        if self.bias_to_fire == -1:
            return 0
        else:
            return int((self.vth_1_mant * 2 ** 6) / (self.bias_to_fire * 2 ** self._b_1_exp))

    @_b_1_mant.setter
    def _b_1_mant(self, value):
        self.__b_1_mant = value

    @property
    def _b_1_exp(self):
        """Get exponent of bias current of middle compartment in is_multicompartment neurons.

        :return: bias current exponent for middle compartment.
        :rtype: int
        """
        return 6

    @_b_1_exp.setter
    def _b_1_exp(self, value):
        self.__b_1_exp = value

    @property
    def b_1_tot(self):
        """Get actual value of bias current of middle compartment in is_multicompartment neurons.

        :return: bias current value for middle compartment.
        :rtype: int
        """
        return self._b_1_mant * 2 ** self._b_1_exp

    @property
    def _b_3_mant(self):
        """Get mantissa for the bias current of the top compartment in is_multicompartment neurons.

        Total bias current should be higher than voltage and not stop firing because of noise.

        :return: mantissa for the bias current of the top compartment.
        :rtype: int
        """
        return 4095 - self._b_1_mant

    @property
    def _b_3_exp(self):
        """Get exponent for the bias current of the top compartment in is_multicompartment neurons.

        Total bias current should be higher than voltage and not stop firing because of noise.

        :return: exponent for the bias current of the top compartment.
        :rtype: int
        """
        return self._b_1_exp

    @property
    def _b_3_tot(self):
        """Get actual value for the bias current of the top compartment in is_multicompartment neurons.

        Total bias current should be higher than voltage and not stop firing because of noise.

        :return: actual value  for the bias current of the top compartment.
        :rtype: int
        """
        return self._b_3_mant * 2 ** self._b_3_exp

    @property
    def v_decay(self):
        """Get compartment voltage decay for the base compartment of is_multicompartment neurons. Such value
        determines if
        the neuron is IF or LIF

        :return: voltage decay of base compartment in is_multicompartment neurons.
        :rtype: int

        """
        return self._v_decay

    @v_decay.setter
    def v_decay(self, value):
        """Set value of voltage decay for base compartment in is_multicompartment neurons.

        :param int value: compartment voltage decay sets the value of compartmentVoltageDecay for the base compartment.
        """
        if not isinstance(value, int):
            warnings.warn('''"v_decay should be an integer''')
            value = int(value)
        if value not in range(0, 2 ** 12):
            raise ValueError('''v_decay should be between 0 and 2**12-1''')
        self._v_decay = value
        for compartment in self.compartment_group:
            compartment.compartmentVoltageDecay = value

    @property
    def u_decay(self):
        """Get compartment current decay for the base compartment of is_multicompartment neurons, it should be 0 for
        box_synapse.

        :return: current decay of base compartment in is_multicompartment neurons.
        :rtype: int

        """
        return self._u_decay

    @u_decay.setter
    def u_decay(self, value):
        """Set value of current decay for base compartment in is_multicompartment neurons.

        :param int value: compartment current decay sets the value of compartmentVoltageDecay for the base compartment.
        """
        if not isinstance(value, int):
            warnings.warn('''"u_decay should be an integer''')
            value = int(value)
        if value not in range(0, 2 ** 12 - 1):
            raise ValueError('''v_decay should be between 0 and 2**12-1''')
        self._u_decay = value

    @property
    def _w_12_mant(self):
        """Get mantissa for the weight of the the connection between middle and top compartments.

        Set to maximum to aleviate parameter constraints.

        :return: weight mantissa for middle to top autapse.
        :rtype: int
        """
        return 254

    @property
    def _w_12_exp(self):
        """Get exponent for the weight of the the connection between middle and top compartments.

        Set to maximum to aleviate parameter constraints.

        :return: weight exponent for middle to top autapse.
        :rtype: int
        """
        return 6

    @property
    def _w_12_tot(self):
        """Get actual value for the weight of the the connection between middle and top compartments.

        Set to maximum to alleviate parameter constraints.

        :return: weight value for middle to top autapse.
        :rtype: int
        """
        return self._w_12_mant * 2 ** (6 + self._w_12_exp)

    @property
    def _w_32_mant(self):
        """Get mantissa for the weight of the the connection between top and middle compartments.

        This synapse is used to encode the state of the neuron 3 timesteps in the past for solution readout by one of
        the
        LMTs at the present timestep. The value of the weight should be small so that this does not interfere with the
        firing of the middle compartment.

        :return: mantissa for the weight of the the connection between top and middle compartments.
        :rtype: int
        """
        return 1

    @property
    def _w_32_exp(self):
        """Get exponent for the weight of the the connection between top and middle compartments.

        This synapse is used to encode the state of the neuron 3 timesteps in the past for solution readout by one of
        the
        LMTs at the present timestep. The value of the weight should be small so that this does not interfere with the
        firing of the middle compartment.

        :return: exponent for the weight of the the connection between top and middle compartments.
        :rtype: int
        """
        return 0

    @property
    def _w_32_tot(self):
        """Get actual value for the weight of the the connection between top and middle compartments.

        This synapse is used to encode the state of the neuron 3 timesteps in the past for solution readout by one of
        the
        LMTs at the present timestep. The value of the weight should be small so that this does not interfere with the
        firing of the middle compartment.

        :return: actual value for the weight of the the connection between top and middle compartments.
        :rtype: int
        """
        return self._w_32_mant * 2 ** (6 + self._w_32_exp)

    @property
    def multicompartment_prototype0(self):
        """Define prototype and compartment operations for the second compartment of is_multicompartment neurons.
        """
        compartment_prototype = nx.CompartmentPrototype(biasMant=self._b_1_mant,
                                                        biasExp=self._b_1_exp,
                                                        vThMant=self.vth_1_mant,
                                                        compartmentVoltageDecay=self.v_decay,
                                                        compartmentCurrentDecay=self.u_decay,
                                                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                                        enableNoise=self.enable_noise,
                                                        randomizeVoltage=1 if self.enable_noise else 0,
                                                        randomizeCurrent=0,
                                                        logicalCoreId=self._logical_core_id,
                                                        vMinExp=self.v_min_exp,
                                                        numDendriticAccumulators=self._num_dendritic_acumulators,
                                                        **self.noise_at_multicompartment
                                                        )
        # threshold operation
        compartment_prototype.thresholdBehavior = enums.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
        # input operation
        compartment_prototype.stackIn = enums.COMPARTMENT_INPUT_MODE.SKIP
        # output operation
        compartment_prototype.stackOut = enums.COMPARTMENT_OUTPUT_MODE.PUSH_U
        # join operation
        compartment_prototype.compartmentJoinOperation = enums.COMPARTMENT_JOIN_OPERATION.SKIP
        return compartment_prototype

    @property
    def multicompartment_prototype1(self):
        """Define prototype and compartment operations for the third compartment of is_multicompartment neurons.
        """
        compartment_prototype = nx.CompartmentPrototype(biasMant=0,
                                                        biasExp=0,
                                                        vThMant=self._vth_2_mant,
                                                        compartmentVoltageDecay=2 ** 12 - 1,
                                                        compartmentCurrentDecay=0,
                                                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                                        enableNoise=self.enable_noise,
                                                        randomizeVoltage=1 if self.enable_noise else 0,
                                                        randomizeCurrent=0,
                                                        logicalCoreId=self._logical_core_id,
                                                        vMinExp=self.v_min_exp,
                                                        numDendriticAccumulators=self._num_dendritic_acumulators,
                                                        **self.noise_at_multicompartment
                                                        )
        # threshold operation
        compartment_prototype.thresholdBehavior = enums.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
        # input operation
        compartment_prototype.stackIn = enums.COMPARTMENT_INPUT_MODE.SKIP
        # output operation
        compartment_prototype.stackOut = enums.COMPARTMENT_OUTPUT_MODE.PUSH
        # join operation
        compartment_prototype.compartmentJoinOperation = enums.COMPARTMENT_JOIN_OPERATION.SKIP
        return compartment_prototype

    @property
    def multicompartment_prototype2(self):
        """Define prototype and compartment operations for the third compartment of is_multicompartment neurons.
        """
        compartment_prototype = nx.CompartmentPrototype(biasMant=self._b_3_mant,
                                                        biasExp=self._b_3_exp,
                                                        vThMant=self._vth_3_mant,
                                                        compartmentVoltageDecay=2 ** 12 - 1,
                                                        compartmentCurrentDecay=2 ** 12 - 1,
                                                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                                        enableNoise=self.enable_noise,
                                                        randomizeVoltage=1 if self.enable_noise else 0,
                                                        randomizeCurrent=0,
                                                        logicalCoreId=self._logical_core_id,
                                                        vMinExp=self.v_min_exp,
                                                        numDendriticAccumulators=self._num_dendritic_acumulators,
                                                        **self.noise_at_multicompartment
                                                        )

        # threshold operation
        compartment_prototype.thresholdBehavior = enums.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET
        # input operation
        compartment_prototype.stackIn = enums.COMPARTMENT_INPUT_MODE.POP_A_AND_B
        # # output operation
        compartment_prototype.stackOut = enums.COMPARTMENT_OUTPUT_MODE.SKIP
        # # join operation
        compartment_prototype.compartmentJoinOperation = enums.COMPARTMENT_JOIN_OPERATION.PASS
        return compartment_prototype

    @property
    def multicompartment_prototype3(self):
        """Define prototype and compartment operations for the second compartment of is_multicompartment neurons.
        """
        compartment_prototype = nx.CompartmentPrototype(biasMant=0,
                                                        biasExp=6,
                                                        vThMant=131071,
                                                        compartmentVoltageDecay=2 ** 12 - 1,
                                                        compartmentCurrentDecay=0,  # 2 ** 12 - 1,
                                                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                                        enableNoise=self.enable_noise,
                                                        randomizeVoltage=1 if self.enable_noise else 0,
                                                        randomizeCurrent=0,
                                                        logicalCoreId=self._logical_core_id,
                                                        vMinExp=self.v_min_exp,
                                                        refractoryDelay=1,
                                                        numDendriticAccumulators=self._num_dendritic_acumulators,
                                                        **self.noise_at_multicompartment
                                                        )
        return compartment_prototype

    def _create_multicompartment_neuron(self):
        """Create multicompartment neuron using the prototypes defined previously to the call of this method.

        The second compartment should be excitatorily connected to the third compartment using a box synapse with the
        same duration as the one for synapse between different multicompartments.
        """
        cmpt_grp = self.main_net.createCompartmentGroup()

        prototypes = [self.multicompartment_prototype0, self.multicompartment_prototype1,
                      self.multicompartment_prototype2, self.multicompartment_prototype3]
        for prototype in prototypes:
            cmpt = self.main_net.createCompartment(prototype=prototype)
            cmpt_grp.addCompartments(cmpt)
        # postSynResponseMode is 1 for a BOX synapse.
        multicompartment_1_to_2 = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                         weight=self._w_12_mant,
                                                         weightExponent=self._w_12_exp,
                                                         # enableDelay=True,
                                                         postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
                                                         delay=self.box_duration,
                                                         numDelayBits=self._num_delay_bits
                                                         )
        # Base to Middle compartments autapse
        cmpt_grp[0].connect(cmpt_grp[1], prototype=multicompartment_1_to_2)

        # autapse to save state for late readout from LMT
        multicompartment_3_to_4 = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                         weight=2,
                                                         weightExponent=0,
                                                         # enableDelay=True,
                                                         postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX,
                                                         delay=1,
                                                         numDelayBits=self._num_delay_bits)
        cmpt_grp[2].connect(cmpt_grp[3], prototype=multicompartment_3_to_4)
        return cmpt_grp

    def _create_multicompartment_clamp_inactive(self):
        """Create an inactive compartment for use as complement to clamp values in is_multicompartment network."""
        prototype_clamp_complement = nx.CompartmentPrototype(biasMant=0,
                                                             biasExp=0,
                                                             vThMant=2 ** 17 - 1,
                                                             compartmentVoltageDecay=self.v_decay,
                                                             compartmentCurrentDecay=self.u_decay,
                                                             functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                                             enableNoise=self.enable_noise,
                                                             randomizeVoltage=1,
                                                             randomizeCurrent=0,
                                                             logicalCoreId=self._logical_core_id,
                                                             vMinExp=self.v_min_exp,
                                                             numDendriticAccumulators=self._num_dendritic_acumulators,
                                                             **self.noise_at_multicompartment
                                                             )
        if self.enable_activity_traces:
            prototype_clamp_complement.activityImpulse = 3
            prototype_clamp_complement.activityTimeConstant = 1
            prototype_clamp_complement.enableHomeostasis = True
            prototype_clamp_complement.maxActivity = 127
            prototype_clamp_complement.minActivity = 0
            prototype_clamp_complement.activity = 0
            prototype_clamp_complement.tEpoch = 1
            prototype_clamp_complement.homeostasisGain = 0
        cmpt_grp = self.main_net.createCompartmentGroup(size=4, prototype=prototype_clamp_complement)
        return cmpt_grp

    def _configure_activity_traces(self):
        """Configure activity traces for last compartment, increase at every spike and decay to zero after 3
        timesteps"""
        prototypes = [self.multicompartment_prototype0, self.multicompartment_prototype1,
                      self.multicompartment_prototype2, self.multicompartment_prototype3]
        for prototype in prototypes:
            prototype.activityImpulse = 3
            prototype.activityTimeConstant = 1
            prototype.enableHomeostasis = True
            prototype.maxActivity = 127
            prototype.minActivity = 0
            prototype.activity = 0
            prototype.tEpoch = 1
            prototype.homeostasisGain = 0

    def run(self, runtime):
        self.compartment_group.net.run(runtime)
        self.compartment_group.net.disconnect()


def _overwrite_c_snip(lfsr_offset):
    """Used for writing the offset time from the lfsr hack."""
    import fileinput
    path = os.path.dirname(os.path.realpath(__file__)) + "/spiking.c"
    with fileinput.input(path, inplace=True) as file:
        for line in file:
            if 'int lfsr_offset =' in line:
                print('int lfsr_offset = %d; \n' % lfsr_offset, end='')
            else:
                print(line, end='')


def _draw_random_seed():
    """ Generate a 32-bit binary random seed"""
    return random.getrandbits(32)
