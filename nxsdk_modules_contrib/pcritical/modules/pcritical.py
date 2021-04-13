"""
This is the main code for P-CRITICAL on Loihi.
The NxPCritical class provides the input and reservoir layers of a liquid state machine­.
Output is time-binned on the lakemonts and returned through a snip channel.
Usage examples are available on the scripts directory.
"""
import os
import logging
from time import sleep
from enum import IntEnum
import numpy as np
import networkx as netx
from quantities import ms
from scipy.sparse import coo_matrix
import nxsdk.api.n2a as nx
from nxsdk.arch.n2a.n2board import N2Board
from nxsdk.graph.monitor.probes import SpikeProbeCondition, IntervalProbeCondition
from tqdm import trange

_SCALING_FACTOR = 256
_logger = logging.getLogger(__name__)


def rescale(var, dt):
    """
    Rescale variable to fit dt, based on quantities library
    :param var: Variable to rescale
    :param dt: Time steps
    :return: Rescaled integer
    """
    return (var.rescale(dt.units) / dt).magnitude.astype(int).item()


def calc_minimum_number_of_cores(nb_of_nodes, nb_of_conn):
    """Calc an approximate minimum number of loihi neuro cores required"""
    MAX_NEURONS_PER_CORE = 1024
    MAX_CONN_PER_CORE = 10 * MAX_NEURONS_PER_CORE
    neuron_bounded = nb_of_nodes / MAX_NEURONS_PER_CORE
    conn_bounded = nb_of_conn / MAX_CONN_PER_CORE
    return int(np.ceil(max(neuron_bounded, conn_bounded)))


class NxPCritical(object):
    class PairWeightMode(IntEnum):
        BIN_SIZE_SYNC = 1 << 0
        MEAN_VALUE = 1 << 1
        HALF_VTH = 1 << 2

    def __init__(
        self,
        topology: netx.DiGraph,
        input_dim: int,
        nb_of_conn_per_input: int = 1,
        alpha=2,
        beta=0.25,
        tau_v=40 * ms,
        tau_i=5 * ms,
        v_th=1.0,
        refractory_period=2 * ms,
        dt=1 * ms,
        tau_v_pair=None,
        tau_i_pair=None,
        bin_size=60 * ms,
        pair_weight_mode: PairWeightMode = PairWeightMode.HALF_VTH,
        network=None,
        debug=False,
        get_power_eff=False,
        power_eff_input_freq=None,
    ):
        self.net = nx.NxNet() if network is None else network
        self.board = None
        self.topology = topology
        self.number_of_neurons = topology.number_of_nodes()
        self.pair_weight_mode = pair_weight_mode
        self.debug = debug
        self.get_power_eff = get_power_eff
        self.input_dim = input_dim
        if get_power_eff:
            assert not debug, "Can't get power efficiency in debug mode"
            assert power_eff_input_freq is not None
            self.power_eff_input_freq = rescale(power_eff_input_freq, 1 / dt)

        # Rescale variables for Loihi
        refractory_period = rescale(refractory_period, dt)
        v_decay = int(2 ** 12 * (1 - np.exp(-1 / rescale(tau_v, dt))))
        c_decay = int(2 ** 12 * (1 - np.exp(-1 / rescale(tau_i, dt))))

        v_decay_pair = (
            v_decay
            if tau_v_pair is None
            else int(2 ** 12 * (1 - np.exp(-1 / rescale(tau_v_pair, dt))))
        )
        c_decay_pair = (
            c_decay
            if tau_i_pair is None
            else int(2 ** 12 * (1 - np.exp(-1 / rescale(tau_i_pair, dt))))
        )
        v_th = int(v_th * _SCALING_FACTOR)
        self.bin_size = rescale(bin_size, dt)

        build_neuron_nargs = {
            "nb_of_neurons": topology.number_of_nodes(),
            "nb_of_synapses": topology.number_of_edges(),
            "nb_inputs": nb_of_conn_per_input * input_dim,
            "v_decay": v_decay,
            "c_decay": c_decay,
            "v_decay_pair": v_decay_pair,
            "c_decay_pair": c_decay_pair,
            "v_th": v_th,
            "refractory_period": refractory_period,
            "alpha": alpha,
        }
        build_synapse_nargs = {
            "topology": topology,
            "alpha": alpha,
            "beta": beta,
        }

        if get_power_eff:
            cores_left = 128  # For one full loihi chip
            self.nb_replicas = 0
            while True:
                self.nb_replicas += 1
                build_neuron_nargs["starting_core"] = 128 - cores_left
                nb_cores_used = self._build_neurons(**build_neuron_nargs)
                self._build_synapses(**build_synapse_nargs)
                cores_left -= nb_cores_used
                if cores_left < nb_cores_used:
                    break
        else:
            self._build_neurons(**build_neuron_nargs)
            self._build_synapses(**build_synapse_nargs)

            self._build_fake_probes()  # For snips bin-counters
            self._build_input_gen(
                nb_neurons=topology.number_of_nodes(),
                input_dim=input_dim,
                nb_of_conn_per_input=nb_of_conn_per_input,
            )

            self.weight_probe = self.connections.probe(
                [nx.ProbeParameter.SYNAPSE_WEIGHT],
                probeConditions=[IntervalProbeCondition(dt=self.bin_size)],
            )

        if debug:
            (self.spike_probe,) = self.grp.probe([nx.ProbeParameter.SPIKE])
            (self.pair_spike_probe,) = self._pair_grp.probe([nx.ProbeParameter.SPIKE])
            self.tag_probe = self.connections.probe(
                [nx.ProbeParameter.SYNAPSE_TAG],
                probeConditions=[IntervalProbeCondition(dt=self.bin_size)],
            )

    def _build_board(self):
        if self.board is not None:
            self.board.disconnect()
        compiler = nx.N2Compiler()
        self.board = compiler.compile(self.net)
        # self.board.sync = True  # TODO Validate
        self._build_snips()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.board is not None:
            self.board.disconnect()
        if self.net is not None:
            self.net.disconnect()

    def power_efficiency_run(self, duration: int):
        """Run a simulation for duration timesteps and return power profile dictionary"""
        with self:
            self._build_board()
            buffer_size = 1024 * 2  # from characterization.py
            self.energy_probe = self.board.probe(
                probeType=nx.ProbeParameter.ENERGY,
                probeCondition=nx.PerformanceProbeCondition(
                    tStart=1,
                    tEnd=duration,
                    bufferSize=buffer_size,
                    binSize=int(np.power(2, np.ceil(np.log2(duration / buffer_size)))),
                ),
            )
            self.board.run(duration)
            self.board.finishRun()

        power_profile_stats = self.board.energyTimeMonitor.powerProfileStats
        power_profile_stats["nb_replicas"] = self.nb_replicas
        return power_profile_stats

    def __call__(self, spike_trains: np.ndarray = None, nb_of_bins=None):
        current_time = 0
        spike_times = [[] for _ in range(self.input_dim)]
        sample_start_times = []
        for spike_train in spike_trains:
            sample_start_times.append(current_time)
            # Pad spike_trains to match bin size
            if (spike_train.shape[-1] % self.bin_size) != 0:
                padding = self.bin_size - (spike_train.shape[-1] % self.bin_size)
                spike_train = np.pad(
                    spike_train,
                    ((0, 0), (0, padding)),
                    mode="constant",
                    constant_values=0,
                )

            sample_duration = spike_train.shape[-1]
            for i, ts in enumerate(spike_train):
                current_ts = np.flatnonzero(ts) + current_time
                spike_times[i] += current_ts.tolist()
            current_time += sample_duration

        duration = current_time
        nb_samples = len(spike_trains)

        self.spike_gen.addSpikes(list(range(self.input_dim)), spikeTimes=spike_times)
        self._build_board()

        if self.pair_weight_mode & (
            self.PairWeightMode.MEAN_VALUE | self.PairWeightMode.HALF_VTH
        ):
            self.board.run(duration * nb_samples, aSync=True)

        sample_start_times = np.asarray(sample_start_times)

        next_run = 0
        sync_every = 100
        all_bins = []
        for i, t in enumerate(range(0, duration, self.bin_size)):
            if self.pair_weight_mode & self.PairWeightMode.BIN_SIZE_SYNC:
                if i < 10:  # For the first 10 bin_size duration, sync every bin
                    self.board.run(self.bin_size)
                    self._update_weights()
                elif next_run <= i:  # After, sync weights less frequently
                    while not self.board.isRunComplete():
                        sleep(0.1)
                    self._update_weights()
                    self.board.run(sync_every * self.bin_size, aSync=True)
                    next_run = sync_every + i
                    if t + sync_every * self.bin_size >= duration:
                        sync_every = (duration - t) // self.bin_size

            _logger.info("Reading from channel at bin %i", i)
            buff = self.spike_cntr_channel.read(1)
            _logger.info("Channel read success")

            buff = b"".join(
                [i.to_bytes(4, "little") for i in buff]
            )  # Convert int32 to uin8
            bins = np.frombuffer(buff, dtype=np.uint8)
            all_bins.append(bins)

        # Format bin back to samples
        binned_output = np.zeros((nb_samples, self.number_of_neurons, nb_of_bins))
        prev_id = None
        current_bin = 0
        i = 0
        for t in range(self.bin_size, duration, self.bin_size):
            sample_id = np.max(np.where(t > sample_start_times))
            if sample_id != prev_id:
                current_bin = 0
                prev_id = sample_id
            binned_output[sample_id, :, current_bin] = all_bins[i]
            current_bin += 1
            i += 1

        self.board.finishRun()
        self._update_weights()

        return binned_output

    def _build_input_gen(self, nb_neurons, input_dim, nb_of_conn_per_input):
        if self.get_power_eff:
            # Create an input neuron for power/time efficiency as spike injector are time costly
            # That will spike at some specific frequency
            neuron_spikegen_param = nx.CompartmentPrototype(
                biasMant=self.power_eff_input_freq,
                biasExp=6,
                compartmentVoltageDecay=0,
                vThMant=1000,
            )

            self.spike_gen = self.net.createCompartmentGroup(
                size=input_dim, prototype=neuron_spikegen_param
            )
        else:
            self.spike_gen = self.net.createSpikeGenProcess(numPorts=input_dim)

        input_proto = nx.ConnectionPrototype(weight=128, weightExponent=6,)
        pre = np.arange(input_dim * nb_of_conn_per_input) % input_dim
        post = (
            np.random.permutation(max(input_dim, nb_neurons) * nb_of_conn_per_input)[
                : input_dim * nb_of_conn_per_input
            ]
            % nb_neurons
        )
        connection_mask = np.zeros((input_dim, nb_neurons), dtype=np.int)
        connection_mask[pre, post] = 1
        connection_mask = coo_matrix(connection_mask)
        self.spike_gen.connect(
            self.grp, prototype=input_proto, connectionMask=connection_mask.T
        )

    def read_weights(self):
        weights = [self.weight_probe[i][0].data for i in range(len(self.weight_probe))]
        weights = np.asarray(weights)
        return weights

    def adj_matrix(self):
        weights = self.read_weights()
        last_recorded = weights[:, -1]
        cmask = self.connection_mask_grp_to_pair.todense().T
        nonzeros = cmask != 0
        cmask[nonzeros] = last_recorded
        return cmask.astype(float) / _SCALING_FACTOR

    def _update_weights(self):
        # TODO: Would be faster in snips
        weights = self.read_weights()
        last_recorded = weights[:, -1]

        # Update local weights from probe output
        self.connections.setSynapseState("weight", last_recorded[:, None].tolist())

        if self.pair_weight_mode & self.PairWeightMode.BIN_SIZE_SYNC:
            # Update pair weights
            cmask = self.connection_mask_grp_to_pair.todense().T
            nonzeros = cmask != 0
            cmask[nonzeros] = last_recorded
            cmask = cmask.T
            self._grp_to_pair.setSynapseState(
                "weight", cmask[nonzeros][0, :, None].tolist()
            )

        # weights = self.connections.getSynapseState("weight")
        # synapses = self.board.n2Chips[0].n2Cores[0].synapses
        # weights = [synapses[i].Wgt for i in range(len(synapses.data))]

    def _build_neurons(
        self,
        nb_of_neurons,
        nb_of_synapses,
        nb_inputs,
        v_decay,
        c_decay,
        v_decay_pair,
        c_decay_pair,
        v_th,
        refractory_period,
        alpha,
        starting_core=64,  # Since we use lmt 0, starting with core 64 reduces barrier sync region
    ):
        proto_args = {
            "biasMant": 0,
            "biasExp": 0,
            "vThMant": v_th,
            "compartmentVoltageDecay": v_decay,
            "refractoryDelay": refractory_period,
            "enableSpikeBackprop": 1,
            "enableSpikeBackpropFromSelf": 1,
            "compartmentCurrentDecay": c_decay,
            "thresholdBehavior": nx.COMPARTMENT_THRESHOLD_MODE.SPIKE_AND_RESET,
            "logicalCoreId": 0,
        }
        proto = nx.CompartmentPrototype(**proto_args)
        self.grp = self.net.createCompartmentGroup(size=nb_of_neurons, prototype=proto)
        proto_args["vThMant"] = v_th + alpha
        if self.pair_weight_mode & self.PairWeightMode.HALF_VTH:
            proto_args["vThMant"] += 0.5 * v_th
        proto_args["compartmentVoltageDecay"] = v_decay_pair
        proto_args["compartmentCurrentDecay"] = c_decay_pair
        proto_args["refractoryDelay"] = max(1, refractory_period - 2)
        proto_pair = nx.CompartmentPrototype(**proto_args)
        self._pair_grp = self.net.createCompartmentGroup(
            size=nb_of_neurons, prototype=proto_pair
        )

        nb_of_cores = calc_minimum_number_of_cores(
            nb_of_neurons * 2 + nb_inputs, nb_of_synapses * 2 + nb_inputs
        )
        _logger.info("Using %i cores" % nb_of_cores)
        if nb_of_cores > 64:
            starting_core = 0

        ## TODO: Could be more optimal for nb_of_cores > 16
        main_neurons_per_core = int(np.ceil(nb_of_neurons / nb_of_cores))
        for i, compartment in enumerate(self.grp):
            core = i // main_neurons_per_core + starting_core
            compartment.logicalCoreId = core
            self._pair_grp[i].logicalCoreId = core

        return nb_of_cores

    def _build_synapses(self, topology, alpha, beta):
        number_of_neurons = topology.number_of_nodes()
        weight_matrix = (
            netx.adjacency_matrix(topology).tocoo() * _SCALING_FACTOR
        ).astype(
            int
        )  # Scale sparse weight matrix

        _logger.info("number_of_neurons: %i", number_of_neurons)
        _logger.info("number_of_synapses: %i", topology.number_of_edges())
        _logger.info("Average initial weight: %.3f", np.mean(weight_matrix.data))
        _logger.info(
            "Average excitatory weights: %.3f",
            np.mean(weight_matrix.data[weight_matrix.data > 0]),
        )
        _logger.info(
            "Average inhibitory weights: %.3f",
            np.mean(weight_matrix.data[weight_matrix.data < 0]),
        )
        _logger.info("Excitatory connections: %i", np.sum(weight_matrix.data > 0))
        _logger.info("Inhibitory connections: %i", np.sum(weight_matrix.data < 0))

        inhibitory_neurons = np.unique(
            np.nonzero(weight_matrix < 0)[0]
        )  # Pre-synaptic neurons with negative weights
        excitatory_neurons = np.array(
            list(set(range(number_of_neurons)) - set(inhibitory_neurons))
        )

        _logger.info(
            "Excitatory/inhibitory neurons: %i / %i",
            len(excitatory_neurons),
            len(inhibitory_neurons),
        )

        connection_mask = np.abs(weight_matrix.sign()).astype(int)

        def bin_notation(x):
            if np.isclose(x, 0):
                return "0"
            binary = bin(int(x * 2 ** 8))[2:]
            exp = binary[::-1].index("1") - 8
            mant = int(x / 2 ** exp)
            mant_str = "" if mant == 1 else str(mant) + "*"
            return mant_str + "2^" + str(exp)

        # Build the learning rules (one per excitatory neuron with out degree > 0)
        delta_tag = "x1*y0-2^-1*u0"
        delta_w = "%s*u0-%s*r1*t*u0" % (bin_notation(beta), bin_notation(alpha),)
        learning_rule_parameters = {
            "dw": delta_w,
            "dt": delta_tag,
            "r1Impulse": 2,
            "r1TimeConstant": 1,
            "x1Impulse": 2,
            "x1TimeConstant": 50,
        }

        _logger.info("Learning rule dw: %s", delta_w)
        _logger.info("Learning rule dt: %s", delta_tag)

        self._learning_rules = []
        for i in excitatory_neurons:
            if topology.out_degree(i) > 0:
                learning_rule = self.net.createLearningRule(**learning_rule_parameters)
                self._learning_rules.append(learning_rule)
            else:
                _logger.info("Neuron id %i was not assigned a learning rule", i)
                self._learning_rules.append(None)  # Add None for index matching

        # TODO: We could re-use prototypes for inhibitory neurons -- if that's helpful
        prototypes = (
            []
        )  # Every neuron in the reservoir is associated with a connection prototype for the learning rule
        for i in range(number_of_neurons):
            is_excitatory = i in excitatory_neurons
            prototype_args = {
                "numWeightBits": 8,
                "numTagBits": 8,
                "signMode": nx.SYNAPSE_SIGN_MODE.EXCITATORY
                if is_excitatory
                else nx.SYNAPSE_SIGN_MODE.INHIBITORY,
                "weightLimitMant": 8,  # Doesn't stop the weight from growing but limits accumulation
                "weigthLimitExp": 5,
                "enableDelay": 0,
            }
            if is_excitatory:  # Create learning rule only for excitatory synapses
                learning_index = np.where(excitatory_neurons == i)[0][0]
                lr = self._learning_rules[learning_index]
                if lr is not None:  # No learning rule for neurons with out_degree of 0
                    prototype_args["enableLearning"] = 1
                    prototype_args["learningRule"] = lr

            prototypes.append(nx.ConnectionPrototype(**prototype_args))

        prototype_map = np.tile(np.arange(number_of_neurons), (number_of_neurons, 1))
        self.connections = self.grp.connect(
            self.grp,
            prototype=prototypes,
            prototypeMap=prototype_map,
            weight=weight_matrix.T,  # transpose for (src, dst) => (dst, src)
            connectionMask=connection_mask.T,
        )

        # We now create the "pair" connection using the transpose graph of the topology
        pair_proto = nx.ConnectionPrototype(
            numWeightBits=8,
            delay=0,
            enableDelay=0,
            enableLearning=0,
            signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
        )

        # TODO: We could save memory by removing inhibitory pair neurons
        self.connection_mask_grp_to_pair = connection_mask
        if self.pair_weight_mode & self.PairWeightMode.HALF_VTH:
            pair_wgt_matrix = connection_mask * (_SCALING_FACTOR - 1)
        elif self.pair_weight_mode & self.PairWeightMode.BIN_SIZE_SYNC:
            pair_wgt_matrix = weight_matrix
        elif self.pair_weight_mode & self.PairWeightMode.MEAN_VALUE:
            pair_wgt_matrix = connection_mask * int(np.mean(weight_matrix))

        self._grp_to_pair = self.grp.connect(
            self._pair_grp,
            prototype=pair_proto,
            connectionMask=connection_mask,
            weight=pair_wgt_matrix,
        )

        # We now connect the pair neurons to their respective RL channels
        for i, n_idx in enumerate(excitatory_neurons):
            learning_rule = self._learning_rules[i]
            if learning_rule is None:
                continue
            self._pair_grp[n_idx].connect(learning_rule.reinforcementChannel)

    def _build_fake_probes(self):
        # For snips code
        self.fake_spike_probes = self.grp.probe(
            nx.ProbeParameter.SPIKE, SpikeProbeCondition(tStart=100000000)
        )

    def _build_snips(self):
        """Build snips code for time-binning of reservoir spikes, see c code in snips directory"""
        if self.get_power_eff:  # We don't want to wait on I/O during energy probing
            return

        number_of_neurons = int(self.topology.number_of_nodes())
        if not self.debug:
            probe_ids = [
                prb.n2Probe.counterId - 0x20 for prb in self.fake_spike_probes[0].probes
            ]
            assert (
                list(range(number_of_neurons)) == probe_ids
            ), "Probe Ids don't match in c code, did you add probes before the spike cntr probes ?"

        include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snips")
        run_mgmt_process = self.board.createProcess(
            "runMgmt",
            includeDir=include_dir,
            cFilePath=include_dir + "/runmgmt.c",
            funcName="run_mgmt",
            guardName="do_run_mgmt",
            phase="mgmt",
            lmtId=0,
        )
        assert number_of_neurons % 16 == 0, "TODO: pad channel to 16 bytes"
        with open(os.path.join(include_dir, "constants.h"), "w") as f:
            f.write(
                "\n".join(
                    [
                        "#define nb_of_neurons %i" % number_of_neurons,
                        "#define bin_size %i" % self.bin_size,
                        "",  # EOF new line
                    ]
                )
            )
        # Keep 4*N bins in channel buffer
        self.spike_cntr_channel = self.board.createChannel(
            b"nxspkcntr", messageSize=number_of_neurons, numElements=4
        )
        self.spike_cntr_channel.connect(run_mgmt_process, None)
