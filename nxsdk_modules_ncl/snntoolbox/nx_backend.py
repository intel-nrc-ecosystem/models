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

# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import warnings
import tempfile

import numpy as np
import keras

import nxsdk_modules_ncl.dnn.src.dnn_layers as nxtf
from nxsdk.graph.nxboard import N2Board
from nxsdk.graph.processes.phase_enums import Phase
from nxsdk.graph.monitor.probes import IntervalProbeCondition

from snntoolbox.parsing.utils import get_type
from snntoolbox.conversion.utils import get_scale_fac
from snntoolbox.simulation.utils import AbstractSNN, is_spiking
from snntoolbox.simulation.plotting import plot_probe
from snntoolbox.utils.utils import ClampedReLU

from matplotlib import pyplot as plt

W_EXP_MIN = - 2 ** 3
W_EXP_MAX = - W_EXP_MIN - 1
V_THR_MAX = 2 ** 17 - 1

# Debug variables
USE_PROBES = True
INJECT_INPUT = True


class SNN(AbstractSNN):
    """Class to hold the compiled spiking neural network.

    Represents the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    """

    def __init__(self, config, queue=None):
        AbstractSNN.__init__(self, config, queue)

        self.snn = None
        self._spiking_layers = {}
        self.spike_probes = None
        self.voltage_probes = None
        self.param_scales = None
        self.slopes = None
        self._previous_layer_name = None
        self.do_probe_spikes = \
            any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                 'hist_spikerates_activations'} & self._plot_keys) \
            or 'spiketrains_n_b_l_t' in self._log_keys
        self.num_neurons_to_probe = self.config.getint(
            'loihi', 'num_neurons_to_probe', fallback=np.inf
            )
        self.neurons_to_probe = {}
        self.num_weight_bits = eval(self.config.get(
            'loihi', 'connection_kwargs'))['numWeightBits']
        self._logdir = self.config.get('paths', 'log_dir_of_current_run')
        # We could use ``fallback=None``, but if a user leaves a field empty
        # in the config file (e.g. save_output =   ), then it will show up as
        # '' here.
        self._layer_to_probe = self.config.get('loihi', 'layer_to_probe',
                                               fallback='')
        use_reset_snip = self.config.getboolean('loihi', 'use_reset_snip',
                                                fallback='')
        self._use_reset_snip = use_reset_snip if use_reset_snip != '' else True
        self._has_reset_snip = False
        self._executionTimeProbe = None
        self._energyProbe = None
        self.normalize_thresholds = self.config.getboolean(
            'loihi', 'normalize_thresholds', fallback=True)

        # Configure probing
        self.probeExecutionTime = self.config.getboolean(
            'loihi', 'probe_execution_time', fallback=False)
        self.probeEnergy = self.config.getboolean(
            'loihi', 'probe_energy', fallback=False)

        # Configure board dumping
        self.doDumpLoad = self.config.getboolean(
            'loihi', 'do_dump_load', fallback=False)

        # Limit probing interval to increase performance.
        self.probeIntervalStart = self.config.getint(
            'loihi', 'probe_interval_start', fallback=0
        )
        assert self.probeIntervalStart >= 0

        self.dt = self.config.getint(
            'loihi', 'probe_dt', fallback=1
        )
        assert self.dt >= 1


    @property
    def is_parallelizable(self):
        """
        Whether or not the simulator is able to test multiple samples in
        parallel.
        """

        return False

    def get_model_kwargs(self):
        """Return keyword arguments for NxModel constructor.

        :return: Model kwargs.
        :rtype: dict
        """

        kwargs = {'logdir': self._logdir}

        save_output = self.config.getboolean(
            'loihi', 'save_output', fallback='')
        if save_output != '':
            kwargs['saveOutput'] = save_output

        store_all_candidates = self.config.getboolean(
            'loihi', 'store_all_candidates', fallback='')
        if store_all_candidates != '':
            kwargs['storeAllCandidates'] = store_all_candidates

        num_candidates_to_compute = self.config.getint(
            'loihi', 'num_candidates_to_compute', fallback='')
        if num_candidates_to_compute != '':
            kwargs['numCandidatesToCompute'] = num_candidates_to_compute

        maxNumCoresPerChip = self.config.getint(
            'loihi', 'num_cores_per_chip', fallback=128)
        if maxNumCoresPerChip:
            kwargs['maxNumCoresPerChip'] = maxNumCoresPerChip

        return kwargs

    def get_layer_kwargs(self, layer):
        """Return keyword arguments for NxLayer constructor.

        :param keras.layers.Layer | keras.layers.Conv layer: Keras layer.

        :return: Layer kwargs.
        :rtype: dict
        """

        layer_kwargs = layer.get_config()
        compartment_kwargs = eval(self.config.get('loihi',
                                                  'compartment_kwargs'))

        # If this is the output layer and uses softmax, we determine the
        # classification guess by reading out the voltages instead of spikes.
        # For the voltages to be accurate, we need to prevent overflow by
        # setting threshold to maximum and disabling spikes and reset.
        if (hasattr(layer, 'activation') and
                layer.activation.__name__ == 'softmax'):
            compartment_kwargs['vThMant'] = V_THR_MAX
            # theshOp = 3 will prevent reset. Will not prevent overflow by
            # saturating at threshold (only works in multicompartment neurons).
            compartment_kwargs['threshOp'] = 3

        elif self.normalize_thresholds:
            # The final threshold mantissa is calculated using the thresh_mant
            # and thresh_exp from te normalization algorithm.
            vThMant = self.thresh_mants[layer.name]
            vThExp = self.thresh_exps[layer.name]
            compartment_kwargs['vThMant'] = int(vThMant * 2**vThExp)

        else:
            desired_threshold_to_input_ratio = eval(self.config.get(
                'loihi', 'desired_threshold_to_input_ratio'))
            scale = np.round(np.log2(desired_threshold_to_input_ratio))
            compartment_kwargs['vThMant'] = int(2**(8 + scale)) - 1

        if self.do_probe_spikes:
            compartment_kwargs['probeSpikes'] = True
        layer_kwargs.update(compartment_kwargs)

        connection_kwargs = eval(self.config.get('loihi', 'connection_kwargs'))

        # Check for soft-reset.
        resetMode = self.config.get(
            'loihi', 'reset_mode', fallback='hard')
        layer_kwargs.update({'resetMode': resetMode})

        layer_kwargs.update(connection_kwargs)

        vp = self.config.getboolean('loihi', 'visualize_partitions',
                                    fallback='')
        if vp != '':
            layer_kwargs['visualizePartitions'] = vp

        vp = self.config.getboolean('loihi', 'validate_partitions',
                                    fallback='')
        if vp != '':
            layer_kwargs['validatePartitions'] = vp

        encoding = self.config.get('loihi', 'synapse_encoding', fallback='')
        if encoding != '':
            layer_kwargs['synapseEncoding'] = encoding

        return layer_kwargs

    def add_input_layer(self, input_shape):
        """Add input layer.

        :param list | tuple input_shape: Input shape to the network, including
            the batch size as first dimension.
        """

        # Skip if model has already been loaded.
        if self.snn is not None:
            return

        if self._poisson_input:
            raise NotImplementedError

        name = self.parsed_model.layers[0].name
        layer_kwargs = {}
        compartment_kwargs = eval(self.config.get('loihi',
                                                  'compartment_kwargs'))

        # Check if input layer uses signed spikes.
        layer_kwargs['signed'] = self.config.getboolean(
            'loihi', 'signed_input', fallback=False
        )

        if self.normalize_thresholds:
            vThMant = self.thresh_mants[name]
            vThExp = self.thresh_exps[name]
            compartment_kwargs['vThMant'] = int(vThMant * 2 ** vThExp)

        # Check for soft-reset.
        resetMode = self.config.get(
            'loihi', 'reset_mode', fallback='hard')
        layer_kwargs.update({'resetMode': resetMode})

        if self.do_probe_spikes:
            compartment_kwargs['probeSpikes'] = True
        input_layer = nxtf.NxInputLayer(batch_input_shape=input_shape,
                                        **layer_kwargs, **compartment_kwargs)

        maxNumCompartments = self.config.getint(
            'loihi', 'max_num_compartments', fallback=2**10)
        input_layer.maxNumCompartments = maxNumCompartments

        self._spiking_layers[name] = input_layer.input
        self._previous_layer_name = name

    def add_layer(self, layer):
        """Do anything that concerns adding any layer independently of its
        type.

        :param keras.layers.Layer | keras.layers.Conv layer: Layer.
        """

        # Skip if model has already been loaded.
        if self.snn is not None:
            return

        nx_layer_type = 'Nx' + get_type(layer)

        if not hasattr(nxtf, nx_layer_type):
            return

        nx_layer_name = getattr(nxtf, nx_layer_type)

        layer_kwargs = self.get_layer_kwargs(layer)

        nx_layer = nx_layer_name(**layer_kwargs)

        # The softmax layer vmem can saturate during inference.
        # To prevent this add decay.
        if (hasattr(layer, 'activation') and
                    layer.activation.__name__ == 'softmax'):
            softmax_decay = self.config.get(
                'loihi', 'softmax_decay', fallback=2**8)
            nx_layer.compartmentKwargs['compartmentVoltageDecay'] = \
                softmax_decay

        inbound = self._spiking_layers[self._previous_layer_name]

        self._spiking_layers[layer.name] = nx_layer(inbound)

        is_pooling = 'AveragePooling' in get_type(layer)

        desired_threshold_to_input_ratio = eval(self.config.get(
            'loihi', 'desired_threshold_to_input_ratio'))

        plot_histograms = self.config.getboolean(
            'loihi', 'plot_histograms', fallback=False)

        connection_kwargs = eval(self.config.get('loihi', 'connection_kwargs'))
        num_weight_bits = connection_kwargs.get('numWeightBits', 8)
        num_bias_bits = connection_kwargs.get('numBiasBits', 12)

        # Todo : Enable saturating activations in snntoolbox.
        # The snntoolbox does not support saturating activations.
        # ReLU layers are removed. Instead, we add a custom activation to
        # the parsed model layers.
        saturation = self.config.getfloat(
            'loihi', 'saturation', fallback=0.
        )
        if saturation and (hasattr(layer, 'activation') and
                         layer.activation.__name__ != 'softmax'):
            clampedRelu = ClampedReLU(threshold=0, max_value=saturation)
            layer.activation = clampedRelu

        # Convert weights to integers.
        if len(layer.weights) or is_pooling:
            # Keras AveragePooling layers have no weights, but NxTF layers do.
            weights, biases = nx_layer.get_weights() if is_pooling else \
                layer.get_weights()

            if not is_pooling:
                layer.set_weights([weights, biases])
            else:
                # Average pooling weights are scaled by the inverse
                # of the number of averaged units.
                weights *= 1 / np.prod(layer.pool_size)

            # Get parameter scaling factor
            param_scale = self.param_scales.get(layer.name, None)
            if param_scale is None:
                param_percentile = self.config.getfloat('normalization',
                                                  'param_percentile')
                param_scale = get_scale_fac(np.abs(np.concatenate(
                    [weights, biases], None)), param_percentile)
                self.param_scales[layer.name] = param_scale

            # Get previous layer slope used for scaling biases.
            prev_slope = self.slopes.get(self._previous_layer_name, 1)

            biases = biases * prev_slope

            # Quantize weights.
            weights = np.clip(
                weights * param_scale,
                -2 ** num_weight_bits,
                2 ** num_weight_bits - 1).astype(int)

            # Quantize biases.
            biases = np.clip(
                biases * param_scale,
                -2 ** num_bias_bits,
                2 ** num_bias_bits - 1).astype(int)

            do_overflow_estimate = self.config.getboolean(
                'loihi', 'do_overflow_estimate', fallback=False)
            if do_overflow_estimate:
                check_q_overflow(weights,
                                 1 / desired_threshold_to_input_ratio)

            if plot_histograms:
                bins = 32
                plt.figure()
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax1.hist(
                    weights.ravel(), bins=bins, label='weights', alpha=0.5
                )
                ax2.hist(
                    biases.ravel(), bins=bins, label='biases', color='m', alpha=0.5
                )
                plt.legend()
                plt.savefig(self._logdir + '/hist_{}_nxmodel'.format(layer.name))

            nx_layer.set_weights([weights, biases])

        maxNumCompartments = self.config.get(
            'loihi', 'maxNumCompartments', fallback=2**10)
        nx_layer.maxNumCompartments = maxNumCompartments

        # Check if previous layer was ZeroPadding.
        if 'ZeroPadding' in self._previous_layer_name:
            padding = self._spiking_layers[self._previous_layer_name]._keras_history[0].padding
            nx_layer.zeroPadding = tuple(np.ravel(padding))
        self._previous_layer_name = layer.name

    def build_dense(self, layer):
        """Build spiking fully-connected layer.

        Not needed here.

        :param keras.layers.Dense layer: Keras Dense layer.
        """

        # Skip if model has already been loaded.
        if self.snn is not None:
            print("Skipped: Already done.")
            return

    def build_convolution(self, layer):
        """Build spiking convolution layer.

        Not needed here.

        :param keras.layers.Conv2D layer: Keras convolution layer.
        """

        # Skip if model has already been loaded.
        if self.snn is not None:
            print("Skipped: Already done.")
            return

    def build_pooling(self, layer):
        """Build spiking pooling layer.

        Not needed here.

        :param keras.layers.Pooling layer: Keras pooling layer.
        """

        # Skip if model has already been loaded.
        if self.snn is not None:
            print("Skipped: Already done.")
            return

        if layer.__class__.__name__ == 'MaxPooling2D':
            warnings.warn("Layer type 'MaxPooling' not supported yet. " +
                          "Falling back on 'AveragePooling'.", RuntimeWarning)

    def compile(self):
        """Compile the spiking network."""

        # Model may have been loaded already; otherwise, create new.
        if self.snn is None:
            self.snn = self.get_model()

        # Todo : Enable saturating activations in snntoolbox.
        # The snntoolbox does not support saturating activations.
        # ReLU layers are removed. Instead, we add a custom activation to
        # the parsed model layers. Here, we save and load the model in order
        # to apply the modifications.
        saturation = self.config.getfloat(
            'loihi', 'saturation', fallback=0.
        )
        if saturation:
            clampedReLU = ClampedReLU(threshold=0, max_value=saturation)
            custom_objects = {clampedReLU.__name__: clampedReLU}
            self.parsed_model = apply_modifications(
                self.parsed_model, custom_objects)

        # Set partition environment variable before board is started.
        partition = self.config.get('loihi', 'partition', fallback='')
        if partition != '':
            os.environ['PARTITION'] = partition

        path_models = os.path.join(self._logdir, 'model_dumps', 'runnables')
        if not os.path.exists(path_models):
            os.makedirs(path_models)

        # Try to load board from disk.
        try:
            assert self.doDumpLoad
            print("Trying to load board from {}.".format(path_models))
            board, channels = self.load_board(path_models)
            self.snn.board = board

        # Otherwise, compile model again, possibly using intermediate results.
        except (OSError, AssertionError):

            print("Could not load board.")
            self.snn.summary()
            mapper = self.snn.compileModel()
            numChips = len(self.snn.board.n2Chips)
            numCores = [len(self.snn.board.n2Chips[i].n2Cores)
                        for i in range(numChips)]
            print("numChips: {}\nnumCoresPerChip: {}\nnumCores: {}".format(
                numChips, numCores, np.sum(numCores)))

            print("Saving NxModel to {}.".format(path_models))
            self.snn.save(os.path.join(path_models, 'nxModel.h5'))

            # Snips and probes need to be created before board is started,
            # which happens during dumping and loading of the board.
            channels = None
            if self._use_reset_snip:
                try:
                    channels = self.setup_snips(self.snn.board)
                except OSError:
                    pass

            # Set up probes.
            self.set_vars_to_record()

            if self.doDumpLoad:
                print("Saving board to {}.".format(path_models))
                self.save_board(path_models)

        if self.probeExecutionTime:
            from nxsdk.api.enums.api_enums import ProbeParameter
            from nxsdk.graph.monitor.probes import PerformanceProbeCondition
            binSize = 2
            self._executionTimeProbe = self.snn.board.probe(
                probeType=ProbeParameter.EXECUTION_TIME,
                probeCondition=PerformanceProbeCondition(
                    tStart=1, tEnd=self._duration,
                    bufferSize=self._duration // binSize,
                    binSize=binSize))

        if self.probeEnergy:
            from nxsdk.api.enums.api_enums import ProbeParameter
            from nxsdk.graph.monitor.probes import PerformanceProbeCondition
            binSize = 2
            self._energyProbe = self.snn.board.probe(
                probeType=ProbeParameter.ENERGY,
                probeCondition=PerformanceProbeCondition(
                    tStart=1, tEnd=self._duration,
                    bufferSize=self._duration // binSize,
                    binSize=binSize))

        # Channels need to be configured after board is started.
        if channels is not None:
            configure_channels(self.snn.board, channels, self._num_timesteps)

    def simulate(self, **kwargs):
        """Simulate a spiking network for a certain duration

        Records any variables of interest (spike trains, membrane potentials,
        ...)

        :param kwargs: Keywoard arguments should include the dataset (key
            'x_b_l').
        :return: Array of shape (`batch_size`, `num_classes`,
            ``num_timesteps``), containing the number of output spikes of the
            neurons in the final layer, for each sample and for each time step
            during the simulation.
        :rtype: np.ndarray
        """

        data = kwargs[str('x_b_l')]

        self.set_inputs(data)

        # Clamping layers at the beginning of inference may increase
        # accuracy with reduced number of steps.
        # The lenInterval determines how many steps per call to snn.run.
        # All layer are initially clamped exept the input layer. Layers
        # are sequentially unclamped in periods of lenInterval step.
        clampLayers = self.config.getboolean(
            'loihi', 'clamp_layers', fallback=False)
        lenInterval = self.config.getint(
            'loihi', 'interval', fallback=2**10)
        numLayers = len(self.snn.layers)
        enableLayer = 1
        if self._duration <= lenInterval:
            self.snn.run(self._duration)
        else:
            numIntervals = self._duration // lenInterval
            if clampLayers:
                # Disable layer updates
                print('Disabling Layer updates')
                for layer in self.snn.layers[1:]:
                    layer.disableUpdates()

                if numIntervals < numLayers:
                    print("Duration {} less time needed inference"
                          "with layer clamping using interval {}."
                          " Using {}".format(
                        self._duration, lenInterval, numLayers * lenInterval)
                    )
                    numIntervals = numLayers

            for _ in range(numIntervals):
                self.snn.run(lenInterval)

                if clampLayers:
                    # Enable updates for next layer
                    if enableLayer < numLayers:
                        self.snn.layers[enableLayer].enableUpdates()
                        enableLayer += 1

            residue = np.max([self._duration - lenInterval * numIntervals,
                             0])
            if residue:
                self.snn.run(residue)

        if self._executionTimeProbe is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 5))
            self._executionTimeProbe.plotExecutionTime()
            plt.savefig(os.path.join(self._logdir, 'etprobe'))
            executionTime = np.stack([
                self._executionTimeProbe.totalTimePerTimeStep,
                self._executionTimeProbe.hostTimePerTimeStep,
                self._executionTimeProbe.managementTimePerTimeStep,
                self._executionTimeProbe.learningTimePerTimeStep,
                self._executionTimeProbe.spikingTimePerTimeStep,

            ], -1)
            np.savetxt(os.path.join(self._logdir, 'etprobe_csv'),
                       executionTime,
                       fmt='%.4e',
                       delimiter=',')

        if self._energyProbe is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 5))
            self._energyProbe.plotEnergy()
            plt.savefig(os.path.join(self._logdir, 'eprobe'))

        print("\nCollecting results...")
        output_b_l_t = self.get_recorded_vars(self.snn.layers)

        return output_b_l_t

    def reset(self, sample_idx):
        """Reset network variables.

        :param int sample_idx: Index of sample that has just been simulated.
            In certain applications (video data), we may want to turn off
            reset between samples.
        """

        if self._has_reset_snip:
            return

        print("Resetting membrane potentials...")
        for layer in self.snn.layers:
            if not is_spiking(layer, self.config):
                continue
            neuronSize = 2 if layer.resetMode == 'soft' else 1
            for i in range(int(np.prod(layer.output_shape[1:])) * neuronSize):
                layer[i].voltage = 0
        print("Done.")

    def end_sim(self):
        """Clean up after run."""

        self.snn.disconnect()

    def save(self, path, filename):
        """Write model architecture and parameters to disk.

        :param str path: Path to directory where to save model.
        :param str filename: Name of file to write model to.
        """

        pass

    def load(self, path, filename):
        """Load model architecture and parameters to disk.

        :param str path: Path to directory where to load model from.
        :param str filename: Name of file to load model from.
        """

        raise NotImplementedError

    def init_cells(self):
        """Set cellparameters of neurons and initialize membrane potential."""

        pass

    def set_vars_to_record(self):
        """Set variables to record during simulation."""

        # Get probeCondition to limit probing activity.
        probeCondition = None
        if self.probeIntervalStart:
            probeCondition = IntervalProbeCondition(
                dt=self.dt, tStart=self.probeIntervalStart)

        a = nxtf.ProbableStates.ACTIVITY
        v = nxtf.ProbableStates.VOLTAGE
        s = nxtf.ProbableStates.SPIKE

        do_probe_v = \
            'mem_n_b_l_t' in self._log_keys or 'v_mem' in self._plot_keys

        self.spike_probes = {}
        if do_probe_v:
            self.voltage_probes = {}

        for layer in self.snn.layers:
            if not is_spiking(layer, self.config):
                continue

            # In large networks, may not be able to probe more than a single
            # layer at the time.
            if (self._layer_to_probe != '' and
                    self._layer_to_probe != layer.name):
                continue

            if self.do_probe_spikes:
                self.spike_probes[layer.name] = []
            if do_probe_v:
                self.voltage_probes[layer.name] = []

            num_neurons = int(np.prod(layer.output_shape[1:]))
            neurons_to_probe = range(num_neurons)

            if self.num_neurons_to_probe < num_neurons and \
                    layer != get_spiking_output_layer(self.snn.layers, self.config):
                neurons_to_probe = np.random.choice(
                    range(num_neurons), size=self.num_neurons_to_probe, replace=False)
                self.neurons_to_probe[layer.name] = neurons_to_probe

            for i in neurons_to_probe:
                # Shift id for multi-compartment neurons in soft-reset mode.
                offset = 0
                if layer.resetMode == 'soft':
                    i *= 2
                    offset = 1
                if self.do_probe_spikes:
                    self.spike_probes[layer.name].append(
                        layer[i + offset].probe(a, probeCondition=probeCondition))
                if do_probe_v:
                    self.voltage_probes[layer.name].append(
                        layer[i].probe(v, probeCondition=probeCondition))

        if not USE_PROBES:
            return

        # The spikes of the last layer are recorded by default because they
        # contain the networks output (classification guess). We can use spike
        # probes here instead of activity traces because the output layer has
        # no shared output axons. But in case of softmax, we need to probe
        # voltages and convert to spikes later.
        output_layer = get_spiking_output_layer(self.snn.layers, self.config)
        num_neurons = int(np.prod(output_layer.output_shape[1:]))
        p = v if output_layer.activation.__name__ == 'softmax' else s
        neuronSize = 2 if output_layer.resetMode == 'soft' else 1
        somaOffset = 1 if (output_layer.resetMode == 'soft') and p == s \
            else 0
        self.spike_probes[output_layer.name] = \
            [output_layer[i * neuronSize + somaOffset].probe(
                p, probeCondition=probeCondition)
             for i in range(num_neurons)]

    def get_spiketrains(self, **kwargs):
        """Get spike trains of a layer.

        :param dict kwargs: Contains the 'monitor_index'.
        :return: spiketrains_b_l_t
        :rtype: np.ndarray
        """

        j = self._spiketrains_container_counter
        if self.spiketrains_n_b_l_t is None \
                or j >= len(self.spiketrains_n_b_l_t):
            return

        # Outer for-loop that calls this function starts with
        # 'monitor_index' = 0, but this is reserved for the input and handled
        # by `get_spiketrains_input()`.
        i = kwargs[str('monitor_index')]
        if i == 0:
            return

        layer = self.snn.layers[i]
        if not is_spiking(layer, self.config):
            return

        name = layer.name
        shape = self.spiketrains_n_b_l_t[j][0].shape

        if self._layer_to_probe != '' and self._layer_to_probe != name:
            return np.zeros(shape, int)

        probes = self.stack_layer_probes(self.spike_probes[name])
        num_neurons = np.prod(shape[1:-1])
        if num_neurons > self.num_neurons_to_probe and \
                layer != get_spiking_output_layer(self.snn.layers, self.config):
            neurons_to_probe = self.neurons_to_probe[layer.name]
            tProbes = np.zeros((num_neurons, probes.shape[-1]))
            tProbes[neurons_to_probe] = probes
            probes = tProbes

        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)

        is_output_layer = \
            get_spiking_output_layer(self.snn.layers, self.config).name == name
        if is_output_layer:
            if layer.activation.__name__ == 'softmax':
                # If this is the output layer and we are using softmax, we have
                # probed the voltages and need to encode them in spikes here.
                return apply_softmax(spiketrains_b_l_t, 1)
            else:
                # If no softmax was used, we have probed the spikes directly.
                return spiketrains_b_l_t
        else:
            # In all layers except the output, we use soma traces to infer
            # spikes. Need to integer divide by max value that soma traces
            # assume, to get rid of the decay tail of the soma trace. The peak
            # value (marking a spike) is defined as 127 in probe creation and
            # will be mapped to 1.
            return spiketrains_b_l_t // 127

    def get_spiketrains_input(self):
        """Get spike trains of input layer.

        :return: spiketrains_b_l_t
        :rtype: np.ndarray
        """

        layer = self.snn.layers[0]
        if layer.name not in self.spike_probes:
            return

        probes = self.stack_layer_probes(self.spike_probes[layer.name])
        shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
        num_neurons = np.prod(shape[1:-1])
        if num_neurons > self.num_neurons_to_probe and \
                        layer != get_spiking_output_layer(self.snn.layers, self.config):
            neurons_to_probe = self.neurons_to_probe[layer.name]
            tProbes = np.zeros((num_neurons, probes.shape[-1]))
            tProbes[neurons_to_probe] = probes
            probes = tProbes
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        return spiketrains_b_l_t // 127

    def get_spiketrains_output(self):
        """Get spike trains of output layer.

        :return: spiketrains_b_l_t
        :rtype: np.ndarray
        """

        layer = get_spiking_output_layer(self.snn.layers, self.config)
        probes = self.stack_layer_probes(self.spike_probes[layer.name])

        if layer.activation.__name__ == 'softmax':
            probes = apply_softmax(probes, 0)

        offset = 1 if self.probeIntervalStart else 0
        steps = self._num_timesteps - self.probeIntervalStart + offset
        shape = [self.batch_size, self.num_classes, steps]
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)

        return spiketrains_b_l_t

    def get_vmem(self, **kwargs):
        """Get membrane potential of a layer.

        :param dict kwargs: Contains the 'monitor_index'.

        :return: mem_b_l_t
        :rtype: np.ndarray
        """

        if self.voltage_probes is None:
            return

        i = kwargs[str('monitor_index')]

        if not is_spiking(self.snn.layers[i], self.config):
            return

        name = self.snn.layers[i].name
        if self._layer_to_probe != '' and self._layer_to_probe != name:
            shape = self.snn.layers[i].output_shape + (self._num_timesteps,)
            return np.zeros(shape, int) if i else None

        probes = self.voltage_probes[name]
        # Plot instead of returning input layer probes because the toolbox
        # does not expect input to record the membrane potentials.
        if i == 0:
            plot_probe(probes,
                       self.config.get('paths', 'log_dir_of_current_run'),
                       'v_input.png')
        else:
            return self.stack_layer_probes(probes)

    def stack_layer_probes(self, probes):
        """Stack probes of individual neurons in layer.

        :param list probes: Probes.

        :return: Array of probe values. Shape: (num_neurons, num_timesteps)
        :rtype: np.ndarray
        """

        return np.stack([p.data[-self._num_timesteps:] for p in probes])

    def reshape_flattened_spiketrains(self, spiketrains, shape, is_list=True):
        """Reshape flattened spiketrains.

        Converts list of spike times into array where nonzero entries
        (indicating spike times) are properly spread out across array. Then
        reshapes the flat array into original layer ``shape``.

        Parameters
        ----------

        spiketrains: ndarray
            Spike times.
        shape
            Layer shape.
        is_list: Optional[bool]
            If ``True`` (default), ``spiketrains`` is a list of spike times.
            In this case, we distribute the spike times across a numpy array.
            If ``False``, ``spiketrains`` is already a 2D array of shape
            (num_neurons, num_timesteps).

        Returns
        -------

        spiketrains_b_l_t: ndarray
            A batch of spikes for a layer over the simulation time.
            Shape: (`batch_size`, ``shape``, ``num_timesteps``)
        """

        # Temporarily move time axis so we can reshape in Fortran style.
        new_shape = shape[:-1]
        new_shape = np.insert(new_shape, 1, shape[-1])

        # Need to flatten in 'C' mode first to stack the timevectors together,
        # then reshape in 'F' style.
        arr = np.reshape(np.ravel(spiketrains), new_shape, 'F')

        # Finally, move the time axis back again.
        return np.moveaxis(arr, 1, -1)

    def set_spiketrain_stats_input(self):
        """Count number of operations based on the input spike activity."""

        AbstractSNN.set_spiketrain_stats_input(self)

    def set_inputs(self, inputs):
        """Set the input to the network in from of bias currents.

        :param np.ndarray inputs: Input array.
        """

        if not INJECT_INPUT:
            print("Not injecting input.")
            return

        print("Setting inputs", flush=True)

        # When using signed spikes, the negated input is concatenated
        # to the last dimension and values < 0 are set to 0.
        # Todo : Enable signed input spikes with Dense layers.
        # may cause error when using InputLayer followed by Dense.
        if hasattr(self.snn.layers[0], 'signed'):
            if self.snn.layers[0].signed:
                inputs = np.concatenate([inputs, -inputs], axis=-1)
                inputs[inputs < 0] = 0

        inputs = np.ravel(inputs, 'F')
        # Normalize inputs and scale up to 8 bit.
        inputs = (inputs / np.max(inputs) * (2 ** 8 - 1)).astype(int)
        neuronSize = 2 if self.snn.layers[0].resetMode == 'soft' else 1
        for i, biasMant in enumerate(inputs):
            self.snn.layers[0][i * neuronSize].biasMant = biasMant
            self.snn.layers[0][i * neuronSize].phase = 2
        print("Done setting inputs", flush=True)

    def preprocessing(self, **kwargs):
        """Do any preprocessing."""

        # Scale thresholds to bring spikerates in optimal range.
        if self.normalize_thresholds:
            print("\nNormalizing thresholds.")
            kwargs['logdir'] = self._logdir

            self.param_scales, self.slopes, \
            self.thresh_mants, self.thresh_exps = \
                normalize_nx_model(
                    self.parsed_model, self.config, **kwargs)

        else:
            # We can try to load an existing NxModel from disk, but only if we
            # skip threshold normalization, because the threshold scales are
            # only applied when building the NxModel from scratch.
            path = os.path.join(
                self.config.get('paths', 'log_dir_of_current_run'),
                'model_dumps', 'runnables', 'nxModel.h5')
            print("Trying to load NxModel from {}.".format(path))
            if os.path.exists(path):
                self.snn = nxtf.loadNxModel(path, **self.get_model_kwargs())
                return
            print("Could not load NxModel.")

            # If loading the NxModel failed, we build it from scratch later,
            # and allocate identity scale factors for it here.
            print("\nSkipping threshold normalization.\n")
            self.param_scales = {layer.name: None
                                 for layer in self.parsed_model.layers}

    def get_model(self):
        """Instantiate an NxModel from previously created NxTF layers.

        :return: NxModel.
        :rtype: nxtf.NxModel
        """
        path = os.path.join(
            self._logdir,
            'model_dumps', 'runnables', 'nxModel.h5')
        kwargs = self.get_model_kwargs()
        if os.path.exists(path):
            print("Loading NxModel from file")
            return nxtf.loadNxModel(path, **kwargs)
        else:
            input_layer = self._spiking_layers[self.parsed_model.layers[0].name]
            output_layer = self._spiking_layers[self._previous_layer_name]

        return nxtf.NxModel(input_layer, output_layer, **kwargs)

    def load_board(self, path):
        """Load board from disk.

        Expects to find three files in ``path``:
            - The board dump
            - A compressed numpy file containing the number of chips, the
              number of cores per chip, and the number of synapses per core.
            - A compressed numpy file containing the compartment resource maps
              of the NxLayers.

        This method also attempts to create snips, sets up probes. and
        reconstructs the compartment interface of the NxLayers.

        :raises OSError if any of the three files above do not exist.

        :param str path: Where to load the board from.

        :return (board, channels). The channels are None if no snips were
            created.
        """

        path_info = os.path.join(path, 'board_info.npz')
        path_board = os.path.join(path, 'board.dat')
        path_map = os.path.join(path, 'cx_resource_maps.npz')

        if not all([os.path.exists(p)
                    for p in [path_info, path_board, path_map]]):
            raise OSError

        board_info = np.load(path_info)
        board_id = int(board_info['boardId'])
        num_chips = int(board_info['numChips'])
        num_cores_per_chip = board_info['numCoresPerChip']
        num_synapses_per_chip = board_info['numSynapsesPerChip']

        board = N2Board(board_id, num_chips, num_cores_per_chip,
                        num_synapses_per_chip)

        # Snips and probes need to be created before board is started, which
        # happens during dumping and loading of the board.
        channels = None
        if self._use_reset_snip:
            try:
                channels = self.setup_snips(board)
            except OSError:
                pass

        # Reconstruct CompartmentInterface of NxLayers.
        cx_resource_maps = np.load(path_map)
        for name, cx_resource_map in cx_resource_maps.items():
            layer = self.snn.get_layer(name)
            layer.setBoardAndCxResourceMap(board, cx_resource_map)

        # Set up probes.
        self.set_vars_to_record()

        # Load board.
        board.loadNeuroCores(path_board)

        return board, channels

    def save_board(self, path):
        """Dump board to disk.

        Will create three files:
            - The board dump
            - A compressed numpy file containing the number of chips, the
              number of cores per chip, and the number of synapses per core.
            - A compressed numpy file containing the compartment resource maps
              of the NxLayers.
        :param str path: Where to save board.
        """

        path_info = os.path.join(path, 'board_info.npz')
        path_board = os.path.join(path, 'board.dat')
        path_map = os.path.join(path, 'cx_resource_maps.npz')

        numCoresPerChip = []
        numSynapsesPerChip = []
        for chip in self.snn.board.n2Chips:
            numCoresPerChip.append(chip.numCores)
            numSynapsesPerChip.append([core.synapses.numNodes
                                       for core in chip.n2CoresAsList])
        np.savez_compressed(path_info,
                            boardId=self.snn.board.id,
                            numChips=self.snn.board.numChips,
                            numCoresPerChip=numCoresPerChip,
                            numSynapsesPerChip=numSynapsesPerChip)

        self.snn.board.dumpNeuroCores(path_board)

        maps = {layer.name: layer.cxResourceMap for layer in self.snn.layers}
        np.savez_compressed(path_map, **maps)

    def setup_snips(self, board):
        """Setup snips.

        Currently only sets up a snip for resetting membrane potentials
        between samples.

        :param N2Board board: Board.
        """

        snip_dir = self.config.get('loihi', 'snip_dir', fallback='')

        if snip_dir == '':
            snip_dir = os.path.abspath(os.path.join(os.path.dirname(
                nxtf.__file__), '..', 'snips', 'reset_model_states'))

        if not os.path.exists(snip_dir):
            raise OSError

        # Configure channels.
        channels = []
        for chip_id in range(board.numChips):
            # Init SNIP for LMT1 (reset injection).
            snip = board.createSnip(
                name='init',
                cFilePath=os.path.join(snip_dir, 'snip_init.c'),
                includeDir=snip_dir,
                funcName='init_1',
                phase=Phase.EMBEDDED_INIT,
                lmtId=0,
                chipId=chip_id)

            # Reset SNIP
            board.createProcess(
                name='reset',
                cFilePath=os.path.join(snip_dir, 'snip_reset.c'),
                includeDir=snip_dir,
                guardName='do_reset',
                funcName='reset',
                phase='mgmt',
                lmtId=0,
                chipId=chip_id)

            name = bytes('channel_init_ch{}_lmt0'.format(chip_id), 'utf-8')
            channel = board.createChannel(name, 'int', 3)
            channel.connect(None, snip)
            channels.append(channel)

        self._has_reset_snip = True

        return channels


def configure_channels(board, channels, num_timesteps):
    """Configure channels of previously created snips.

    Starts board if it has not been started yet.

    :param N2Board board: Board.
    :param channels: Channel to configure.
    :param int num_timesteps: Number of timesteps that one sample is run for.
    """

    # Board may have been started already for loading / dumping.
    if not board.executor.hasStarted():
        board.start()

    for chip, channel in zip(board.n2Chips, channels):
        channel.write(3, [chip.numCores, num_timesteps, 1])

    board.sync = False


def get_shape_from_label(label):
    """
    Extract the output shape of a flattened pyNN layer from the layer name
    generated during parsing.

    Parameters
    ----------

    label: str
        Layer name containing shape information after a '_' separator.

    Returns
    -------

    : list
        The layer shape.

    Example
    -------
        >>> get_shape_from_label('02Conv2D_16x32x32')
        [16, 32, 32]

    """
    return [int(i) for i in label.split('_')[1].split('x')]


def normalize_nx_model(parsed_model, config, **kwargs):
    """Scale thresholds and weight exponents of network to ideal dynamic range.

    :param keras.Model parsed_model: Parsed Keras model.
    :param configparser.Configparser config: SNN toolbox configuration.
    :return: Scale exponents.
    :rtype: dict
    """

    # Plot histograms. Used for visualizing weights and biases
    # before and after scaling and quantization.
    plot_histograms = config.getboolean(
        'loihi', 'plot_histograms', fallback=False)

    if 'logdir' in kwargs:
        logdir = kwargs['logdir']

    if 'x_norm' in kwargs:
        x_norm = kwargs[str('x_norm')]  # Values in range [0, 1]
    elif 'x_test' in kwargs:
        x_norm = kwargs[str('x_test')]
    elif 'dataflow' in kwargs:
        x_norm, y = kwargs[str('dataflow')].next()
    else:
        raise NotImplementedError
    print("Using {} samples for normalization.".format(len(x_norm)))
    sizes = [
        len(x_norm) * np.array(layer.output_shape[1:]).prod() * 32 /
        (8 * 2**30) for layer in parsed_model.layers if len(layer.weights) > 0]
    size_str = ['{:.2f}'.format(s) for s in sizes]
    print("INFO: Need {} GB for layer activations.\n".format(size_str))

    batch_size = config.getint('simulation', 'batch_size')

    connection_kwargs = eval(config.get('loihi', 'connection_kwargs'))
    compartment_kwargs = eval(config.get('loihi', 'compartment_kwargs'))
    # Weights have a maximum of 8 bits, used for biases as well.
    num_weight_bits = connection_kwargs.get('numWeightBits', 8)
    num_bias_bits = connection_kwargs.get('numBiasBits', 12)
    weight_exponent = connection_kwargs['weightExponent']
    bias_exponent = compartment_kwargs['biasExp']

    # Todo: No need to expose this parameter then.
    assert bias_exponent == 6, (
        "Bias exponent should be equal to 6 to cancel out the fix weight and "
        "threshold gain of 2 ** 6 applied by Loihi.")

    # Todo: No need to expose this parameter then.
    assert weight_exponent == 0, ("Weight exponent is reserved for threshold "
                                  "scaling.")

    # Percentile to use for weight normalization before quantization.
    param_percentile = config.getfloat(
        'normalization', 'param_percentile', fallback=99.999
    )

    # Percentile to use for threshold clipping.
    activation_percentile = config.getfloat(
        'normalization', 'activation_percentile', fallback=99.999
    )

    desired_threshold_to_input_ratio = \
        eval(config.get('loihi', 'desired_threshold_to_input_ratio'))
    assert desired_threshold_to_input_ratio > 0

    reset_mode = config.get('loihi', 'reset_mode', fallback='hard')

    int_scale = 2 ** num_weight_bits - 1

    # Input should already be normalized, but do it again just for safety.
    x = x_norm / np.max(x_norm)
    # Convert to integers.
    dvdt = x * int_scale

    spikerates = None

    param_scales = {}
    slopes = {}
    thresh_mants = {}
    thresh_exps = {}

    W_MAX = 2**num_weight_bits - 1
    b_MAX = 2**num_bias_bits - 1

    # Init param scale
    param_scale = int_scale

    snn_emulation = keras.models.clone_model(parsed_model)
    snn_emulation.set_weights(parsed_model.get_weights())

    # Todo : Enable saturating activations in snntoolbox.
    # The snntoolbox does not support saturating activations.
    # ReLU layers are removed. Instead, we add a custom activation to
    # the parsed model layers.
    saturation = config.getfloat(
        'loihi', 'saturation', fallback=0.
    )

    # Want to optimize thr, while keeping weights and biases in right range.
    for i, layer in enumerate(snn_emulation.layers):
        name = layer.name
        print(name)

        prev_name = snn_emulation.layers[i - 1].name
        prev_param_scale = param_scales.get(prev_name, 1)
        prev_thresh_mant = thresh_mants.get(prev_name, 1)
        prev_thresh_exp = thresh_exps.get(prev_name)
        prev_slope = slopes.get(prev_name, 1)

        # Skip for layers without parameters.
        # Input, AveragePooling
        if len(layer.weights) > 0:
            # Unconstrained floats
            weights, biases = layer.get_weights()

            if plot_histograms:
                bins = 32
                plt.figure()
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax1.hist(
                    weights.ravel(), bins=bins, label='weights', alpha=0.5
                )
                ax2.hist(
                    biases.ravel(), bins=bins, label='biases', color='m', alpha=0.5
                )
                plt.legend()
                plt.savefig(logdir + '/hist_{}_real'.format(layer.name))

            # Scale biases to compensate for previous layer threshold scaling
            biases = biases * prev_slope

            # Instead of using the maximum parameter for normalization, we may
            # choose the value at a certain percentile to clip outliers.
            weight_norm = np.percentile(np.abs(weights.ravel()), param_percentile)

            print('Weight norm: {}'.format(weight_norm))

            # Calculate parameter scale factor kappa.
            scale_ratio = np.percentile(
                np.abs(biases) / weight_norm, param_percentile)
            if scale_ratio > 0:
                param_scale = np.min([W_MAX, b_MAX / scale_ratio]) / weight_norm
            else:
                param_scale = W_MAX / weight_norm
            param_scales[name] = param_scale
            print("Parameter scale: {}".format(param_scale))

            # Quantize weights.
            weights = np.clip(
                weights * param_scale,
                -2 ** num_weight_bits,
                2 ** num_weight_bits - 1).astype(int)

            # Quantize biases.
            biases = np.clip(
                biases * param_scale,
                -2 ** num_bias_bits,
                2 ** num_bias_bits - 1).astype(int)

            if plot_histograms:
                plt.figure()
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax1.hist(
                    weights.ravel(), bins=bins, label='weights', alpha=0.5
                )
                ax2.hist(
                    biases.ravel(), bins=bins, label='biases', color='m', alpha=0.5
                )
                plt.legend()
                plt.savefig(logdir + '/hist_{}_intScale'.format(layer.name))

            # Softmax layers in Loihi use no threshold.
            if (hasattr(layer, 'activation') and
                        layer.activation.__name__ == 'softmax'):
                continue

            # Apply weight changes to layer, so we can estimate PSP. Note that
            # we do not apply the bias exponent here; this method assumes (and
            # asserts above) that the bias exponent is 6 to cancel out with the
            # fix weight and threshold gain of 2 ** 6 applied by Loihi.
            layer.set_weights([weights, biases])

        if i > 0:
            # Get excitatory post-synaptic potential for each neuron in layer.
            dvdt = keras.models.Sequential([layer]).predict(spikerates,
                                                            batch_size)
            parsed_model.get_layer(name)

            # Layers like Flatten do not have spiking neurons and therefore no
            # threshold to tune. So we only need to update the input to the
            # next layer, and propagate the scale.
            if not is_spiking(layer, config):
                spikerates = dvdt
                param_scales[name] = prev_param_scale
                thresh_exps[name] = prev_thresh_exp
                thresh_mants[name] = prev_thresh_mant
                slopes[name] = prev_slope
                continue

            # Loihi AveragePooling layers get weights of ones. To reproduce
            # this in our Keras model, we need to apply the same
            # transformations as for a regular layer that has weights:
            # 1. Integer transformation, 2. Undoing integer trafo of previous
            # layer, 3. Multiplying by threshold to go from rates to voltage
            # change.
            elif 'AveragePooling' in get_type(layer):
                weight_norm = 1 / np.prod(layer.pool_size)
                param_scale = W_MAX / weight_norm
                param_scales[name] = param_scale
                dvdt = dvdt * param_scale

        # The highest EPSP determines whether to raise threshold.
        dvdt_max = get_scale_fac(dvdt[np.nonzero(dvdt)], activation_percentile)
        print("Maximum increase in compartment voltage per timestep: {}."
              "".format(int(dvdt_max)))

        dvdt_max *= desired_threshold_to_input_ratio

        # Calculate the thresh_mant and exponent.
        thresh_mant, thresh_exp = to_mantexp(
            dvdt_max, 2**num_weight_bits, W_EXP_MAX)
        assert thresh_exp <= W_EXP_MAX
        assert thresh_mant <= 2**8
        threshold = thresh_mant * 2**thresh_exp

        # Compute slope
        slope = param_scale * prev_slope / threshold

        # Compute new slope for saturating activations.
        # Todo : Enable saturating activations in snntoolbox.
        # The snntoolbox does not support saturating activations.
        # ReLU layers are removed. Instead, we add a custom activation to
        # the parsed model layers.
        if saturation:
            # If using saturation activations the slope is at most
            # the inverse of the saturation value.
            new_slope = 1 / saturation

            # Compute new threshold estimate
            new_threshold = param_scale * prev_slope / new_slope

            # The threshold is updated if the new threshold is lower.
            if new_threshold < threshold:
                print("Update previous slope {} with new slope {} based "
                      "on saturating activation value of {}.".format(
                    slope, new_slope, saturation
                ))
                thresh_mant, thresh_exp = to_mantexp(
                    new_threshold, 2**num_weight_bits, W_EXP_MAX)
                threshold = thresh_mant * 2 ** thresh_exp
                slope = new_slope

        slopes[name] = slope

        print("Setting threshold of layer {} to {} and scaling biases "
              "of subsequent layer by {}".format(name, threshold, slope))

        thresh_mants[name] = thresh_mant
        thresh_exps[name] = thresh_exp

        if reset_mode == 'soft':
            print('Weight mantissa and exponent for subtractive-reset are'
                  ' {} and {}, respectively'.format(thresh_mant, thresh_exp))

        # Apply activation function (dividing by threshold) to obtain the
        # output of the current layer, which will be used as input to the next.
        spikerates = np.minimum(dvdt / threshold, 1)
        print('\n')

    print("Done scaling thresholds.\n")

    return param_scales, slopes, thresh_mants, thresh_exps


def check_q_overflow(weights, p):
    """Check whether the dendritic accumulator q overflows.

    :param np.ndarray weights: Weights of layer.
    :param float p: Probability of spikes in layer.
    """

    num_channels = weights.shape[-1]
    weights_flat = np.reshape(weights, (-1, num_channels))
    q_min = - 2 ** 15
    q_max = - q_min - 1
    weighted_fanin = np.sum(weights_flat, 0)
    neg = np.mean(weighted_fanin < q_min)
    pos = np.mean(weighted_fanin > q_max)
    if neg or pos:
        print("In the worst case of all pre-synaptic neurons firing "
              "simultaneously, the dendritic accumulator will overflow in "
              "{:.2%} and underflow in {:.2%} of neurons.".format(pos, neg))
        print("Estimating averages...")
        neg = []
        pos = []
        num_fanin = len(weights_flat)
        for i in range(2 ** min(num_fanin, 16)):
            spikes = np.random.binomial(1, p, num_fanin)
            weighted_fanin = np.sum(weights_flat[spikes > 0], 0)
            neg.append(np.mean(weighted_fanin < q_min) * 100)
            pos.append(np.mean(weighted_fanin > q_max) * 100)
        print("On average, the dendritic accumulator will overflow in {:.2f} "
              "+/- {:.2f} % and underflow in {:.2f} +/- {:.2f} % of neurons."
              "".format(np.mean(pos), np.std(pos), np.mean(neg), np.std(neg)))


def overflow_signed(x, num_bits):
    """Compute overflow on an array of signed integers.

    Parameters
    ----------
    x : ndarray
        Integer values for which to compute values after overflow.
    num_bits : int
        Number of bits, not including sign, to compute overflow for.

    Returns
    -------
    out : ndarray
        Values of x overflowed as would happen with limited bit representation.
    overflowed : ndarray
        Boolean array indicating which values of ``x`` actually overflowed.
    """

    x = x.astype(int)

    lim = 2 ** num_bits
    smask = np.array(lim, int)  # mask for the sign bit
    xmask = smask - 1  # mask for all bits <= `bits`

    # Find where x overflowed
    overflowed = (x < -lim) | (x >= lim)

    zmask = x & smask  # if `out` has negative sign bit, == 2**bits
    out = x & xmask  # mask out all bits > `bits`
    out -= zmask  # subtract 2**bits if negative sign bit

    return out, overflowed


def to_mantexp(x, mant_max, exp_max):
    """Transform integer into a mantissa and exponent tuple.

    :param int x: Input value.
    :param int mant_max: Maximum mantissa.
    :param int exp_max: Maximum exponent.
    :return: Mantissa and exponent.
    :rtype: tuple[int]
    """

    r = np.maximum(np.abs(x) / mant_max, 1)
    exp = np.ceil(np.log2(r)).astype(int)
    assert np.all(exp <= exp_max)
    man = np.round(x / 2 ** exp).astype(int)
    assert np.all(np.abs(man) <= mant_max)
    return man, exp


def get_spiking_output_layer(layers, config):
    """Return last layer in network that emits spikes (not Flatten etc).

    :param list layers: Layers of network.
    :param configparser.ConfigParser config: SNN toolbox configuration.

    :return: Layer.
    :rtype keras.layers.Layer
    """

    for layer in reversed(layers):
        if is_spiking(layer, config):
            return layer


def apply_softmax(x, axis=1):
    """If output layer has softmax activation function, we probe the voltage
    instead of spikes, apply the softmax on the voltages here, and
    encode the result in spikes.
    """

    e = np.exp(x - np.max(x, axis, keepdims=True))
    s = np.sum(e, axis, keepdims=True)
    y = e / s
    z = np.random.random_sample(y.shape)
    return z < y


def to_integer(weights, biases, bitwidth, norm=None):
    if norm is None:
        norm = get_max_param(weights, biases)

    a_min = - 2 ** bitwidth
    a_max = - a_min - 1
    weights = np.clip(weights / norm * a_max, a_min, a_max).astype(int)
    biases = np.clip(biases / norm * a_max, a_min, a_max).astype(int)
    return weights, biases


def get_max_param(weights, biases):
    return np.max(np.abs(np.concatenate([weights, biases], None)))


def apply_modifications(model, custom_objects=None):
    """
    Applies modifications to the model layers by saving and loading. Used for rebuilding
    after modification of saturation activations.
    :param Model model:
        Modified keras model.
    :return :
        The modified model.
    """

    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return keras.models.load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)
