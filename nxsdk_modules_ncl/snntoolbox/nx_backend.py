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

import sys

import json
import os
import time
import warnings
import tempfile

import numpy as np
from tensorflow import keras

import nxsdk_modules_ncl.dnn.src.dnn_layers as nxtf
from nxsdk.graph.nxboard import N2Board
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import IntervalProbeCondition, \
    PerformanceProbeCondition
from nxsdk.composable.model import Model as ComposableModel
from nxsdk_modules_ncl.input_generator.input_generator import InputGenerator
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN

from snntoolbox.parsing.utils import get_type
from snntoolbox.conversion.utils import get_scale_fac
from snntoolbox.simulation.utils import AbstractSNN, is_spiking
from snntoolbox.simulation.plotting import plot_probe, plot_energy_probe, \
    plot_execution_time_probe, plot_power_probe, plot_parameter_histogram
from snntoolbox.utils.utils import ClampedReLU


W_EXP_MIN = - 2 ** 3
W_EXP_MAX = - W_EXP_MIN - 1
V_THR_MAX = 2 ** 17 - 1


class SNN(AbstractSNN):
    """Class to hold the compiled spiking neural network.

    Represents the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    """

    def __init__(self, config, queue=None):
        AbstractSNN.__init__(self, config, queue)

        os.environ['SLURM'] = '1'

        self.snn = None
        self.composed_snn = None
        self._spiking_layers = {}
        self.spike_probes = None
        self.voltage_probes = None
        self.param_scales = None
        self.slopes = {}
        self.thresh_exps = None
        self.thresh_mants = None
        self._previous_layer_name = None
        self.do_probe_spikes = \
            any({'spiketrains', 'spikerates', 'correlation', 'spikecounts',
                 'hist_spikerates_activations'} & self._plot_keys) or \
            any({'spiketrains_n_b_l_t',
                 'synaptic_operations_b_t'} & self._log_keys)
        self.num_neurons_to_probe = self.config.getint(
            'loihi', 'num_neurons_to_probe', fallback=np.inf)
        self.neurons_to_probe = {}
        self.num_weight_bits = eval(self.config.get(
            'loihi', 'connection_kwargs'))['numWeightBits']
        self.num_bias_bits = eval(self.config.get(
            'loihi', 'connection_kwargs'))['numBiasBits']
        self.W_MAX = 2 ** self.num_weight_bits - 1
        self.b_MAX = 2 ** self.num_bias_bits - 1
        self._logdir = self.config.get('paths', 'log_dir_of_current_run')
        # We could use ``fallback=None``, but if a user leaves a field empty
        # in the config file (e.g. save_output =   ), then it will show up as
        # '' here.
        self._layer_to_probe = self.config.get('loihi', 'layer_to_probe',
                                               fallback='')
        self._buffer_size = 1024
        self._bin_size = max(1, self._duration // 20)
        self.num_samples = self.config.getint('simulation', 'num_to_test')
        self.normalize_thresholds = self.config.getboolean(
            'loihi', 'normalize_thresholds', fallback=True)

        # Configure probing
        self.profile_performance = self.config.getboolean(
            'loihi', 'profile_performance', fallback=False)
        self._performance_probe = None

        self.clamp_layers = self.config.getboolean('loihi', 'clamp_layers',
                                                   fallback=False)

        self.clamp_duration = self.config.getint('loihi', 'interval',
                                                 fallback=2**10)

        if self.clamp_layers:
            assert self.do_probe_spikes is False, \
                "Currently, probing is not possible while clamping layers."
            assert self.batch_size == 1, \
                "Clamping layers currently does not support batch mode."
            assert self._duration % self.clamp_duration == 0, \
                "When clamping layers, the run time per sample ({}) must be " \
                "a multiple of the clamping duration ({}).".format(
                    self._duration, self.clamp_duration)

        # Plot histograms. Used for visualizing weights and biases before and
        # after scaling and quantization.
        self.plot_histograms = config.getboolean('loihi', 'plot_histograms',
                                                 fallback=False)

        # Todo: Enable saturating activations in snntoolbox. The snntoolbox
        #       does not support saturating activations. ReLU layers are
        #       removed. Instead, we add a custom activation to the parsed
        #       model layers.
        self.saturation = config.getfloat('loihi', 'saturation', fallback=0.)

        self.desired_threshold_to_input_ratio = \
            eval(self.config.get('loihi', 'desired_threshold_to_input_ratio'))
        assert self.desired_threshold_to_input_ratio > 0

        self.reset_mode = config.get('loihi', 'reset_mode', fallback='hard')

        self.signed_input = self.config.getboolean('loihi', 'signed_input',
                                                   fallback=False)

    @property
    def is_parallelizable(self):
        """
        Whether or not the simulator is able to test multiple samples in
        parallel. Return ``True`` because even though it is not really
        parallel, we can run a batch of samples in sequence with input, reset
        and readout snips before returning control to the superhost.
        """

        return True

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

        if self.normalize_thresholds:
            # The final threshold mantissa is calculated using the thresh_mant
            # and thresh_exp from the normalization algorithm.
            vThMant = self.thresh_mants[layer.name]
            vThExp = self.thresh_exps[layer.name]
            compartment_kwargs['vThMant'] = int(vThMant * 2**vThExp)

        # If thresholds are not calibrated based on the dataset, we choose
        # scaling factors simply based on the parameter distribution.
        if len(layer.weights) and not self.normalize_thresholds:
            param_scale = self.get_parameter_scale(*layer.get_weights())
            self.param_scales[layer.name] = param_scale
            # Thresholds need to be modified by the same factor as the
            # parameters so that the overall activity is not changed.
            compartment_kwargs['vThMant'] = int(round(param_scale))

        if self.do_probe_spikes:
            compartment_kwargs['probeSpikes'] = True
        layer_kwargs.update(compartment_kwargs)

        connection_kwargs = eval(self.config.get('loihi', 'connection_kwargs'))

        # Check for soft-reset.
        layer_kwargs.update({'resetMode': self.reset_mode})

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
        elif self._is_aedat_input:
            input_mode = nxtf.InputModes.AEDAT
        else:
            input_mode = nxtf.InputModes.BIAS

        name = self.parsed_model.layers[0].name
        layer_kwargs = {'signed': self.signed_input,
                        'resetMode': self.reset_mode,
                        'inputMode': input_mode,
                        'name': name}
        compartment_kwargs = eval(self.config.get('loihi',
                                                  'compartment_kwargs'))

        if self.normalize_thresholds:
            vThMant = self.thresh_mants[name]
            vThExp = self.thresh_exps[name]
            compartment_kwargs['vThMant'] = int(vThMant * 2 ** vThExp)
        else:
            compartment_kwargs['vThMant'] = 127 if self.signed_input else 255

        if self.do_probe_spikes:
            compartment_kwargs['probeSpikes'] = True
        input_layer = nxtf.NxInputLayer(input_shape[1:], input_shape[0],
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

        inbound = self._spiking_layers[self._previous_layer_name]

        self._spiking_layers[layer.name] = nx_layer(inbound)

        nx_layer.maxNumCompartments = self.config.get(
            'loihi', 'maxNumCompartments', fallback=2**10)

        # Check if previous layer was ZeroPadding.
        if 'ZeroPadding' in self._previous_layer_name:
            zp = self.parsed_model.get_layer(self._previous_layer_name)
            nx_layer.zeroPadding = tuple(np.ravel(zp.padding))

        is_pooling = 'AveragePooling' in get_type(layer)

        if self.saturation and (hasattr(layer, 'activation') and
                                layer.activation.__name__ != 'softmax'):
            layer.activation = ClampedReLU(threshold=0,
                                           max_value=self.saturation)

        # Convert weights to integers.
        if len(layer.weights) or is_pooling:
            # Keras AveragePooling layers have no weights, but NxTF layers do.
            weights, biases = nx_layer.get_weights() if is_pooling else \
                layer.get_weights()

            # Average pooling weights are scaled by the inverse of the number
            # of averaged units.
            if is_pooling:
                pool_size = np.prod(layer.pool_size)
                weights *= 1 / pool_size
                if self.param_scales[layer.name] is None:
                    self.param_scales[layer.name] = \
                        pool_size * 2 ** self.num_weight_bits

            # Get parameter scaling factor
            param_scale = self.param_scales.get(layer.name, None)

            # Get previous layer slope used for scaling biases.
            prev_slope = self.slopes.get(self._previous_layer_name, 1)

            biases = biases * prev_slope

            # Quantize weights.
            weights = to_int(weights, param_scale, self.num_weight_bits)

            # Quantize biases.
            biases = to_int(biases, param_scale, self.num_bias_bits)

            do_overflow_estimate = self.config.getboolean(
                'loihi', 'do_overflow_estimate', fallback=False)
            if do_overflow_estimate:
                check_q_overflow(weights,
                                 1 / self.desired_threshold_to_input_ratio)

            if self.plot_histograms:
                filename = 'hist_{}_nxmodel'.format(layer.name)
                plot_parameter_histogram(self._logdir, filename, weights,
                                         biases)

            nx_layer.set_weights([weights, biases])

        self._previous_layer_name = layer.name

    def build_dense(self, layer):
        """Build spiking fully-connected layer.

        Not needed here.

        :param keras.layers.Dense layer: Keras Dense layer.
        """

        if 'softmax' in layer.activation.__name__ and self.batch_size > 1:
            print("WARNING: When using batch mode (batch_size > 1), the "
                  "current softmax implementation has in rare and yet "
                  "unexplained cases resulted in a drop in accuracy. It is "
                  "recommended to also try batch_size of 1 when using "
                  "softmax.")

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

        if self.saturation:
            clampedReLU = ClampedReLU(threshold=0, max_value=self.saturation)
            custom_objects = {clampedReLU.__name__: clampedReLU}
            self.parsed_model = apply_modifications(self.parsed_model,
                                                    custom_objects)

        # Set partition environment variable before board is started.
        partition = self.config.get('loihi', 'partition', fallback='')
        if partition != '':
            os.environ['PARTITION'] = partition
        node = self.config.get('loihi', 'node', fallback='')
        if node != '':
            os.environ['BOARD'] = node

        path_models = os.path.join(self._logdir, 'model_dumps', 'runnables')

        # Try to load board from disk.
        print("Trying to load board from {}.".format(path_models))
        try:
            self.composed_snn = ComposableModel.load(path_models)
            self.snn = self.composed_snn.composables.dnn
        except OSError:
            print("Could not load board.")

        # Otherwise, compile model again, possibly using intermediate results.
        if self.composed_snn is None:
            self.snn = self.get_model()
            self.snn.summary()

            self.composed_snn = ComposableModel('dnn')
            self.compose_with_input_generator()
            self.composed_snn.compile()

            numChips = len(self.snn.board.n2Chips)
            numCores = [len(self.snn.board.n2Chips[i].n2Cores)
                        for i in range(numChips)]
            print("numChips: {}\nnumCoresPerChip: {}\nnumCores: {}".format(
                numChips, numCores, np.sum(numCores)))

            self.try_saving_composable(path_models)

        # Set up probes.
        self.set_vars_to_record()

        self.composed_snn.start(self.composed_snn.board)

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

        run_kwargs = {'aSync': True}
        batch_duration = self._duration * self.batch_size

        if self._is_aedat_input:
            dvs_gen = kwargs['dvs_gen']
            x_b_l = []
            for _ in range(self._duration):
                x_b_l.append(dvs_gen.next_eventframe_batch())
            x_b_l = np.stack(x_b_l, 1)
            remaining_events = dvs_gen.remaining_events_of_current_batch()
            if remaining_events:
                print("{} events were not processed. Consider increasing the "
                      "simulation duration.".format(remaining_events))
        else:
            x_b_l = kwargs['x_b_l']

        if self.clamp_layers:
            num_intervals, remainder = divmod(batch_duration,
                                              self.clamp_duration)
            outputs = []
            for i in range(num_intervals):
                self.apply_clamp(i)
                self.composed_snn.run(self.clamp_duration, **run_kwargs)
                self.set_inputs(x_b_l)
                outputs.append(self.get_spiketrains_output()[..., -1])
                self.composed_snn.finishRun()  # Todo: Move out of loop?
            if remainder:
                # self.composed_snn.run(remainder, **run_kwargs)
                raise NotImplementedError
            output_b_l = np.stack(outputs, -1)
            output_b_l_t = np.repeat(output_b_l, self.clamp_duration, -1)
        else:
            self.composed_snn.run(batch_duration, **run_kwargs)
            self.set_inputs(x_b_l)

            # This call has to happen before trying to read out probes,
            # otherwise their data array will be empty.
            self.composed_snn.finishRun()

            print("\nCollecting results...")
            output_b_l_t = self.get_recorded_vars(self.snn.layers)

        return output_b_l_t

    def reset(self, sample_idx):
        """Reset network variables.

        :param int sample_idx: Index of sample that has just been simulated.
            In certain applications (video data), we may want to turn off
            reset between samples.
        """

        pass

    def end_sim(self):
        """Clean up after run."""

        self.snn.disconnect()

        if self.profile_performance:
            stats = self.snn.board.energyTimeMonitor.powerProfileStats
            print_performance(stats, self._num_timesteps)
            save_performance_stats(stats, self._logdir, self._num_timesteps)
            plot_execution_time_probe(self._logdir, self._performance_probe)
            plot_energy_probe(self._logdir, self._performance_probe)
            plot_power_probe(self._logdir, self._performance_probe)

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

        if self.profile_performance:
            condition = PerformanceProbeCondition(
                tStart=1, tEnd=self.num_samples * self._duration,
                bufferSize=self._buffer_size, binSize=self._bin_size)
            self._performance_probe = self.snn.board.probe(
                ProbeParameter.ENERGY, condition)

        # Get probeCondition to limit probing activity.
        condition = None
        probe_interval_start = self.config.getint(
            'loihi', 'probe_interval_start', fallback=0)
        assert probe_interval_start >= 0
        if probe_interval_start:
            dt = self.config.getint('loihi', 'probe_dt', fallback=1)
            assert dt > 0
            condition = IntervalProbeCondition(dt=dt,
                                               tStart=probe_interval_start)

        a = nxtf.ProbableStates.ACTIVITY
        v = nxtf.ProbableStates.VOLTAGE
        s = nxtf.ProbableStates.SPIKE

        do_probe_v = \
            'mem_n_b_l_t' in self._log_keys or 'v_mem' in self._plot_keys

        self.spike_probes = {}
        if do_probe_v:
            self.voltage_probes = {}

        for layer in self.snn.layers:
            if not is_spiking(layer, self.config) and \
                    'Input' not in get_type(layer):
                continue

            name = layer.name

            # In large networks, may not be able to probe more than a single
            # layer at the time.
            if self._layer_to_probe != '' and self._layer_to_probe != name:
                continue

            if self.do_probe_spikes:
                self.spike_probes[name] = []
            if do_probe_v:
                self.voltage_probes[name] = []

            num_neurons = int(np.prod(layer.output_shape[1:]))
            neuron_size = 2 if layer.resetMode == 'soft' else 1
            offset = 1 if layer.resetMode == 'soft' else 0
            is_output_layer = name == get_spiking_output_layer(
                self.snn.layers, self.config).name

            if is_output_layer:
                # We can use spike probes here instead of activity traces
                # because the output layer has no shared output axons.
                probe_type = s
                neurons_to_probe = range(num_neurons)
            else:
                probe_type = a
                if num_neurons > self.num_neurons_to_probe:
                    neurons_to_probe = np.random.choice(
                        range(num_neurons), size=self.num_neurons_to_probe,
                        replace=False)
                    self.neurons_to_probe[name] = neurons_to_probe
                else:
                    neurons_to_probe = range(num_neurons)

            for i in neurons_to_probe:
                # Shift id for multi-compartment neurons in soft-reset mode.
                i *= neuron_size
                if self.do_probe_spikes:
                    i += offset
                    p = layer[i].probe(probe_type, probeCondition=condition)
                    self.spike_probes[name].append(p)
                if do_probe_v:
                    p = layer[i].probe(v, probeCondition=condition)
                    self.voltage_probes[name].append(p)

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
        is_output_layer = \
            get_spiking_output_layer(self.snn.layers, self.config).name == name
        if num_neurons > self.num_neurons_to_probe and not is_output_layer:
            neurons_to_probe = self.neurons_to_probe[layer.name]
            tProbes = np.zeros((num_neurons, probes.shape[-1]))
            tProbes[neurons_to_probe] = probes
            probes = tProbes

        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)

        # In all layers except the output, we use soma traces to infer spikes.
        if is_output_layer:
            if layer.activation.__name__ == 'softmax':
                # spiketrains_b_l_t will be all zero when using softmax,
                # because we take the voltage trace as output and set threshold
                # to max.
                print("Warning: When using softmax in the output layer, no "
                      "spikes will be recorded.")
            return spiketrains_b_l_t
        else:
            # Need to integer divide by max value that soma traces assume, to
            # get rid of the decay tail of the soma trace. The peak value
            # (marking a spike) is defined as 127 in probe creation and will be
            # mapped to 1.
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
        not_output = layer != get_spiking_output_layer(self.snn.layers,
                                                       self.config)
        if num_neurons > self.num_neurons_to_probe and not_output:
            neurons_to_probe = self.neurons_to_probe[layer.name]
            tProbes = np.zeros(
                (probes.shape[0], num_neurons, probes.shape[-1]))
            tProbes[:, neurons_to_probe] = probes
            probes = tProbes
        spiketrains_b_l_t = self.reshape_flattened_spiketrains(probes, shape)
        return spiketrains_b_l_t // 127

    def get_spiketrains_output(self):
        """Get spike trains of output layer.

        The current implementation uses a read-out snip that only retrieves the
        last time step. Thus the output array of this method contains zeros in
        all time steps but the last.

        :return: spiketrains_b_l_t
        :rtype: np.ndarray
        """

        # Get predicted class labels.
        out_class = self.composed_snn.composables.dnn.readout_channel.read(
            self.batch_size)
        # Transform into 1-hot encoded class vectors.
        out_spikes = keras.utils.to_categorical(out_class, self.num_classes)
        shape = (self.batch_size, self.num_classes, self._duration)
        spiketrains_b_l_t = np.zeros(shape)
        # Insert output spikes at last time step. All previous time steps are
        # ignored.
        spiketrains_b_l_t[:, :, -1] = out_spikes

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

        :return: Array of probe values.
                 Shape: (batch_size, num_neurons, num_timesteps)
        :rtype: np.ndarray
        """

        duration = self.batch_size * self._num_timesteps
        # Temporarily use wrong dimension order (batch size second place) so
        # we can directly call np.reshape.
        shape = [len(probes), self.batch_size, self._num_timesteps]
        arr = np.concatenate([p.data[-duration:] for p in probes])
        arr = np.reshape(arr, shape)
        # Move batch axis to front.
        return np.moveaxis(arr, 1, 0)

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

        batch_size = shape[0]
        layer_shape = shape[1:-1]
        num_timesteps = shape[-1]

        # Temporarily swap batch and time axis so we can reshape in Fortran
        # style.
        new_shape = [num_timesteps] + list(layer_shape) + [batch_size]

        # Need to flatten in 'C' mode first to stack the timevectors together,
        # then reshape in 'F' style.
        arr = np.reshape(np.ravel(spiketrains), new_shape, 'F')

        # Finally, move the time axis back again.
        return np.swapaxes(arr, 0, -1)

    def set_spiketrain_stats_input(self):
        """Count number of operations based on the input spike activity."""

        AbstractSNN.set_spiketrain_stats_input(self)

    def set_inputs(self, inputs):
        """Set the input to the network in form of bias currents.

        :param np.ndarray inputs: Input array.
        """

        print("Setting inputs...", flush=True)

        # Normalize inputs and scale up to 8 bit.
        inputs = (inputs / np.max(inputs) * (2 ** 8 - 1)).astype(int)

        def f(z):
            self.composed_snn.composables.input.encode(np.expand_dims(z, 0))

        for x in inputs:
            if self._is_aedat_input:
                # Input x consists of num_timesteps "event-frames". Send them
                # in sequentially.
                for x_t in x:
                    f(x_t)
            else:
                # Input x consists of a single image frame, which is sent in
                # only once.
                f(x)
        print("Done setting inputs.", flush=True)

    def preprocessing(self, **kwargs):
        """Do any preprocessing."""

        # Scale thresholds to bring spikerates in optimal range.
        if self.normalize_thresholds:
            print("\nNormalizing thresholds.")

            temp = self.normalize_nx_model(**kwargs)
            self.param_scales = temp[0]
            self.slopes = temp[1]
            self.thresh_mants = temp[2]
            self.thresh_exps = temp[3]

        else:
            print("\nSkipping threshold normalization.\n")
            self.param_scales = {layer.name: None
                                 for layer in self.parsed_model.layers}

    def get_model(self):
        """Instantiate an NxModel from previously created NxTF layers.

        :return: NxModel.
        :rtype: nxtf.NxModel
        """

        kwargs = self.get_model_kwargs()
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

        # Reconstruct CompartmentInterface of NxLayers.
        cx_resource_maps = np.load(path_map)
        for name, cx_resource_map in cx_resource_maps.items():
            layer = self.snn.get_layer(name)
            layer.setBoardAndCxResourceMap(board, cx_resource_map)

        # Set up probes.
        self.set_vars_to_record()

        # Load board.
        board.loadNeuroCores(path_board)

        return board

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

    def try_saving_composable(self, path):
        """Try to save composable model to ``path``.

        Always saves nxModel, but relies on the save method of the composable
        to be implemented.

        :param str path: Where to save model.
        """

        # The save function raises OSError if path exists.
        if os.path.exists(path):
            if len(os.listdir(path)):
                backup_path = path + '_{}'.format(time.time())
                print("Found existing model dumps while trying to save "
                      "current model. Backing up to {}.".format(backup_path))
                os.rename(path, backup_path)
            else:
                os.rmdir(path)

        print("Saving NxModel and board to {}.".format(path))
        try:
            # Todo: There are two issues here: 1. The ComposableDNN.save method
            #       is not yet implemented. 2. Parts of the Composable.save
            #       method are implemented, but calling the save method in this
            #       try clause results in a deadlock during board.run().
            #       So we skip this here.
            # self.composed_snn.save(path)
            raise NotImplementedError
        except NotImplementedError:
            print("Could not save composable model (method not implemented).")
            # If we don't call self.composed_snn.save(path), the dump folder
            # does not get created and we have to do it here.
            # Todo: Remove once we can call save function above.
            os.makedirs(path)
        self.snn.save(os.path.join(path, 'nxModel.h5'))

    def compose_with_input_generator(self):
        # Determine the number of time steps ("interval") to run for.
        if self.clamp_layers:
            # If layer clamping is enabled, check that all settings are valid.
            num_layers = len(self.parsed_model.layers)
            try:
                assert self.clamp_duration * num_layers <= self._duration
            except AssertionError:
                new_clamp_duration, res = divmod(self._duration, num_layers)
                msg = "When clamping layers, the run time per sample ({}) " \
                      "must be at least as long as the clamping duration " \
                      "({}) times the number of layers ({}). Reduced the " \
                      "clamp duration to {}.".format(
                        self._duration, self.clamp_duration, num_layers,
                        new_clamp_duration)
                msg2 = "\nNew clamp duration is either too small or does " \
                       "not evenly divide the total runtime per sample."
                assert new_clamp_duration > 1
                assert self._duration % new_clamp_duration == 0, msg + msg2
                print("SNN Toolbox Warning:", msg)
                self.clamp_duration = new_clamp_duration
            interval = self.clamp_duration
        else:
            interval = self._duration

        enable_reset = self.config.getint('simulation',
                                          'reset_between_nth_sample') > 0
        cdnn = ComposableDNN(self.snn, interval, enable_reset=enable_reset)
        cdnn.name = 'dnn'

        # Configure input generator to stream images via channels from super
        # host to Loihi. Use batch size 1 regardless of actual batch size.
        shape = (1,) + tuple(self.snn.input_shape[1:])
        # When using aedat (DVS) input, the InputGenerator processes one event
        # frame each time step. Otherwise, a dense frame is processed for
        # self._duration timesteps.
        input_interval = 1 if self._is_aedat_input else interval
        input_generator = InputGenerator(shape, interval=input_interval,
                                         numSnipsPerChip=3)
        input_generator.name = 'input'
        input_generator.setBiasExp(6)

        # Add all components to parent model and connect input generator to
        # SNN.
        self.composed_snn.add(cdnn)
        self.composed_snn.add(input_generator)
        input_generator.connect(cdnn)

        # Enforce ordering of input and reset snip.
        # The reset must execute before new input is injected.
        input_generator.processes.inputEncoder.executeAfter(
            cdnn.processes.reset)

    def apply_clamp(self, interval_idx):
        """Turn on / off updates of subsequent layers.

        Clamping higher layers at the beginning of a run may increase accuracy
        because wrong spikes due to unstable input rates are avoided.

        All layers except the input are initially clamped and released one by
        one after ``len_interval`` timesteps, starting with the lowest.

        :param int interval_idx: Current interval. Needed to determine when to
                                 release a layer, or when a sample has finished
                                 and the whole network should be clamped again.
        """

        # In batch mode, many samples may be presented in sequence.
        num_intervals_per_sample = self._duration // self.clamp_duration
        rel_interval_idx = interval_idx % num_intervals_per_sample

        # Clamp all layers at beginning of a new sample.
        if rel_interval_idx == 0:
            for layer in self.snn.layers:
                layer.disableUpdates()

        # Release clamp for the next higher layer after each interval.
        if rel_interval_idx < len(self.snn.layers):
            self.snn.layers[rel_interval_idx].enableUpdates()

    def normalize_nx_model(self, **kwargs):
        """Scale thresholds and weight exponents to ideal dynamic range.

        :return: Scale exponents.
        :rtype: dict
        """

        # Get dataset for normalization.
        if 'x_norm' in kwargs:
            x_norm = kwargs[str('x_norm')]  # Values in range [0, 1]
        elif 'x_test' in kwargs:
            x_norm = kwargs[str('x_test')]
        elif 'dataflow' in kwargs:
            x_norm = []
            dataflow = kwargs[str('dataflow')]
            num_samples_norm = self.config.getint('normalization',
                                                  'num_samples', fallback='')
            if num_samples_norm == '':
                num_samples_norm = len(dataflow) * dataflow.batch_size
            while len(x_norm) * self.batch_size < num_samples_norm:
                x = dataflow.next()
                if isinstance(x, tuple):  # Remove class label if present.
                    x = x[0]
                x_norm.append(x)
            x_norm = np.concatenate(x_norm)
        else:
            raise NotImplementedError
        print("Using {} samples for normalization.".format(len(x_norm)))
        sizes = [
            len(x_norm) * np.array(layer.output_shape[1:]).prod() * 32 /
            (8 * 2**30) for layer in self.parsed_model.layers
            if len(layer.weights) > 0]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print("INFO: Need {} GB for layer activations.\n".format(size_str))

        # Percentile to use for threshold clipping.
        activation_percentile = self.config.getfloat(
            'normalization', 'activation_percentile', fallback=99.999)

        # Input should already be normalized, but do it again just for safety.
        x = x_norm / np.max(x_norm)

        # Init param scale
        input_scale = 127 if self.signed_input else 255
        param_scale = input_scale

        # Convert input to integers.
        dvdt = x * input_scale

        spikerates = None

        param_scales = {}
        slopes = {}
        thresh_mants = {}
        thresh_exps = {}

        # Make a copy of the model so we can safely modify weights.
        snn_emulation = keras.models.clone_model(self.parsed_model)
        snn_emulation.set_weights(self.parsed_model.get_weights())

        # Want to optimize threshold while keeping weights and biases in given
        # limits.
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

                if self.plot_histograms:
                    filename = 'hist_{}_real'.format(layer.name)
                    plot_parameter_histogram(self._logdir, filename, weights,
                                             biases)

                # Scale biases to compensate for previous layer threshold
                # scaling.
                biases = biases * prev_slope

                param_scale = self.get_parameter_scale(weights, biases)
                param_scales[name] = param_scale
                print("Parameter scale: {}".format(param_scale))

                # Quantize weights.
                weights = to_int(weights, param_scale, self.num_weight_bits)

                # Quantize biases.
                biases = to_int(biases, param_scale, self.num_bias_bits)

                if self.plot_histograms:
                    filename = 'hist_{}_int'.format(layer.name)
                    plot_parameter_histogram(self._logdir, filename, weights,
                                             biases)

                # Softmax layers in Loihi use no threshold.
                if (hasattr(layer, 'activation') and
                        layer.activation.__name__ == 'softmax'):
                    thresh_mants[name] = 1  # V_THR_MAX
                    thresh_exps[name] = 0
                    continue

                # Apply weight changes to layer, so we can estimate PSP. Note
                # that we do not apply the bias exponent here; this method
                # assumes (and asserts above) that the bias exponent is 6 to
                # cancel out with the fix weight and threshold gain of 2 ** 6
                # applied by Loihi.
                layer.set_weights([weights, biases])

            if i > 0:
                # Get excitatory post-synaptic potential for each neuron.
                dvdt = keras.models.Sequential([layer]).predict(spikerates)

                # Layers like Flatten do not have spiking neurons and therefore
                # no threshold to tune. So we only need to update the input to
                # the next layer, and propagate the scale.
                if not is_spiking(layer, self.config):
                    spikerates = dvdt
                    param_scales[name] = prev_param_scale
                    thresh_exps[name] = prev_thresh_exp
                    thresh_mants[name] = prev_thresh_mant
                    slopes[name] = prev_slope
                    continue

                # Loihi AveragePooling layers get weights of ones. To reproduce
                # this in our Keras model, we need to apply the same
                # transformations as for a regular layer that has weights:
                # 1. Integer transformation, 2. Undoing integer trafo of
                # previous layer, 3. Multiplying by threshold to go from rates
                # to voltage change.
                elif 'AveragePooling' in get_type(layer):
                    weight_norm = 1 / np.prod(layer.pool_size)
                    param_scale = self.W_MAX / weight_norm
                    param_scales[name] = param_scale
                    dvdt = dvdt * param_scale

            # The highest EPSP determines whether to raise threshold.
            dvdt_max = get_scale_fac(dvdt[np.nonzero(dvdt)],
                                     activation_percentile)
            print("Maximum increase in compartment voltage per timestep: {}."
                  "".format(int(dvdt_max)))

            dvdt_max *= self.desired_threshold_to_input_ratio

            # Calculate the thresh_mant and exponent.
            thresh_mant, thresh_exp = to_mantexp(
                dvdt_max, 2**self.num_weight_bits, W_EXP_MAX)
            assert thresh_exp <= W_EXP_MAX
            assert thresh_mant <= 2**8
            threshold = thresh_mant * 2**thresh_exp

            # Compute slope
            slope = param_scale * prev_slope / threshold

            # Compute new slope for saturating activations.
            if self.saturation:
                # If using saturation activations the slope is at most the
                # inverse of the saturation value.
                new_slope = 1 / self.saturation

                # Compute new threshold estimate
                new_threshold = param_scale * prev_slope / new_slope

                # The threshold is updated if the new threshold is lower.
                if new_threshold < threshold:
                    print("Update previous slope {} with new slope {} based "
                          "on saturating activation value of {}.".format(
                            slope, new_slope, self.saturation))
                    thresh_mant, thresh_exp = to_mantexp(
                        new_threshold, 2**self.num_weight_bits, W_EXP_MAX)
                    threshold = thresh_mant * 2 ** thresh_exp
                    slope = new_slope

            slopes[name] = slope

            print("Setting threshold of layer {} to {} and scaling biases "
                  "of subsequent layer by {}".format(name, threshold, slope))

            thresh_mants[name] = thresh_mant
            thresh_exps[name] = thresh_exp

            if self.reset_mode == 'soft':
                print('Weight mantissa and exponent for subtractive-reset are'
                      ' {} and {}, respectively'.format(thresh_mant,
                                                        thresh_exp))

            # Apply activation function (dividing by threshold) to obtain the
            # output of the current layer, which will be used as input to the
            # next.
            spikerates = np.minimum(dvdt / threshold, 1)
            print('\n')

        print("Done scaling thresholds.\n")

        return param_scales, slopes, thresh_mants, thresh_exps

    def get_parameter_scale(self, weights, biases):
        param_percentile = self.config.getfloat(
            'normalization', 'param_percentile', fallback=100)

        # Instead of using the maximum parameter for normalization, we may
        # choose the value at a certain percentile to clip outliers.
        weight_norm = np.percentile(np.abs(weights.ravel()), param_percentile)
        scale_ratio = np.percentile(np.abs(biases) / weight_norm,
                                    param_percentile)

        if scale_ratio > 0:
            return np.min([self.W_MAX, self.b_MAX / scale_ratio]) / weight_norm

        return self.W_MAX / weight_norm


def print_performance(stats, num_timesteps):

    lakemont_static = stats['power']['lakemont']['static']
    lakemont_dynamic = stats['power']['lakemont']['dynamic']
    core_static = stats['power']['core']['static']
    core_dynamic = stats['power']['core']['dynamic']
    print("Static power (x86): {} mW".format(lakemont_static))
    print("Dynamic power (x86): {} mW".format(lakemont_dynamic))
    print("Total power (x86): {} mW".format(lakemont_static +
                                            lakemont_dynamic))
    print("Static power (neuro-cores): {} mW".format(core_static))
    print("Dynamic power (neuro-cores): {} mW".format(core_dynamic))
    print("Total power (neuro-cores): {} mW".format(core_static +
                                                    core_dynamic))
    print("Static power (system): {} mW".format(stats['power']['static']))
    print("Dynamic power (system): {} mW".format(stats['power']['dynamic']))
    print("Total power (system): {} mW".format(stats['power']['total']))
    time_per_sample = stats.timePerTimestep * num_timesteps / 1e3
    power = lakemont_static + lakemont_dynamic + core_static + core_dynamic
    energy_per_sample = power * time_per_sample / 1e3  # mJ
    print("Timesteps per inference: {}".format(num_timesteps))
    print("Energy per inference: {} mJ".format(energy_per_sample))
    print("Execution time per inference: {} ms".format(time_per_sample))
    print("Energy Delay Product: {} uJs".format(energy_per_sample *
                                                time_per_sample))


def save_performance_stats(stats, path, num_timesteps):
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    with open(os.path.join(path, 'performance_stats'), 'w') as f:
        json.dump(dict(stats), f, default=convert)

    with open(os.path.join(path, 'performance_summary.txt'), 'w') as f:
        stdout = sys.stdout
        sys.stdout = f
        print_performance(stats, num_timesteps)
        sys.stdout = stdout


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
    """Transform number into a mantissa and exponent tuple.

    :param float x: Input value.
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


def apply_modifications(model, custom_objects=None):
    """Apply modifications to the model layers by saving and loading.

    Used for rebuilding after modification of saturation activations.

    :param Model model: Modified keras model.
    :param dict custom_objects: Dictionary of custom objects.
    :return: The modified model.
    """

    model_path = os.path.join(tempfile.gettempdir(), str(hash(model)) + '.h5')
    try:
        model.save(model_path)
        return keras.models.load_model(model_path,
                                       custom_objects=custom_objects)
    finally:
        os.remove(model_path)


def to_int(value, scale, num_bits):
    return np.clip(np.round(value * scale),
                   -2 ** num_bits, 2 ** num_bits - 1).astype(int)
