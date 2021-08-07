# Copyright © 2018-2021 Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#   * Neither the name of Intel Corporation nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNCoreTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNConnectionTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNCoresStruct
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNConnectionsStruct
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNCopyStruct
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitCoreTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitConnectionTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitConnectionsStruct
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitInstanceStruct
from nxsdk_modules.lsnn.apps.relnet.RecurrentLsnn import RecurrentLsnn
import functools


def convertInputSpikesToSpikeTimes(inputSpikes):
    return (np.where(inputSpikes > 0)[0] + 1).tolist()


def connect_spike_generator_to_neuron_group(spike_generator, neuron_group, mask, weights, delays, weight_exp):
    neuron_group.createInputToReccurentLayerConnections(spike_generator, weights.T, delays.T, weight_exp)


def connect_spike_generator_to_output_neuron_group(spike_generator, neuron_group, mask, weights, delays, weight_exp):
    neuron_group.createInputToOutputLayerConnections(spike_generator, weights.T, delays.T)


# i.e. the recurrent LSNN connections
def connect_neuron_groups(presyn_neuron_group, postsyn_neuron_group, mask, weights, delays, weight_exp):
    RecurrentLsnn.createConnectionGroup(presyn_neuron_group.recurrentNeuronGroup,
                                        postsyn_neuron_group.recurrentNeuronGroup,
                                        weights.T, delays.T, wgtExp=weight_exp)


def connect_neuron_groups_to_output_groups(presyn_neuron_group, postsyn_neuron_group, mask, weights, delays, weight_exp):
    RecurrentLsnn.createConnectionGroup(presyn_neuron_group.recurrentNeuronGroup,
                                        postsyn_neuron_group.outputNeuronGroup,
                                        weights.T, delays.T)


def create_lsnn_output_spike_generators(LSNN_core_connection_array, lsnn_output_spikes, net):

    n_time_steps = lsnn_output_spikes.shape[0]
    LSNN_size = lsnn_output_spikes.shape[1]

    if LSNN_core_connection_array.ndim == 1:
        # This is the case of the question LSNN
        n_instances = 0
        n_fanout_copies = len(LSNN_core_connection_array)
        assert lsnn_output_spikes.shape == (n_time_steps, LSNN_size)
    else:
        # This is the case of the sentence LSNN
        n_instances, n_fanout_copies = LSNN_core_connection_array.shape
        assert lsnn_output_spikes.shape == (n_instances, n_time_steps, LSNN_size)

    actual_n_instances = n_instances
    if n_instances == 0:
        n_instances = 1
        LSNN_core_connection_array = np.expand_dims(LSNN_core_connection_array, 0)
        lsnn_output_spikes = np.expand_dims(lsnn_output_spikes, 0)

    lsnn_id_to_output_spike_gen_map = {}
    lsnn_output_spike_gen_array = np.ndarray((n_instances, n_fanout_copies), dtype=object)

    for instance_ind in range(n_instances):
        for fanout_copy_ind in range(n_fanout_copies):
            lsnn_copy = LSNN_core_connection_array[instance_ind, fanout_copy_ind]  # type: LSNNCopyStruct
            current_lsnn_cores = lsnn_copy.cores  # type: LSNNCoresStruct
            lsnn_cores = current_lsnn_cores.lsnn

            output_spike_gen_tuple = []
            for core in lsnn_cores:   # type: LSNNCoreTuple
                num_input = core.end - core.start
                spike_gen = net.createSpikeGenProcess(numPorts=num_input)

                # configure spikes for spike generator
                for i in range(0, num_input):
                    spikeTimeList = convertInputSpikesToSpikeTimes(lsnn_output_spikes[instance_ind, :, core.start + i])
                    spike_gen.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikeTimeList)

                output_spike_gen_tuple.append(spike_gen)
                lsnn_id_to_output_spike_gen_map[core.id] = spike_gen

            output_spike_gen_tuple = tuple(output_spike_gen_tuple)
            lsnn_output_spike_gen_array[instance_ind, fanout_copy_ind] = output_spike_gen_tuple

    if actual_n_instances == 0:
        lsnn_output_spike_gen_array = np.reshape(lsnn_output_spike_gen_array, (n_fanout_copies,))

    return lsnn_id_to_output_spike_gen_map, lsnn_output_spike_gen_array


def create_relay_of_spike_generators(id_to_spike_gen_map, spike_generator_array, net):

    # create inverse map
    spike_gen_to_id_map = {id(spike_gen): id_ for id_, spike_gen in id_to_spike_gen_map.items()}

    id_to_neuron_group_map = {}
    neuron_group_list_of_tuples = []
    for spike_gen_tuple in spike_generator_array.flatten():
        neuron_group_tuple = []
        for spike_gen in spike_gen_tuple:
            core_id = spike_gen_to_id_map[id(spike_gen)]  # spike generator core
            n_neurons = ...  # num_ports from the spike generator
            relay_neuron_params = {
                'n_reg': n_neurons,
                'n_adap': 0,
                'tau_V': 0.,
                'tau_I': 0.,
                'scaled_thr':  1,
                # These parameters are just specified for the sake of it. the are irrelevant
                'weight_aux':  0,
                'weight_exp': 0,
                'n_refractory': 1,
                'tau_adaptation': 0,
                'beta_exp': 0
            }

            relay_neuron_group = RecurrentLsnn(relay_neuron_params, net, core_id)
            relay_neuron_group.createRecurrentNeuronGroup()

            connect_spike_generator_to_neuron_group(
                spike_gen,
                relay_neuron_group,
                mask=np.eye(n_neurons),
                weights=np.eye(n_neurons)*5,
                delays=np.zeros((n_neurons, n_neurons)),
                weight_exp=0)

            id_to_neuron_group_map[core_id] = relay_neuron_group
            neuron_group_tuple.append(relay_neuron_group)

        neuron_group_tuple = tuple(neuron_group_tuple)
        neuron_group_list_of_tuples.append(neuron_group_tuple)

    n_neuron_group_tuples = len(neuron_group_list_of_tuples)
    neuron_group_array = np.ndarray(n_neuron_group_tuples, dtype=object)
    for i in range(n_neuron_group_tuples):
        neuron_group_array[i] = neuron_group_list_of_tuples[i]
    neuron_group_array = np.reshape(neuron_group_array, spike_generator_array.shape)

    return id_to_neuron_group_map, neuron_group_array


def perform_LSNN_placement_with_artificial_LSNN(LSNN_core_connection_array, lsnn_output_spikes, net):


    n_time_steps = lsnn_output_spikes.shape[0]
    LSNN_size = lsnn_output_spikes.shape[1]

    if LSNN_core_connection_array.ndim == 1:
        # This is the case of the question LSNN
        n_instances = 0
        n_fanout_copies = len(LSNN_core_connection_array)
        assert lsnn_output_spikes.shape == (n_time_steps, LSNN_size)
    else:
        # This is the case of the sentence LSNN
        n_instances, n_fanout_copies = LSNN_core_connection_array.shape
        assert lsnn_output_spikes.shape == (n_instances, n_time_steps, LSNN_size)

    actual_n_instances = n_instances
    if n_instances == 0:
        n_instances = 1
        LSNN_core_connection_array = np.expand_dims(LSNN_core_connection_array, 0)
        lsnn_output_spikes = np.expand_dims(lsnn_output_spikes, 0)

    lsnn_id_to_spike_gen_map, lsnn_spike_gen_array = \
        create_lsnn_output_spike_generators(LSNN_core_connection_array, lsnn_output_spikes, net)

    # print("spike_generator_ids: ")
    for spike_gen in lsnn_spike_gen_array[0, 0]:
        assert not isinstance(spike_gen, tuple) and not isinstance(spike_gen, np.ndarray)
        # print(id(spike_gen))
    lsnn_id_to_spike_gen_map_sorted = sorted(list(lsnn_id_to_spike_gen_map.items()))
    # print("spike_generator_map_ids: ")
    # for i in range(8):
        # print(id(lsnn_id_to_spike_gen_map_sorted[i]))

    lsnn_id_to_neuron_group_map, lsnn_neuron_group_array = \
        create_relay_of_spike_generators(lsnn_id_to_spike_gen_map, lsnn_spike_gen_array, net)

    n_instances, n_fanout_copies = lsnn_spike_gen_array.shape
    lsnn_id_to_relay_core_map = {}
    lsnn_id_to_relay_neuron_group_map = {}
    relay_neuron_group_array = np.ndarray((n_instances, n_fanout_copies), dtype=object)

    for instance_ind in range(n_instances):
        for fanout_copy_ind in range(n_fanout_copies):
            current_LSNN_copy = LSNN_core_connection_array[instance_ind, fanout_copy_ind]
            relay_neuron_group_tuple = []
            for core in current_LSNN_copy.cores.relay:  # type: LSNNCoreTuple
                lsnn_params = {
                    'n_reg': core.end - core.start,
                    'n_adap': 0,
                    'tau_V': 0.,
                    'tau_I': 0.,
                    'scaled_thr':  1,
                    # These parameters are just specified for the sake of it. the are irrelevant
                    'weight_aux':  0,
                    'weight_exp': 0,
                    'n_refractory': 1,
                    'tau_adaptation': 0,
                    'beta_exp': 0
                }

                # Here create LIF Neuron Group with NO connections, the connections will be created later
                ALIF_neuron_group = RecurrentLsnn(lsnn_params, net, core.id)
                ALIF_neuron_group.createRecurrentNeuronGroup()

                lsnn_id_to_relay_neuron_group_map[core.id] = ALIF_neuron_group
                relay_neuron_group_tuple.append(ALIF_neuron_group)

            relay_neuron_group_tuple = tuple(relay_neuron_group_tuple)
            relay_neuron_group_array[instance_ind, fanout_copy_ind] = tuple(relay_neuron_group_tuple)

            assert all(x.compileParams.logicalCoreId == id_ for id_, x in lsnn_id_to_neuron_group_map.items())
            assert all(x.compileParams.logicalCoreId == id_ for id_, x in lsnn_id_to_relay_neuron_group_map.items())

            # lsnn_id_to_lsnn_core_map = {core.id: core for core in current_LSNN_copy.cores.lsnn}
            # lsnn_id_to_relay_core_map = {core.id: core for core in current_LSNN_copy.cores.relay}

            # Here we connect the LSNN to the relay neurons
            for connection in current_LSNN_copy.connections.lsnn_to_relay:  # type: LSNNConnectionTuple
                presyn_core_ind = connection.presyn_core
                postsyn_core_ind = connection.postsyn_core

                # presyn_core = lsnn_id_to_lsnn_core_map[presyn_core_ind]
                # postsyn_core = lsnn_id_to_relay_core_map[postsyn_core_ind]

                # presyn_neuron_arr = np.arange(presyn_core.start, presyn_core.end)
                # postsyn_neuron_arr = np.arange(postsyn_core.start, postsyn_core.end)

                # neurons_connected = np.where(connection.mask | (connection.weights != 0))
                # neurons_connected = [presyn_neuron_arr[neurons_connected[0]],
                #                      postsyn_neuron_arr[neurons_connected[1]]]
                # neurons_connected = np.stack(neurons_connected, axis=-1)
                # print("Connections made:")
                # print('\n'.join("{} -> {}".format(x[0], x[1]) for x in neurons_connected))

                connect_neuron_groups(lsnn_id_to_neuron_group_map[presyn_core_ind],
                                      lsnn_id_to_relay_neuron_group_map[postsyn_core_ind],
                                      mask=connection.mask,
                                      weights=connection.weights,
                                      delays=connection.delays,
                                      weight_exp=connection.weight_exp)

    if actual_n_instances == 0:
        lsnn_neuron_group_array = np.reshape(lsnn_neuron_group_array, (n_fanout_copies,))
        relay_neuron_group_array = np.reshape(relay_neuron_group_array, (n_fanout_copies,))

    return (lsnn_id_to_neuron_group_map, lsnn_neuron_group_array,
            lsnn_id_to_relay_neuron_group_map, relay_neuron_group_array)


def create_input_mask_spike_generator(input_mask_core, input_mask_relay_cores, input_mask_to_relay_connections, input_mask_spikes, net,  max_n_sentences):

    n_time_steps, _ = input_mask_spikes.shape
    
    spike_gen = net.createSpikeGenProcess(numPorts=max_n_sentences)

    # configure spikes for spike generator
    for i in range(0, max_n_sentences):
        spikeTimeList = convertInputSpikesToSpikeTimes(input_mask_spikes[:, i])
        spike_gen.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikeTimeList)

    input_mask_id_to_spike_gen_map = {}
    input_mask_id_to_spike_gen_map[input_mask_core.id] = spike_gen
    
    # This piece of code is so that in the case of the non-optimized layout
    if len(input_mask_relay_cores) == 0:
        return spike_gen, input_mask_id_to_spike_gen_map

    id_to_relay_core_map = {core.id: core for core in input_mask_relay_cores}
    id_to_relay_neuron_group_map = {}

    for core in input_mask_relay_cores:
        relay_params = {
            'n_reg': max_n_sentences,
            'n_adap': 0,
            'tau_V': 0.,
            'tau_I': 0.,
            'scaled_thr':  1,
            # These parameters are just specified for the sake of it. the are irrelevant
            'weight_aux':  0,
            'weight_exp': 0,
            'n_refractory': 1,
            'tau_adaptation': 0,
            'beta_exp': 0
        }
        
        # print("Allocating input mask relay core: {}".format(core.id))

        # Here create LIF Neuron Group with NO connections, the connections will be created later
        ALIF_neuron_group = RecurrentLsnn(relay_params, net, core.id)
        ALIF_neuron_group.createRecurrentNeuronGroup()

        id_to_relay_neuron_group_map[core.id] = ALIF_neuron_group

    # Edit connection weights to account for a modified value of max_n_sentences
    assert all(len(x) == 1 for x in input_mask_to_relay_connections)
    input_mask_to_relay_connections = list(input_mask_to_relay_connections)
    for i, conn_tuple in enumerate(input_mask_to_relay_connections):
        assert all(x.presyn_core == input_mask_core.id for x in conn_tuple)
        connection = conn_tuple[0]
        input_mask_to_relay_connections[i] = (LSNNConnectionTuple(presyn_core=connection.presyn_core,
                                                                  postsyn_core=connection.postsyn_core,
                                                                  mask=connection.mask[:max_n_sentences, :max_n_sentences],
                                                                  weights=connection.weights[:max_n_sentences, :max_n_sentences],
                                                                  delays=connection.delays[:max_n_sentences, :max_n_sentences],
                                                                  weight_exp=connection.weight_exp,
                                                                  sentence_ind=connection.sentence_ind,
                                                                  fanout_copy_ind=connection.fanout_copy_ind),)

    # Connect using modified connections
    for conn_tuple in input_mask_to_relay_connections:
        connection = conn_tuple[0]
        connect_spike_generator_to_neuron_group(spike_generator=spike_gen,
                                                neuron_group=id_to_relay_neuron_group_map[connection.postsyn_core],
                                                mask=connection.mask,
                                                weights=connection.weights,
                                                delays=connection.delays,
                                                weight_exp=connection.weight_exp)

    return spike_gen, id_to_relay_neuron_group_map


def create_relnet_output_spike_generators(relnet_core_connection_array, relnet_output_spikes, net):
    max_n_sentences = relnet_core_connection_array.shape[0]

    relnet_id_to_output_spike_gen_map = {}
    relnet_output_spike_gen_array = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)

    for sentence1 in range(max_n_sentences):
        for sentence2 in range(sentence1, max_n_sentences):
            current_relnet_instance = relnet_core_connection_array[sentence1, sentence2]  # type: RelnetInitInstanceStruct

            output_spike_gen_tuple = []

            # Create the cores corresponding to the current relnet instance
            for core in current_relnet_instance.cores:  # type: RelnetInitCoreTuple
                num_input = core.end - core.start
                spike_gen = net.createSpikeGenProcess(numPorts=num_input)

                # configure spikes for spike generator
                for i in range(0, num_input):
                    spikeTimeList = convertInputSpikesToSpikeTimes(relnet_output_spikes[sentence1, sentence2, :, core.start + i])
                    spike_gen.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikeTimeList)

                output_spike_gen_tuple.append(spike_gen)
                relnet_id_to_output_spike_gen_map[core.id] = spike_gen

            output_spike_gen_tuple = tuple(output_spike_gen_tuple)
            relnet_output_spike_gen_array[sentence1, sentence2] = output_spike_gen_tuple

    return relnet_output_spike_gen_array, relnet_id_to_output_spike_gen_map


def create_relnet_output_relay_neurons(relnet_core_connection_array, relnet_output_spikes, net, max_n_sentences):

    relnet_id_to_output_relay_neuron_map = {}
    relnet_output_relay_neuron_array = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)

    for sentence1 in range(max_n_sentences):
        for sentence2 in range(sentence1, max_n_sentences):
            current_relnet_instance = relnet_core_connection_array[sentence1, sentence2]  # type: RelnetInitInstanceStruct

            output_relay_neuron_tuple = []

            # Create the cores corresponding to the current relnet instance
            for core in current_relnet_instance.cores:  # type: RelnetInitCoreTuple
                num_input = core.end - core.start
                spike_gen = net.createSpikeGenProcess(numPorts=num_input)

                # configure spikes for spike generator
                for i in range(0, num_input):
                    spikeTimeList = convertInputSpikesToSpikeTimes(relnet_output_spikes[sentence1, sentence2, :, core.start + i])
                    spike_gen.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikeTimeList)

                # Create relay group with input from above spike generator
                relay_parameters = {
                    'n_adap': 0,
                    'n_reg': num_input,
                    'tau_V': 20.,
                    'tau_I': 0.,
                    'scaled_thr': 1,
                    'weight_aux': 0,
                    'weight_exp': 0,
                    'n_refractory': 1,
                    'tau_adaptation': 0,
                    'beta_exp': 0,
                }
                relay_neuron_group = RecurrentLsnn(relay_parameters, net, core.id)
                relay_neuron_group.createRecurrentNeuronGroup()

                # connect
                connect_function = connect_spike_generator_to_neuron_group

                connect_function(
                    spike_gen,
                    relay_neuron_group,
                    mask=np.eye(num_input),
                    weights=np.eye(num_input)*5,
                    delays=np.zeros((num_input, num_input)))

                output_relay_neuron_tuple.append(relay_neuron_group)
                relnet_id_to_output_relay_neuron_map[core.id] = relay_neuron_group

            output_relay_neuron_tuple = tuple(output_relay_neuron_tuple)
            relnet_output_relay_neuron_array[sentence1, sentence2] = output_relay_neuron_tuple

    return relnet_output_relay_neuron_array, relnet_id_to_output_relay_neuron_map


def create_final_MLP_output_spike_generators(final_MLP_core_connection, final_MLP_output_spikes, net):

    final_MLP_id_to_output_spike_gen_map = {}

    final_MLP_output_spike_gen_tuple = []

    # Create the cores corresponding to the current final_MLP instance
    for core in final_MLP_core_connection.cores:  # type: RelnetInitCoreTuple
        num_input = core.end - core.start
        spike_gen = net.createSpikeGenProcess(numPorts=num_input)

        # configure spikes for spike generator
        for i in range(0, num_input):
            spikeTimeList = convertInputSpikesToSpikeTimes(final_MLP_output_spikes[:, core.start + i])
            spike_gen.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikeTimeList)

        final_MLP_output_spike_gen_tuple.append(spike_gen)
        final_MLP_id_to_output_spike_gen_map[core.id] = spike_gen

    final_MLP_output_spike_gen_tuple = tuple(final_MLP_output_spike_gen_tuple)

    return final_MLP_output_spike_gen_tuple, final_MLP_id_to_output_spike_gen_map


def perform_LSNN_placement(LSNN_core_connection_array, LSNN_cell_data, lsnn_input_spikes, relay_weight, net, max_n_sentences=0):

    probesReg = []
    probesMain = []
    probesAux = []

    n_time_steps = lsnn_input_spikes.shape[-2]
    LSNN_size = lsnn_input_spikes.shape[-1]

    if isinstance(LSNN_core_connection_array, LSNNCopyStruct):
        # This is the case of the question LSNN
        n_instances = 0
        n_relay_fanout_copies = len(LSNN_core_connection_array.cores.relay)
        assert lsnn_input_spikes.shape == (n_time_steps, LSNN_size)
    else:
        # This is the case of the sentence LSNN
        n_instances, = LSNN_core_connection_array.shape
        n_relay_fanout_copies = len(LSNN_core_connection_array[0].cores.relay)
        assert lsnn_input_spikes.shape == (n_instances, n_time_steps, LSNN_size)
        
    if max_n_sentences > 0:
        assert n_instances > 0, "max_n_sentences specified for question LSNN"
        n_instances = max_n_sentences

    actual_n_instances = n_instances
    if n_instances == 0:
        n_instances = 1
        temp = np.ndarray(1, dtype=object)
        temp[0] = LSNN_core_connection_array
        LSNN_core_connection_array = temp
        lsnn_input_spikes = np.expand_dims(lsnn_input_spikes, 0)
        
    spike_generator_array = np.ndarray((n_instances,), dtype=object)
    lsnn_neuron_group_array = np.ndarray((n_instances,), dtype=object)
    relay_neuron_group_array = np.ndarray((n_instances, n_relay_fanout_copies), dtype=object)
    lsnn_id_to_neuron_group_map = {}
    lsnn_id_to_relay_neuron_group_map = {}
    lsnn_id_to_spike_gen_map = {}
    lsnn_id_to_relay_core_map = {}
    lsnn_id_to_lsnn_core_map = {}

    for instance_ind in range(n_instances):
        lsnn_neuron_group_tuple = []
        spike_generator_tuple = []  # TODO: Confirm this
        current_LSNN_copy = LSNN_core_connection_array[instance_ind]  # type: LSNNCopyStruct

        for core in current_LSNN_copy.cores.lsnn:  # type: LSNNCoreTuple

            scaled_thr = (LSNN_cell_data['scaled_thr'] / LSNN_cell_data['scale_thr'])[0]
            weight_aux = (LSNN_cell_data["scaled_beta"]/(2**(6+LSNN_cell_data["beta_exp"])))[-1]

            lsnn_params = {
                'n_reg': max(min(LSNN_cell_data['n_reg'] - core.start, core.end - core.start), 0),
                'n_adap': max(min(core.end - LSNN_cell_data['n_reg'], core.end - core.start), 0),
                'tau_V': LSNN_cell_data['tau'],
                'tau_I': LSNN_cell_data['tau_I'],
                'scaled_thr':  scaled_thr,
                'weight_aux':  weight_aux,
                'weight_exp': LSNN_cell_data['weight_exp'],
                'n_refractory': LSNN_cell_data['n_refractory'],
                'tau_adaptation': LSNN_cell_data['tau_adaptation'],
                'beta_exp': LSNN_cell_data['beta_exp']
            }

            # Here create LIF Neuron Group with NO connections, the connections will be created later
            ALIF_neuron_group = RecurrentLsnn(lsnn_params, net, core.id)
            ALIF_neuron_group.createRecurrentNeuronGroup()

            lsnn_id_to_neuron_group_map[core.id] = ALIF_neuron_group
            lsnn_id_to_lsnn_core_map[core.id] = core
            lsnn_neuron_group_tuple.append(ALIF_neuron_group)

        lsnn_neuron_group_tuple = tuple(lsnn_neuron_group_tuple)

        # Here we need to create the input generators. Not entirely sure of this so leaving this for now
        assert len(current_LSNN_copy.cores.input) == 1, \
            "Currently Input is implemented on the lakemount so there's no point in generating it on multiple cores"
        current_input_core = current_LSNN_copy.cores.input[0]
        input_params = {
            'n_input': None,
            'input_spikes': lsnn_input_spikes[instance_ind]
        }

        # Using the above input parameters create the spike generator for the current copy of the
        input_len, num_input = input_params["input_spikes"].shape  # amount of time steps for input, num input neurons

        # TODO: maybe only use one spike generator process with 180 * 20 amount of ports
        spike_gen = net.createSpikeGenProcess(numPorts=num_input)

        # configure spikes for spike generator
        for i in range(0, num_input):
            spikeTimeList = convertInputSpikesToSpikeTimes(input_params["input_spikes"][:, i])
            spike_gen.addSpikes(spikeInputPortNodeIds=i, spikeTimes=spikeTimeList)

        current_spike_gen = spike_gen
        spike_generator_tuple.append(current_spike_gen)
        spike_generator_tuple = tuple(spike_generator_tuple)
        lsnn_id_to_spike_gen_map[current_input_core.id] = current_spike_gen

        lsnn_neuron_group_array[instance_ind] = lsnn_neuron_group_tuple
        spike_generator_array[instance_ind] = spike_generator_tuple

        # Set out each fanout copy of the relay neurons
        for fanout_copy_ind in range(n_relay_fanout_copies):

            relay_neuron_group_tuple = []

            # Here we create the neuron groups pertaining to the relay neurons
            current_relay_cores = current_LSNN_copy.cores.relay[fanout_copy_ind]
            for core in current_relay_cores:  # type: LSNNCoreTuple
                relay_params = {
                    'n_reg': core.end - core.start,
                    'n_adap': 0,
                    'tau_V': 0.,
                    'tau_I': 0.,
                    'scaled_thr':  1,
                    # These parameters are just specified for the sake of it. the are irrelevant
                    'weight_aux':  0,
                    'weight_exp': 0,
                    'n_refractory': 1,
                    'tau_adaptation': 0,
                    'beta_exp': 0
                }

                # Here create LIF Neuron Group with NO connections, the connections will be created later
                ALIF_neuron_group = RecurrentLsnn(relay_params, net, core.id)
                ALIF_neuron_group.createRecurrentNeuronGroup()

                lsnn_id_to_relay_core_map[core.id] = core
                lsnn_id_to_relay_neuron_group_map[core.id] = ALIF_neuron_group
                relay_neuron_group_tuple.append(ALIF_neuron_group)

            relay_neuron_group_tuple = tuple(relay_neuron_group_tuple)
            relay_neuron_group_array[instance_ind, fanout_copy_ind] = relay_neuron_group_tuple

        # Here we need to create the connection between the spike generator and the LSNN cores
        for connection in current_LSNN_copy.connections.input_to_lsnn:  # type: LSNNConnectionTuple
            spike_generator_index = connection.presyn_core
            lsnn_core_index = connection.postsyn_core
            connect_spike_generator_to_neuron_group(lsnn_id_to_spike_gen_map[spike_generator_index],
                                                    lsnn_id_to_neuron_group_map[lsnn_core_index],
                                                    mask=connection.mask,
                                                    weights=connection.weights,
                                                    delays=connection.delays,
                                                    weight_exp=connection.weight_exp)

        # Here we need to perform the connections between the cores of the above neuron_group_tuple
        # Performing recurrent connections
        for connection in current_LSNN_copy.connections.lsnn_to_lsnn:  # type: LSNNConnectionTuple
            presyn_core_ind = connection.presyn_core
            postsyn_core_ind = connection.postsyn_core

            connect_neuron_groups(lsnn_id_to_neuron_group_map[presyn_core_ind],
                                  lsnn_id_to_neuron_group_map[postsyn_core_ind],
                                  mask=connection.mask,
                                  weights=connection.weights,
                                  delays=connection.delays,
                                  weight_exp=connection.weight_exp)

        # Here we connect the LSNN to the relay neurons
        for connection in current_LSNN_copy.connections.lsnn_to_relay:  # type: LSNNConnectionTuple
            presyn_core_ind = connection.presyn_core
            postsyn_core_ind = connection.postsyn_core

            # presyn_core = lsnn_id_to_lsnn_core_map[presyn_core_ind]
            # postsyn_core = lsnn_id_to_relay_core_map[postsyn_core_ind]

            # presyn_neuron_arr = np.arange(presyn_core.start, presyn_core.end)
            # postsyn_neuron_arr = np.arange(postsyn_core.start, postsyn_core.end)

            # neurons_connected = np.where(connection.mask | (connection.weights != 0))
            # neurons_connected = [presyn_neuron_arr[neurons_connected[0]],
            #                      postsyn_neuron_arr[neurons_connected[1]]]
            # neurons_connected = np.stack(neurons_connected, axis=-1)

            # print("Connections made:")
            # print('\n'.join("{} -> {}".format(x[0], x[1]) for x in neurons_connected))

            connect_neuron_groups(lsnn_id_to_neuron_group_map[presyn_core_ind],
                                  lsnn_id_to_relay_neuron_group_map[postsyn_core_ind],
                                  mask=connection.mask,
                                  weights=connection.weights,
                                  delays=connection.delays,
                                  weight_exp=connection.weight_exp)

    if actual_n_instances == 0:
        spike_generator_array = spike_generator_array[0]
        lsnn_neuron_group_array = lsnn_neuron_group_array[0]
        relay_neuron_group_array = np.reshape(relay_neuron_group_array, (n_relay_fanout_copies,))

    return (lsnn_id_to_neuron_group_map, lsnn_neuron_group_array,
            lsnn_id_to_relay_neuron_group_map, relay_neuron_group_array,
            lsnn_id_to_spike_gen_map, spike_generator_array)


def perform_relnet_placement(input_id_to_neuron_group_map,
                             relnet_core_connection_array, relnet_cell_data, net, n_sentence1=None, n_sentence2=None):
    
    max_n_sentences = n_sentence1
    if n_sentence1 is None:
        n_sentence1 = relnet_core_connection_array.shape[0]
    if n_sentence2 is None:
        n_sentence2 = relnet_core_connection_array.shape[0]
        
    relnet_id_to_neuron_group_map = {}
    relnet_neuron_group_array = np.ndarray((n_sentence1, n_sentence2), dtype=object)
    relnet_params = {
        'n_adap': 0,
        'tau_V': relnet_cell_data['tau'],
        'tau_I': relnet_cell_data['tau_I'],
        'scaled_thr': (relnet_cell_data['scaled_thr'] / relnet_cell_data['scale_thr'])[0],
        'weight_aux': 0.,
        'weight_exp': relnet_cell_data['weight_exp'],
        'n_refractory': relnet_cell_data['n_refractory'],
        'tau_adaptation': 0,
        'beta_exp': 0,
    }

    from collections import defaultdict
    input_id_to_output_axon_map = defaultdict(lambda: 0)
    for sentence1 in range(0, n_sentence1):
         for sentence2 in range(sentence1, n_sentence2):
            # print("Configuring Relnet for sentence pair {}, {}".format(sentence1, sentence2))
            current_relnet_instance = relnet_core_connection_array[sentence1, sentence2]  # type: RelnetInitInstanceStruct
            relnet_neuron_group_tuple = []
            # Create the cores corresponding to the current relnet instance
            for core in current_relnet_instance.cores:  # type: RelnetInitCoreTuple
                relnet_params['n_reg'] = core.end - core.start
                relnet_neuron_group = RecurrentLsnn(relnet_params, net, core.id)
                relnet_neuron_group.createRecurrentNeuronGroup()
                relnet_neuron_group_tuple.append(relnet_neuron_group)
                relnet_id_to_neuron_group_map[core.id] = relnet_neuron_group
            relnet_neuron_group_tuple = tuple(relnet_neuron_group_tuple)
            # A special case for when the input is the lsnn's
            if isinstance(current_relnet_instance.connections, RelnetInitConnectionsStruct):
                connection_tuple = ()
                connection_tuple += current_relnet_instance.connections.sentence1_to_relnet
                connection_tuple += current_relnet_instance.connections.sentence2_to_relnet
                connection_tuple += current_relnet_instance.connections.question_to_relnet
                # print("Connections before clipping")
                
                def print_conn(conn):

                    # print(relnet_id_to_neuron_group_map[conn.postsyn_core])
                    return ("Hello World")
                    return ("presyn_core = {}\n".format(conn.presyn_core) +
                            "postsyn_core = {}\n".format(conn.postsyn_core) +
                            "presyn_core size = {}\n".format(input_id_to_neuron_group_map[conn.presyn_core].recurrentNeuronGroup.numNodes) + 
                            "postsyn_core size = {}\n".format(relnet_id_to_neuron_group_map[conn.postsyn_core].recurrentNeuronGroup.numNodes) + 
                            "mask = {}\n".format(conn.mask.shape) +
                            "weights = {}\n".format(conn.weights.shape) +
                            "delays = {}\n".format(conn.delays.shape) +
                            "presyn_core size = {}\n".format("trash") + # input_id_to_neuron_group_map[conn.presyn_core].recurrentNeuronGroup.numNodes
                            "postsyn_core size = {}\n".format("trash") + # relnet_id_to_neuron_group_map[conn.postsyn_core].recurrentNeuronGroup.numNodes
                            #"weight_exp = {}\n".format(conn.weight_exp) +
                            #"sentence1_ind = {}\n".format(conn.sentence1_ind) +
                            "sentence2_ind = {}\n".format(conn.sentence2_ind))
                
                
                # print("\n".join(print_conn(x) for x in current_relnet_instance.connections.mask_to_relnet))
                new_mask_conn_tuple = []
                for connection in current_relnet_instance.connections.mask_to_relnet:
                    new_mask_conn_tuple.append(RelnetInitConnectionTuple(presyn_core=connection.presyn_core,
                                                                         postsyn_core=connection.postsyn_core,
                                                                         mask=connection.mask[:max_n_sentences],
                                                                         weights=connection.weights[:max_n_sentences],
                                                                         delays=connection.delays[:max_n_sentences],
                                                                         weight_exp=connection.weight_exp,
                                                                         sentence1_ind=connection.sentence1_ind,
                                                                         sentence2_ind=connection.sentence2_ind))
                new_mask_conn_tuple = tuple(new_mask_conn_tuple)
                # print("Connections after clipping")
                # print("\n".join(print_conn(x) for x in new_mask_conn_tuple))
                connection_tuple += new_mask_conn_tuple
                # print("Final Connections")
                # print("\n".join(print_conn(x) for x in connection_tuple))
                # for ctuple in current_relnet_instance.connections:
                #     connection_tuple = connection_tuple + ctuple
            else:
                connection_tuple = current_relnet_instance.connections
            has_printed = False
            # Create connections from input to relnet
            for connection in connection_tuple:  # type: RelnetInitConnectionTuple
                presyn_core_ind = connection.presyn_core
                postsyn_core_ind = connection.postsyn_core
                input_id_to_output_axon_map[presyn_core_ind] += np.count_nonzero(np.sum(connection.weights, axis=1))
                if isinstance(input_id_to_neuron_group_map[presyn_core_ind], RecurrentLsnn):
                    connect_function = connect_neuron_groups
                    if not has_printed:
                        # print("Connecting Neuron Groups")
                        has_printed = True
                else:
                    connect_function = connect_spike_generator_to_neuron_group

                actual_presyn_core_id = input_id_to_neuron_group_map[presyn_core_ind].compileParams.logicalCoreId if isinstance(input_id_to_neuron_group_map[presyn_core_ind], RecurrentLsnn) else 99999
                actual_postsyn_core_id = relnet_id_to_neuron_group_map[postsyn_core_ind].compileParams.logicalCoreId
                actual_n_connections = np.count_nonzero((connection.mask != 0) | (connection.weights != 0))
                weights_hash = hash(connection.weights.data.tobytes())
                delays_hash = hash(connection.delays.data.tobytes())
                weight_delays_hash = hash((weights_hash, delays_hash))

                # print("Performing {:<4d} Connections from Cores {:>5d} -> {:<5d}, hash: {}".format(int(actual_n_connections), actual_presyn_core_id, actual_postsyn_core_id, weight_delays_hash))

                connect_function(
                    input_id_to_neuron_group_map[presyn_core_ind],
                    relnet_id_to_neuron_group_map[postsyn_core_ind],
                    mask=connection.mask,
                    weights=connection.weights,
                    delays=connection.delays,
                    weight_exp=connection.weight_exp)
            # print('\n'.join('{}: {}'.format(k, v) for k, v in sorted(input_id_to_output_axon_map.items())))
            relnet_neuron_group_array[sentence1, sentence2] = relnet_neuron_group_tuple
    return relnet_neuron_group_array, relnet_id_to_neuron_group_map


def perform_translation_layer_placement(relnet_final_id_to_neuron_group_map,
                                        translation_layer_core_connection,
                                        translation_layer_cell_data, net, 
                                        n_sentence1=None, n_sentence2=None):

    if n_sentence1 is None:
        n_sentence1 = translation_layer_core_connection.connections.shape[0]
    if n_sentence2 is None:
        n_sentence2 = translation_layer_core_connection.connections.shape[0]
    
    translation_layer_params = {
        'n_adap': 0,
        'tau_V': translation_layer_cell_data['tau'],
        'tau_I': translation_layer_cell_data['tau_I'],
        'scaled_thr': (translation_layer_cell_data['scaled_thr'] / translation_layer_cell_data['scale_thr'])[0],
        'weight_aux': 0.,
        'weight_exp': translation_layer_cell_data['weight_exp'],
        'n_refractory': translation_layer_cell_data['n_refractory'],
        'tau_adaptation': 0,
        'beta_exp': 0,
    }

    translation_layer_neuron_group_tuple = []
    translation_layer_id_to_neuron_group_map = {}
    for core in translation_layer_core_connection.cores:
        translation_layer_params['n_reg'] = core.end - core.start
        translation_layer_neuron_group = RecurrentLsnn(translation_layer_params, net, core.id)
        translation_layer_neuron_group.createRecurrentNeuronGroup()

        translation_layer_neuron_group_tuple.append(translation_layer_neuron_group)
        translation_layer_id_to_neuron_group_map[core.id] = translation_layer_neuron_group

    translation_layer_neuron_group_tuple = tuple(translation_layer_neuron_group_tuple)

    has_printed = False
    for sentence1 in range(n_sentence1):
        for sentence2 in range(sentence1, n_sentence2):
            current_connection_tuple = translation_layer_core_connection.connections[sentence1, sentence2]
            # print('placing connections from relational function applied to sentence pair {}, {}'.format(sentence1, sentence2))
            for connection in current_connection_tuple:
                presyn_core_ind = connection.presyn_core
                postsyn_core_ind = connection.postsyn_core

                assert np.all(connection.weights[connection.mask == 0] == 0)
                if isinstance(relnet_final_id_to_neuron_group_map[presyn_core_ind], RecurrentLsnn):
                    connection_func = connect_neuron_groups
                    if not has_printed:
                        # print('Connecting From RecurrentLSNN in Translation Layer')
                        has_printed = True
                else:
                    connection_func = connect_spike_generator_to_neuron_group
                    if not has_printed:
                        # print('Connecting From SpikeGen to Translation Layer')
                        has_printed = True

                connection_func(
                    relnet_final_id_to_neuron_group_map[presyn_core_ind],
                    translation_layer_id_to_neuron_group_map[postsyn_core_ind],
                    mask=connection.mask,
                    weights=connection.weights,
                    delays=connection.delays,
                    weight_exp=connection.weight_exp)

    return translation_layer_neuron_group_tuple, translation_layer_id_to_neuron_group_map


def perform_final_MLP_placement(input_id_to_neuron_group_map,
                                final_MLP_core_connection,
                                final_MLP_cell_data, net):

    final_MLP_params = {
        'n_adap': 0,
        'tau_V': final_MLP_cell_data['tau'],
        'tau_I': final_MLP_cell_data['tau_I'],
        'scaled_thr': (final_MLP_cell_data['scaled_thr'] / final_MLP_cell_data['scale_thr'])[0],
        'weight_aux': 0.,
        'weight_exp': final_MLP_cell_data['weight_exp'],
        'n_refractory': final_MLP_cell_data['n_refractory'],
        'tau_adaptation': 0,
        'beta_exp': 0,
    }

    final_MLP_neuron_group_tuple = []
    final_MLP_id_to_neuron_group_map = {}
    for core in final_MLP_core_connection.cores:
        final_MLP_params['n_reg'] = core.end - core.start
        final_MLP_neuron_group = RecurrentLsnn(final_MLP_params, net, core.id)
        final_MLP_neuron_group.createRecurrentNeuronGroup()

        final_MLP_neuron_group_tuple.append(final_MLP_neuron_group)
        final_MLP_id_to_neuron_group_map[core.id] = final_MLP_neuron_group

    final_MLP_neuron_group_tuple = tuple(final_MLP_neuron_group_tuple)

    for connection in final_MLP_core_connection.connections:
        presyn_core_ind = connection.presyn_core
        postsyn_core_ind = connection.postsyn_core

        if isinstance(input_id_to_neuron_group_map[presyn_core_ind], RecurrentLsnn):
            connection_func = connect_neuron_groups
        else:
            connection_func = connect_spike_generator_to_neuron_group

        connection_func(
            input_id_to_neuron_group_map[presyn_core_ind],
            final_MLP_id_to_neuron_group_map[postsyn_core_ind],
            mask=connection.mask,
            weights=connection.weights,
            delays=connection.delays,
            weight_exp=connection.weight_exp)

    return final_MLP_neuron_group_tuple, final_MLP_id_to_neuron_group_map


def perform_readout_placement(input_id_to_neuron_group_map,
                              readout_core_connection,
                              readout_cell_data, net):
    # FIXME: Store the readout tau_I
    readout_params = {
        'n_reg': 0,
        'n_adap': 0,
        'tau_V': 2**12,
        'tau_I': 20.,
        'scaled_thr': 1,
        'weight_aux': 0.,
        'weight_exp': readout_cell_data['weight_exp'],
        'n_refractory': 1,
        'tau_adaptation': 0,
        'beta_exp': 0,
    }

    readout_neuron_group_tuple = []
    readout_id_to_neuron_group_map = {}
    for core in readout_core_connection.cores:
        readout_params['n_reg'] = core.end - core.start
        readout_neuron_group = RecurrentLsnn(readout_params, net, core.id)
        readout_neuron_group.createOutputNeuronGroup()

        readout_neuron_group_tuple.append(readout_neuron_group)
        readout_id_to_neuron_group_map[core.id] = readout_neuron_group

    readout_neuron_group_tuple = tuple(readout_neuron_group_tuple)

    for connection in readout_core_connection.connections:
        presyn_core_ind = connection.presyn_core
        postsyn_core_ind = connection.postsyn_core

        if isinstance(input_id_to_neuron_group_map[presyn_core_ind], RecurrentLsnn):
            connection_func = connect_neuron_groups_to_output_groups
        else:
            connection_func = connect_spike_generator_to_output_neuron_group

        connection_func(
            input_id_to_neuron_group_map[presyn_core_ind],
            readout_id_to_neuron_group_map[postsyn_core_ind],
            mask=connection.mask,
            weights=connection.weights,
            delays=connection.delays,
            weight_exp=connection.weight_exp)

    return readout_neuron_group_tuple, readout_id_to_neuron_group_map
