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

MAX_SYNAPSES_FF = 40000
MAX_SYNAPSES_LSNN = 60000
MAX_SYNAPSES_TRANSLATION = 30000

INPUT_AXON_LIMIT = 4096
OUTPUT_AXON_LIMIT = 2048
OUTPUT_AXON_LIMIT_WITHIN_CHIP = 4096
OUTPUT_AXON_PER_NEURON_LIMIT = 511  # This is a strange constraint that is undocumented
N_CORES_PER_CHIP = 128
MAX_TOTAL_NEURONS_IN_CORE = 1024
MAX_N_SENTENCES = 20

WEIGHT_SHARING = False

# A placement block is as follows:
# It consists of a layer or several copies of layers with identical input weights
# In the placement of a placement block we can have weight sharing

# Each neuron is identified by a core index and a neuron index in that core (core, index)
# Each Weight matrix has associated to it a list of input neurons and output neurons
# Each element of the list of input (output) neurons corresponds to one row (column) of the
# weight matrix.
# The element can be a scalar (meaning that the input (output) neuron is common) or an
# array of length = n_rows (n_columns)


def get_weight_input_output_for_relnet_input():

    # need the start core index
    pass


def get_minimum_n_cores_for_LSNN_placement(relnet_input_neurons_per_core, relnet_input_layers_per_core):
    """
    :param relnet_input_neurons_per_core: int array of size MAX_N_SENTENCES, such that
        relnet_input_neurons_per_core[i] - the neurons per core for the relnet layers that
        have sentence2 = i
    :param relnet_input_layers_per_core: int array of size MAX_N_SENTENCES, such that
        relnet_input_layers_per_core[i] - the layers per core for the relnet layers that
        have sentence2 = i
    """
    pass


def get_minimum_n_cores_for_only_LSNN_placement_no_ws(LSNN_size, input_dim, n_relay_copies, use_cores_for_input=False):
    """
    Unlike the function get_minimum_n_cores_for_LSNN_placement_no_ws, this function
    only calculates the parameters for the placement of the input and LSNN layers
    given the number of relay copies.

    It doesn't take into account the relational network nor even how many
    instances of the LSNN are there. It just returns the number of cores for
    one instance
    """

    # ==================================================
    # Here we place the LSNN Network
    # ==================================================
    # Here we don't make copies for fanout as the fanout is the one-one connections to the relay network
    # The constraints to deal with are
    # 1.  The synaptic Memory Limit
    # 2.  The fanout to the relay network copies
    # 3.  The fanin from the input is always within limits

    net_fanout = n_relay_copies

    # Assuming that the neuron limit of MAX_TOTAL_NEURONS_IN_CORE is never reached
    # Assuming that the input axonal limit is never reached
    max_neurons_per_core = min((MAX_SYNAPSES_LSNN // (LSNN_size + input_dim), LSNN_size))

    neurons_per_core_array = np.arange(1, max_neurons_per_core+1).astype(np.int64)

    # cores to implement one copy of LSNN
    n_cores_array = np.ceil(LSNN_size / neurons_per_core_array)
    # max_fanout_per_core = (total axons - axons used up for recurrent conns) / number of neurons per core
    max_fanout_per_core_array = np.floor((OUTPUT_AXON_LIMIT - neurons_per_core_array*(n_cores_array-1)) / neurons_per_core_array)

    if use_cores_for_input:
        output_axon_limit = OUTPUT_AXON_LIMIT
    else:
        output_axon_limit = 2**23

    # Filter based on whether the max fanout allows fanning out to the copies of the relay network
    valid_indices = max_fanout_per_core_array >= net_fanout
    neurons_per_core_array = neurons_per_core_array[valid_indices]
    n_cores_array = n_cores_array[valid_indices]
    max_fanout_per_core_array = max_fanout_per_core_array[valid_indices]

    opt_index = np.argmin(n_cores_array)
    opt_neurons_per_core = neurons_per_core_array[opt_index]
    opt_n_cores = n_cores_array[opt_index]

    # calculating the number of cores required for input neurons (constraint is the output axons)
    input_fanout_required_array = opt_n_cores
    max_fanout_per_core_for_input = np.floor(output_axon_limit / input_dim)
    n_input_cores = np.ceil(input_fanout_required_array / max_fanout_per_core_for_input).astype(np.int64)

    input_neurons_per_core = np.ceil(input_dim / n_input_cores).astype(np.int64)

    total_cores = opt_n_cores + n_input_cores
    return total_cores, input_neurons_per_core, opt_neurons_per_core


def get_minimum_n_cores_for_LSNN_placement_no_ws(LSNN_size, input_dim, relnet_cores_per_copy, fanout_to_relnets, n_instances,
                                                 use_cores_for_input=False,
                                                 create_masking_core=True):
    """
    :param relnet_cores_per_copy: int that gives the number of cores required to implement
        each copy of the first layer of the relational network
    """
    assert not WEIGHT_SHARING

    n_cores_total = 0
    # Here we have the following constraints to deal with:
    # 1.  The input axon count is never reached
    # 2.  The synaptic memory limit is never reached as the connections to the relay network are one-one
    # 3.  The relay network fans out to the relational network and so must take into account the output axons
    # 4.  The number of neurons per core cannot exceed MAX_TOTAL_NEURONS_IN_CORE
    # 5.  The strange condition that the fanout per neuron cannot exceed a certain value
    # 
    # max_relay_neurons_per_core - affected only by MAX_TOTAL_NEURONS_IN_CORE
    # n_copies_for_relay_network - affected by the fanout requirements to the relational network

    net_relay_fanout = fanout_to_relnets
    max_relay_neurons_per_core = min(MAX_TOTAL_NEURONS_IN_CORE, LSNN_size)
    relay_neurons_per_core_array = np.arange(1, max_relay_neurons_per_core).astype(np.int64)  # Each relay core has at-most as many neurons as specified here
    n_relay_cores_per_copy_array = np.ceil(LSNN_size / relay_neurons_per_core_array).astype(np.int64)  # Each relay network copy requires as many cores as specified here
    max_fanout_per_relay_core_array = OUTPUT_AXON_LIMIT // (relay_neurons_per_core_array*relnet_cores_per_copy)  # This is the maximum possibly fanout from each core (assuming it has relay_neurons_per_core_array neurons)
    n_relay_copies_array = np.ceil(net_relay_fanout / max_fanout_per_relay_core_array).astype(np.int64)  # It will require this number of copies of the relay network in order to satisfy the requirement

    # Filtering the values here to minimize the number of relay network cores
    valid_indices = max_fanout_per_relay_core_array*relnet_cores_per_copy <= OUTPUT_AXON_PER_NEURON_LIMIT  # take into account condition 5.
    relay_neurons_per_core_array = relay_neurons_per_core_array[valid_indices]
    n_relay_cores_per_copy_array = n_relay_cores_per_copy_array[valid_indices]
    max_fanout_per_relay_core_array = max_fanout_per_relay_core_array[valid_indices]
    n_relay_copies_array = n_relay_copies_array[valid_indices]
    n_relay_cores_total_array = n_relay_cores_per_copy_array * n_relay_copies_array

    opt_index = np.argmin(n_relay_cores_total_array)
    opt_relay_neurons_per_core = relay_neurons_per_core_array[opt_index]
    opt_n_relay_cores_per_copy = n_relay_cores_per_copy_array[opt_index]
    opt_n_relay_copies = n_relay_copies_array[opt_index]
    opt_n_relay_cores_total = n_relay_cores_total_array[opt_index]

    total_cores, input_neurons_per_core, opt_neurons_per_core = \
        get_minimum_n_cores_for_only_LSNN_placement_no_ws(LSNN_size, input_dim, opt_n_relay_copies, use_cores_for_input)

    # Add number of input cores (one copy for each copy of the LSNN needed to acheive fanout)
    n_cores_total = (opt_n_relay_cores_total + total_cores)*n_instances  # for input core

    # Here we create the core containing the masking neurons
    # Assumption: All the neurons fit in one core.
    if create_masking_core:
        assert fanout_to_relnets*n_instances*relnet_cores_per_copy <= OUTPUT_AXON_LIMIT, "It appears that all the masking neurons cannot be placed in a single core"

    return int(n_cores_total), int(input_neurons_per_core), int(opt_neurons_per_core), int(opt_relay_neurons_per_core), int(opt_n_relay_copies)


def get_minimum_n_cores_for_placement_block(max_layers_available, layer_size, max_layers_per_core_by_axon, max_neurons_per_core_by_memory):

    neurons_per_core_list = []
    layers_per_core_list = []

    max_neurons_per_core = min(max_neurons_per_core_by_memory, layer_size)
    max_layers_per_core = min(max_layers_per_core_by_axon, max_layers_available)

    assert max_neurons_per_core_by_memory <= MAX_TOTAL_NEURONS_IN_CORE

    for n in range(1, max_neurons_per_core+1):
        # Imposing constrant of total_neurons_in_core
        max_layers_by_neuron_limit = MAX_TOTAL_NEURONS_IN_CORE // n
        final_max_layers_per_core = min(max_layers_per_core, max_layers_by_neuron_limit)

        neurons_per_core_list.extend([n]*final_max_layers_per_core)
        layers_per_core_list.extend(range(1, final_max_layers_per_core+1))

    layers_per_core_array = np.array(layers_per_core_list)
    neurons_per_core_array = np.array(neurons_per_core_list)

    n_cores_array = np.ceil(max_layers_available / layers_per_core_array) * np.ceil(layer_size / neurons_per_core_array)
    argmin_n_cores = np.argmin(n_cores_array)
    return (n_cores_array[argmin_n_cores],
            neurons_per_core_array[argmin_n_cores],
            layers_per_core_array[argmin_n_cores])


def get_placement_param_list_relnet_input():

    layer_size = 256
    LSNN_sentence_size = 200
    LSNN_question_size = 200
    input_size = LSNN_sentence_size*2 + LSNN_question_size

    # The first layer of the relational network is partitioned into MAX_N_SENTENCES placement blocks
    # each placement block is indexed by the index of the sentence object fed to sentence2
    # Each placement block receives the same question and sentence1

    # max limits as determined by axon and memory limits
    # Axonal constraint (since only one sentence varies, new connections are only required for that)
    max_layers_per_core_by_axon = (INPUT_AXON_LIMIT - LSNN_sentence_size - LSNN_question_size) // LSNN_sentence_size
    # Memory constraint (600 dim input, shared weights)
    max_neurons_per_core_by_memory = min((MAX_SYNAPSES_FF // input_size, layer_size))

    min_n_cores_neurons_per_core = []
    min_n_cores_layers_per_core = []
    min_n_cores = []

    for sentence_ind in range(MAX_N_SENTENCES):
        # sentence1 and question are fixed, sentence2 varies from sentence_ind to MAX_N_SENTENCES - 1
        max_layers_available = MAX_N_SENTENCES - sentence_ind
        n_cores, neurons_per_core, layers_per_core = get_minimum_n_cores_for_placement_block(max_layers_available=max_layers_available,
                                                                                             layer_size=layer_size,
                                                                                             max_layers_per_core_by_axon=max_layers_per_core_by_axon,
                                                                                             max_neurons_per_core_by_memory=max_neurons_per_core_by_memory)
        min_n_cores.append(n_cores)
        min_n_cores_neurons_per_core.append(neurons_per_core)
        min_n_cores_layers_per_core.append(layers_per_core)

    return min_n_cores, min_n_cores_neurons_per_core, min_n_cores_layers_per_core


def get_placement_param_list_relnet_intermediate():

    layer_size = 256
    input_size = 256

    # max limits as determined by axon and memory limits
    # Axonal constraint input dimension = input_size
    max_layers_per_core_by_axon = INPUT_AXON_LIMIT // input_size
    # Memory constraint
    max_neurons_per_core_by_memory = min((MAX_SYNAPSES_FF // input_size, layer_size))

    assert max_neurons_per_core_by_memory <= MAX_TOTAL_NEURONS_IN_CORE

    # sentence1 and question are fixed, sentence2 varies from sentence_ind to MAX_N_SENTENCES - 1
    max_layers_available = (MAX_N_SENTENCES*(MAX_N_SENTENCES+1))//2
    (min_n_cores,
     min_n_cores_neurons_per_core,
     min_n_cores_layers_per_core) = get_minimum_n_cores_for_placement_block(max_layers_available=max_layers_available,
                                                                            layer_size=layer_size,
                                                                            max_layers_per_core_by_axon=max_layers_per_core_by_axon,
                                                                            max_neurons_per_core_by_memory=max_neurons_per_core_by_memory)

    return min_n_cores, min_n_cores_neurons_per_core, min_n_cores_layers_per_core


def get_placement_param_list_relnet_input_no_ws(layer_size, LSNN_sentence_size, LSNN_question_size):

    assert not WEIGHT_SHARING

    input_size = LSNN_sentence_size*2 + LSNN_question_size
    n_layers_total = (MAX_N_SENTENCES * (MAX_N_SENTENCES+1)) // 2

    # The first layer of the relational network is partitioned into MAX_N_SENTENCES placement blocks
    # each placement block is indexed by the index of the sentence object fed to sentence2
    # Each placement block receives the same question and sentence1

    # max limits as determined by axon and memory limits
    # Memory constraint (600 dim input, shared weights)
    max_neurons_per_core_by_memory = min((MAX_SYNAPSES_FF // input_size, layer_size))

    n_cores_per_layer = np.ceil(layer_size/max_neurons_per_core_by_memory)
    neurons_per_core = int(np.ceil(layer_size / n_cores_per_layer))

    assert np.ceil(layer_size / neurons_per_core) == np.ceil(layer_size / max_neurons_per_core_by_memory)
    n_cores = np.ceil(layer_size / neurons_per_core) * n_layers_total
    return int(neurons_per_core), int(n_cores)


def get_placement_param_list_relnet_intermediate_no_ws(layer_size, input_size):

    assert not WEIGHT_SHARING

    n_layers_total = (MAX_N_SENTENCES * (MAX_N_SENTENCES+1)) // 2

    # The first laer of the relational network is partitioned into MAX_N_SENTENCES placement blocks
    # each placement block is indexed by the index of the sentence object fed to sentence2
    # Each placement block receives the same question and sentence1

    # max limits as determined by axon and memory limits
    # Memory constraint (600 dim input, shared weights)
    max_neurons_per_core_by_memory = min((MAX_SYNAPSES_FF // input_size, layer_size))

    n_cores_per_layer = np.ceil(layer_size/max_neurons_per_core_by_memory)
    neurons_per_core = int(np.ceil(layer_size / n_cores_per_layer))

    assert np.ceil(layer_size / neurons_per_core) == np.ceil(layer_size / max_neurons_per_core_by_memory)
    n_cores = np.ceil(layer_size / neurons_per_core) * n_layers_total
    return int(neurons_per_core), int(n_cores)


def get_placement_param_list_translation_layer_one_one_no_ws(relnet_output_dim):

    assert not WEIGHT_SHARING

    n_relnet_layers = (MAX_N_SENTENCES * (MAX_N_SENTENCES+1)) // 2

    # The first laer of the relational network is partitioned into MAX_N_SENTENCES placement blocks
    # each placement block is indexed by the index of the sentence object fed to sentence2
    # Each placement block receives the same question and sentence1

    # max limits as determined by axon and memory limits
    # Memory constraint (600 dim input, shared weights)
    max_neurons_per_core_by_memory = min((MAX_SYNAPSES_FF // n_relnet_layers, relnet_output_dim))
    max_neurons_per_core_by_input_axons = min((INPUT_AXON_LIMIT // relnet_output_dim, relnet_output_dim))

    max_neurons_per_core = min((max_neurons_per_core_by_memory, max_neurons_per_core_by_input_axons))

    n_cores_per_layer = np.ceil(relnet_output_dim / max_neurons_per_core)
    neurons_per_core = int(np.ceil(relnet_output_dim / n_cores_per_layer))

    assert np.ceil(relnet_output_dim / neurons_per_core) == np.ceil(relnet_output_dim / max_neurons_per_core)
    return int(neurons_per_core), int(n_cores_per_layer)


def get_placement_param_list_final_MLP_no_ws(layer_size, input_size):

    assert not WEIGHT_SHARING

    # The first laer of the relational network is partitioned into MAX_N_SENTENCES placement blocks
    # each placement block is indexed by the index of the sentence object fed to sentence2
    # Each placement block receives the same question and sentence1

    # max limits as determined by axon and memory limits
    # Memory constraint (600 dim input, shared weights)
    max_neurons_per_core_by_memory = min((MAX_SYNAPSES_FF // input_size, layer_size))

    assert max_neurons_per_core_by_memory < MAX_TOTAL_NEURONS_IN_CORE

    max_neurons_per_core = max_neurons_per_core_by_memory

    n_cores_per_layer = np.ceil(layer_size / max_neurons_per_core)
    neurons_per_core = int(np.ceil(layer_size / n_cores_per_layer))

    assert np.ceil(layer_size / neurons_per_core) == np.ceil(layer_size / max_neurons_per_core)
    return int(neurons_per_core), int(n_cores_per_layer)


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    with ipdb.launch_ipdb_on_exception():

        neurons_per_core, n_cores = get_placement_param_list_relnet_input_no_ws(layer_size=256,
                                                                                LSNN_sentence_size=200,
                                                                                LSNN_question_size=200)
        n_rel_nets = (MAX_N_SENTENCES * (MAX_N_SENTENCES+1)) // 2
        assert n_cores / n_rel_nets == np.ceil(256 / neurons_per_core), "Num cores mismatch"
        print("For relational networks input layer, neurons per core = {}, total cores = {}"
              .format(neurons_per_core, n_cores))

        (sentence_n_cores_total,
         sentence_input_neurons_per_core,
         sentence_neurons_per_core,
         sentence_relay_neurons_per_core,
         sentence_n_relay_copies) = get_minimum_n_cores_for_LSNN_placement_no_ws(200, 180, n_cores // n_rel_nets, MAX_N_SENTENCES+1, MAX_N_SENTENCES)

        print("For the LSNN Placement for sentences with \n"
              "    {} Neurons and {} Inputs,\n"
              "    {} Cores per relnet instance,\n"
              "    {} Relnets to which each LSNN fans out to,\n"
              "    {} instances (sentences),\n"
              "\n"
              "We have:".format(200, 180, n_cores // n_rel_nets, MAX_N_SENTENCES+1, MAX_N_SENTENCES))

        print("Neurons per core (LSNN): {}, Relay neurons per core {}, Input Neurons per core: {}, Relay Copies For Fanout: {}, Total Cores: {}"
              .format(sentence_neurons_per_core,
                      sentence_relay_neurons_per_core,
                      sentence_input_neurons_per_core,
                      sentence_n_relay_copies,
                      sentence_n_cores_total))
        print("n_cores (LSNN): {}, Relay n_cores_per_copy {}, Input n_cores: {}"
              .format(int(np.ceil(200 / sentence_neurons_per_core)),
                      int(np.ceil(200 / sentence_relay_neurons_per_core)),
                      int(np.ceil(180 / sentence_input_neurons_per_core))))

        (question_n_cores_total,
         question_input_neurons_per_core,
         question_neurons_per_core,
         question_relay_neurons_per_core,
         question_n_relay_copies) = get_minimum_n_cores_for_LSNN_placement_no_ws(200, 180, n_cores // n_rel_nets, ((MAX_N_SENTENCES+1)*MAX_N_SENTENCES)//2, 1)

        print("For the LSNN Placement for questions with \n"
              "    {} Neurons and {} Inputs,\n"
              "    {} Cores per relnet instance,\n"
              "    {} Relnets to which each LSNN fans out to,\n"
              "    {} instances (questions),\n"
              "\n"
              "We have:".format(200, 180, n_cores // n_rel_nets, ((MAX_N_SENTENCES+1)*MAX_N_SENTENCES)//2, 1))

        print("Neurons per core (LSNN): {}, Relay neurons per core {}, Input Neurons per core: {}, Relay Copies For Fanout: {}, Total Cores: {}"
              .format(question_neurons_per_core,
                      question_relay_neurons_per_core,
                      question_input_neurons_per_core,
                      question_n_relay_copies,
                      question_n_cores_total))
        print("n_cores (LSNN): {}, Relay n_cores_per_copy {}, Input n_cores: {}"
              .format(int(np.ceil(200 / question_neurons_per_core)),
                      int(np.ceil(200 / question_relay_neurons_per_core)),
                      int(np.ceil(180 / question_input_neurons_per_core))))

        # neurons_per_core, n_cores = get_placement_param_list_relnet_intermediate_no_ws()
        # n_rel_nets = (MAX_N_SENTENCES * (MAX_N_SENTENCES+1)) // 2
        # assert n_cores / n_rel_nets == np.ceil(256 / neurons_per_core), "Num cores mismatch"
        # print("For relational networks intermediate layers, neurons per core = {}, total cores = {}"
        #       .format(neurons_per_core, n_cores))
