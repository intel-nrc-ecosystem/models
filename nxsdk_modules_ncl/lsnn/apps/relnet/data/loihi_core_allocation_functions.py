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
from typing import Tuple
from collections import namedtuple
from collections import defaultdict
from itertools import product as cart_product
from itertools import chain as iter_chain

from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import OUTPUT_AXON_LIMIT_WITHIN_CHIP
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import N_CORES_PER_CHIP
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import get_minimum_n_cores_for_LSNN_placement_no_ws
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNCoreTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNConnectionTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNCoresStruct
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNConnectionsStruct
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import LSNNCopyStruct
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitCoreTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitConnectionTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitConnectionsStruct
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetInitInstanceStruct
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetIntermediateCoreTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetIntermediateConnectionTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import RelnetIntermediateInstanceStruct
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import TranslationLayerCoreTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import TranslationLayerConnectionTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import TranslationLayerInstanceStruct
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import FinalMLPCoreTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import FinalMLPConnectionTuple
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_data_structs import FinalMLPInstanceStruct


def almost_even_distribution(n_total, n_groups, group_index):
    """
    returns the number of elements in the group_index'th group if n_total elements are split almost evenly into n_groups
    """
    assert group_index < n_groups, "almost_even_distribution: group_index must be < n_groups"
    assert n_groups <= n_total, "almost_even_distribution: n_groups must <= n_total"
    if not np.all(int(x) == x for x in {n_total, n_groups, group_index}):
        raise TypeError("parameters n_total, n_groups, group_index should be Integral")

    n_total = int(n_total)
    n_groups = int(n_groups)
    group_index = int(group_index)

    return n_total // n_groups + int(group_index < n_total % n_groups)


def get_almost_even_index(n_total, n_groups, elem_index):
    """
    returns the index of the group to which the elem_index'th element belongs to when n_total elements are split almost evenly into n_groups
    """
    assert elem_index < n_total, "almost_even_distribution: elem_index must be < n_total"
    assert n_groups <= n_total, "almost_even_distribution: n_groups must <= n_total"
    if not np.all(int(x) == x for x in {n_total, n_groups, elem_index}):
        raise TypeError("parameters n_total, n_groups, index should be Integral")

    n_total = int(n_total)
    n_groups = int(n_groups)
    elem_index = int(elem_index)

    higher_value_cutoff_group = n_total % n_groups
    higher_value = n_total // n_groups + 1
    lower_value = n_total // n_groups
    higher_value_cutoff_elem_index = higher_value_cutoff_group * higher_value

    if elem_index >= higher_value_cutoff_elem_index:
        group_index = higher_value_cutoff_group + (elem_index - higher_value_cutoff_elem_index) // lower_value
    else:
        group_index = elem_index // higher_value

    return group_index


# Placement of LSNN Network
def get_LSNN_placement(LSNN_rec_weights: np.ndarray, LSNN_rec_delays: np.ndarray, LSNN_rec_weight_exp,
                       LSNN_inp_weights: np.ndarray, LSNN_inp_delays: np.ndarray, LSNN_inp_weight_exp,
                       relay_weight,
                       # Parameters for placement
                       input_neurons_per_core, neurons_per_core, relay_neurons_per_core, relay_copies_for_fanout, n_instances,
                       # Other fundamental parameters
                       layer_name, core_index_start):
    """
    create n_lsnn_cores lsnn cores -
        assign the neurons to cores.

    create n_input_cores input cores -
        assign the input neurons to cores.

    create n_relay_cores relay core -
        assign the relay neurons to cores

    for each relay copy -
        create copy of relay cores and store them in array

    for each pair of lsnn cores:
        fetch weight submatrix
        fetch delay submatrix
        assign them to a list with the appropriate core index

    for each relay copy
        perform one-one connections from LSNN to this relay core

    create n_instances copies of the above cores and connections
    for each of these copies (indexed by sentence_ind)
        offset all relevant core indices (in the core definition and in the connection list)
        by sentence_ind*(n_lsnn_cores + n_input_cores + relay_copies_for_fanout*n_relay_cores)

    each core is defined by
        - an id,
        - a layer (to which it belongs),
        - and the start and end index of the neuron range that it contains

    each connection between cores is defined by
        - a presyn core id
        - a postsyn core id
        - a weight matrix
        - a delay matrix

    In the end we return an np.ndarray of type object 'lsnn_core_connection_array'

    where for each i, j

    ret[i, j] corresponds to the cores and connections corresponding to
    sentence i and fanout copy j

    ret[i, j].cores.input = tuple of cores containing the input neurons
    ret[i, j].cores.lsnn = tuple of cores containing the lsnn neurons
    ret[i, j].connections.input_to_lsnn = tuple of connections from the input to lsnn
    ret[i, j].connections.lsnn_to_lsnn = tuple of connections from the lsnn to lsnn
    """

    LSNN_size = LSNN_rec_weights.shape[0]
    input_dim = LSNN_inp_weights.shape[0]

    assert LSNN_rec_weights.ndim == 2 and all(x == LSNN_size for x in LSNN_rec_weights.shape), \
        "Something wrong with LSNN recurrent weight shape"
    assert LSNN_inp_weights.shape == (input_dim, LSNN_size)
    assert LSNN_rec_weights.shape == LSNN_rec_delays.shape
    assert LSNN_inp_weights.shape == LSNN_inp_delays.shape
    assert np.all(np.diag(LSNN_rec_weights) == 0),  "It appears that the diagonal weights of LSNN_rec_weights are not zero"
    assert np.all(np.diag(LSNN_rec_delays) == 0),  "It appears that the diagonal delays of LSNN_rec_weights are not zero"

    # disable write on any of these
    LSNN_rec_weights = LSNN_rec_weights.view()
    LSNN_rec_delays = LSNN_rec_delays.view()
    LSNN_inp_weights = LSNN_inp_weights.view()
    LSNN_inp_delays = LSNN_inp_delays.view()
    LSNN_rec_weights.setflags(write=False)
    LSNN_rec_delays.setflags(write=False)
    LSNN_inp_weights.setflags(write=False)
    LSNN_inp_delays.setflags(write=False)

    LSNN_rec_mask = np.ones(LSNN_rec_weights.shape, dtype=np.int64)
    LSNN_rec_mask[np.arange(LSNN_size), np.arange(LSNN_size)] = 0
    LSNN_inp_mask = np.ones(LSNN_inp_weights.shape, dtype=np.int64)
    LSNN_rec_mask.setflags(write=False)
    LSNN_inp_mask.setflags(write=False)

    LSNN_relay_delays = np.zeros((LSNN_size, LSNN_size), dtype=np.float32)
    LSNN_relay_mask = np.eye(LSNN_size, dtype=np.int64)
    LSNN_relay_weights = relay_weight*np.eye(LSNN_size, dtype=np.float32)
    LSNN_relay_delays.setflags(write=False)
    LSNN_relay_mask.setflags(write=False)
    LSNN_relay_weights.setflags(write=False)

    # =====================================================================
    # Get the parameters corresponding to the optimal placement of the LSNN
    # =====================================================================

    input_neurons_per_core = int(input_neurons_per_core)
    neurons_per_core = int(neurons_per_core)
    n_instances_orig = n_instances
    if n_instances == 0:
        n_instances = 1  # This is the case of the question LSNN

    n_lsnn_cores = int(np.ceil(LSNN_size / neurons_per_core))
    n_input_cores = int(np.ceil(input_dim / input_neurons_per_core))
    if relay_neurons_per_core > 0:
        n_relay_cores = int(np.ceil(LSNN_size / relay_neurons_per_core))
    else:
        n_relay_cores = 0
        relay_copies_for_fanout = 0

    # =====================================================
    # Assigning the Cores for a particular copy of the LSNN
    # =====================================================

    current_core_id = core_index_start

    lsnn_cores = LSNNCoresStruct(input=[], lsnn=[], relay=np.ndarray(relay_copies_for_fanout, dtype=object))

    # Assigning the cores for the input neurons
    start_neuron_index = 0
    for i in range(n_input_cores):
        n_neurons_current = almost_even_distribution(input_dim, n_input_cores, i)
        end_neuron_index = start_neuron_index + n_neurons_current
        assert end_neuron_index <= input_dim

        lsnn_cores.input.append(LSNNCoreTuple(id=current_core_id,
                                              layer_name='{}_input'.format(layer_name),
                                              start=start_neuron_index,
                                              end=end_neuron_index,
                                              sentence_ind=0,
                                              fanout_copy_ind=0))
        current_core_id += 1
        start_neuron_index = end_neuron_index
        if i == n_input_cores - 1:
            assert start_neuron_index == input_dim, \
                "Something went wrong with the input assignment to cores in the LSNN layer {}".format(layer_name)

    # Assigning the cores for the lsnn neurons
    start_neuron_index = 0
    for i in range(n_lsnn_cores):
        n_neurons_current = almost_even_distribution(LSNN_size, n_lsnn_cores, i)
        end_neuron_index = min(start_neuron_index + n_neurons_current, LSNN_size)
        assert end_neuron_index <= LSNN_size

        lsnn_cores.lsnn.append(LSNNCoreTuple(id=current_core_id,
                                             layer_name='{}_recurrent'.format(layer_name),
                                             start=start_neuron_index,
                                             end=end_neuron_index,
                                             sentence_ind=0,
                                             fanout_copy_ind=0))
        current_core_id += 1
        start_neuron_index = end_neuron_index

        if i == n_lsnn_cores - 1:
            assert start_neuron_index == LSNN_size, \
                "Something went wrong with the lsnn assignment to cores in the LSNN layer {}".format(layer_name)

    # Assigning the cores for the relay neurons
    relay_core_tuple = []
    start_neuron_index = 0
    for i in range(n_relay_cores):
        n_neurons_current = almost_even_distribution(LSNN_size, n_relay_cores, i)
        end_neuron_index = min(start_neuron_index + n_neurons_current, LSNN_size)
        assert end_neuron_index <= LSNN_size

        relay_core_tuple.append(LSNNCoreTuple(id=current_core_id,
                                              layer_name='{}_relay'.format(layer_name),
                                              start=start_neuron_index,
                                              end=end_neuron_index,
                                              sentence_ind=0,
                                              fanout_copy_ind=0))
        current_core_id += 1
        start_neuron_index = end_neuron_index

        if i == n_relay_cores - 1:
            assert start_neuron_index == LSNN_size, \
                "Something went wrong with the lsnn assignment to cores in the LSNN layer {}".format(layer_name)

    relay_core_tuple = tuple(relay_core_tuple)
    del start_neuron_index  # delete loop variables
    if n_relay_cores:
        del end_neuron_index, i

    # create relay_copies_for_fanout copies of the relay cores
    def offset_relay_core_by(relay_core, copy_index):
        return LSNNCoreTuple(id=relay_core.id + copy_index*n_relay_cores,
                             layer_name=relay_core.layer_name,
                             start=relay_core.start,
                             end=relay_core.end,
                             sentence_ind=relay_core.sentence_ind,
                             fanout_copy_ind=copy_index)

    for i in range(relay_copies_for_fanout):
        lsnn_cores.relay[i] = tuple(offset_relay_core_by(x, i) for x in relay_core_tuple)
    lsnn_cores.relay.setflags(write=False)

    # Converting lists to tuple (immutability)
    lsnn_cores = LSNNCoresStruct(input=tuple(lsnn_cores.input),
                                 lsnn=tuple(lsnn_cores.lsnn),
                                 relay=lsnn_cores.relay)

    # ===========================================================
    # Assigning the connections for a particular copy of the LSNN
    # ===========================================================
    lsnn_connections = LSNNConnectionsStruct(input_to_lsnn=[],
                                             lsnn_to_lsnn=[],
                                             lsnn_to_relay=[])

    # Assigning the connections for input_to_lsnn connections
    for inpcore in lsnn_cores.input:  # type: LSNNCoreTuple
        for lsnncore in lsnn_cores.lsnn:  # type: LSNNCoreTuple
            presyn_core_id = inpcore.id
            postsyn_core_id = lsnncore.id
            presyn_neuron_range = slice(inpcore.start, inpcore.end)
            postsyn_neuron_range = slice(lsnncore.start, lsnncore.end)

            lsnn_connections.input_to_lsnn.append(
                LSNNConnectionTuple(presyn_core=presyn_core_id,
                                    postsyn_core=postsyn_core_id,
                                    mask=LSNN_inp_mask[presyn_neuron_range, postsyn_neuron_range],
                                    weights=LSNN_inp_weights[presyn_neuron_range, postsyn_neuron_range],
                                    delays=LSNN_inp_delays[presyn_neuron_range, postsyn_neuron_range],
                                    weight_exp=LSNN_inp_weight_exp,
                                    sentence_ind=0,
                                    fanout_copy_ind=0))

    # Assigning the connections for lsnn_to_lsnn connections
    for lsnncorepre in lsnn_cores.lsnn:  # type: LSNNCoreTuple
        for lsnncorepost in lsnn_cores.lsnn:  # type: LSNNCoreTuple
            presyn_core_id = lsnncorepre.id
            postsyn_core_id = lsnncorepost.id
            presyn_neuron_range = slice(lsnncorepre.start, lsnncorepre.end)
            postsyn_neuron_range = slice(lsnncorepost.start, lsnncorepost.end)

            lsnn_connections.lsnn_to_lsnn.append(
                LSNNConnectionTuple(presyn_core=presyn_core_id,
                                    postsyn_core=postsyn_core_id,
                                    mask=LSNN_rec_mask[presyn_neuron_range, postsyn_neuron_range],
                                    weights=LSNN_rec_weights[presyn_neuron_range, postsyn_neuron_range],
                                    delays=LSNN_rec_delays[presyn_neuron_range, postsyn_neuron_range],
                                    weight_exp=LSNN_rec_weight_exp,
                                    sentence_ind=0,
                                    fanout_copy_ind=0))
    del lsnncorepre, lsnncorepost  # delete loop variables

    # Assigning the connections for lsnn_to_relay connections
    for relay_cores_copy in lsnn_cores.relay:
        for lsnncore in lsnn_cores.lsnn:  # type: LSNNCoreTuple
            for relaycore in relay_cores_copy:  # type: LSNNCoreTuple
                presyn_core_id = lsnncore.id
                postsyn_core_id = relaycore.id
                presyn_neuron_range = slice(lsnncore.start, lsnncore.end)
                postsyn_neuron_range = slice(relaycore.start, relaycore.end)
                common_range_start = max(lsnncore.start, relaycore.start)
                common_range_end = min(lsnncore.end, relaycore.end)

                # This conditions is because if th's a one-one connections we only
                # connect if there's a common range in the presynaptic and
                # postsynaptic neuron groups
                if common_range_end > common_range_start:
                    conn_tuple = \
                        LSNNConnectionTuple(presyn_core=presyn_core_id,
                                            postsyn_core=postsyn_core_id,
                                            mask=LSNN_relay_mask[presyn_neuron_range, postsyn_neuron_range],
                                            weights=LSNN_relay_weights[presyn_neuron_range, postsyn_neuron_range],
                                            delays=LSNN_relay_delays[presyn_neuron_range, postsyn_neuron_range],
                                            weight_exp=0,
                                            sentence_ind=0,
                                            fanout_copy_ind=relaycore.fanout_copy_ind)
                    lsnn_connections.lsnn_to_relay.append(conn_tuple)

                    # neurons_connected_ind = np.stack(np.where(conn_tuple.mask | (conn_tuple.weights != 0)), axis=-1)
                    # neurons_connected = neurons_connected_ind + np.array([[presyn_neuron_range.start, postsyn_neuron_range.start]])
                    # connected_weights = conn_tuple.weights[neurons_connected_ind[:, 0], neurons_connected_ind[:, 1]]
                    # print("Connections")
                    # print('\n'.join("{} -> {}, with weight {}".format(x[0], x[1], w) for x, w in zip(neurons_connected, connected_weights)))

    if relay_copies_for_fanout:  # cleanup variables only if loop above has been entered
        del presyn_core_id, postsyn_core_id, presyn_neuron_range, postsyn_neuron_range

    # Converting lists to tuple (immutability)
    lsnn_connections = LSNNConnectionsStruct(input_to_lsnn=tuple(lsnn_connections.input_to_lsnn),
                                             lsnn_to_lsnn=tuple(lsnn_connections.lsnn_to_lsnn),
                                             lsnn_to_relay=tuple(lsnn_connections.lsnn_to_relay))

    # =============================
    # Creating copies for sentences
    # =============================
    lsnn_copy_data = LSNNCopyStruct(cores=lsnn_cores, connections=lsnn_connections)
    lsnn_core_connection_array = np.ndarray(n_instances, dtype=object)

    def offset_cores_and_connections(lsnn_copy_data: LSNNCopyStruct, sentence_ind: int):
        connections = lsnn_copy_data.connections
        cores = lsnn_copy_data.cores

        core_ind_offset = sentence_ind*(n_lsnn_cores + n_input_cores + relay_copies_for_fanout*n_relay_cores)

        def update_connection(conn: LSNNConnectionTuple):
            new_conn = LSNNConnectionTuple(presyn_core=conn.presyn_core + core_ind_offset,
                                           postsyn_core=conn.postsyn_core + core_ind_offset,
                                           mask=conn.mask,
                                           weights=conn.weights,
                                           delays=conn.delays,
                                           weight_exp=conn.weight_exp,
                                           sentence_ind=sentence_ind,
                                           fanout_copy_ind=conn.fanout_copy_ind)

            return new_conn

        def update_core(core: LSNNCoreTuple):
            new_core = LSNNCoreTuple(id=core.id + core_ind_offset,
                                     layer_name=core.layer_name,
                                     start=core.start,
                                     end=core.end,
                                     sentence_ind=sentence_ind,
                                     fanout_copy_ind=core.fanout_copy_ind)

            return new_core

        new_connections = LSNNConnectionsStruct(
            input_to_lsnn=tuple(update_connection(x) for x in connections.input_to_lsnn),
            lsnn_to_lsnn=tuple(update_connection(x) for x in connections.lsnn_to_lsnn),
            lsnn_to_relay=tuple(update_connection(x) for x in connections.lsnn_to_relay))

        new_relay_cores = np.ndarray(relay_copies_for_fanout, dtype=object)
        for i in range(relay_copies_for_fanout):
            new_relay_cores[i] = tuple(update_core(x) for x in cores.relay[i])
        new_relay_cores.setflags(write=False)

        new_cores = LSNNCoresStruct(
            input=tuple(update_core(x) for x in cores.input),
            lsnn=tuple(update_core(x) for x in cores.lsnn),
            relay=new_relay_cores)

        new_lsnn_copy_data = LSNNCopyStruct(cores=new_cores,
                                            connections=new_connections)

        return new_lsnn_copy_data

    for sentence_ind in range(n_instances):
        lsnn_core_connection_array[sentence_ind] = \
            offset_cores_and_connections(lsnn_copy_data=lsnn_copy_data,
                                         sentence_ind=sentence_ind)

    # update current_core_id taking into account all the copies created
    current_core_id = (n_lsnn_cores + n_input_cores + n_relay_cores * relay_copies_for_fanout) * n_instances + core_index_start

    if relay_neurons_per_core > 0:
        last_core = lsnn_core_connection_array[-1].cores.relay[-1][-1]
    else:
        last_core = lsnn_core_connection_array[-1].cores.lsnn[-1]

    assert last_core.id == current_core_id - 1, \
        "There appears to be an issue with the assignment of core indices"

    if n_instances_orig == 0:
        # In case of questions, return the single LSNNCopyStruct instance
        lsnn_core_connection_array = lsnn_core_connection_array[0]

    return lsnn_core_connection_array, current_core_id


def get_input_mask_placement(n_instances, core_index_start):
    return LSNNCoreTuple(core_index_start, 'LSNN_mask', 0, n_instances,
                         sentence_ind=None, fanout_copy_ind=None), core_index_start + 1


def get_relnet_init_placement(relnet_inp_weights, relnet_inp_delays,
                              lsnn_sentence_core_connection_array, lsnn_question_core_connection_array,
                              input_mask_core, input_mask_weight, input_mask_weight_exp,
                              # Parameters for placement
                              neurons_per_core, max_n_sentences,
                              # Other fundamental parameters
                              layer_name, core_index_start):
    """
    create n_relnet_cores relnet cores -
        assign the neurons to cores.

    for each valid sentence pair:
        select fanout copy of the sentence1 lstm*
        select fanout copy of the sentence2 lstm
        select fanout copy of the question lstm

        create the connection matrices corresponding to the relnet

    * This involves the following:
      each sentence_index has a fanout counter initialized to 0
      each time a sentence is used as input,
      fanout_counter[sentence_ind] += 1

      fanout_copy_index = index_of_current_fanout % copies_for_fanout

    In the end we return an np.ndarray of type object RelnetInitInstanceStruct

    where for each i, j

    ret[i, j] corresponds to the cores and connections corresponding to
    sentence1 = i and sentence2 = j

    for i > j ret[i, j] is None

    ret[i, j].cores = tuple of cores containing the input neurons
    ret[i, j].connections.sentence1_to_relnet = tuple of connections from the sentence1 lsnn to relnet
    ret[i, j].connections.sentence2_to_relnet = tuple of connections from the sentence2 lsnn to relnet
    ret[i, j].connections.question_to_relnet = tuple of connections from the question lsnn to relnet
    """

    assert len(lsnn_sentence_core_connection_array[0].cores.relay) > 0, \
        "Currently it is required that you use relay groups in the LSNN"

    relnet_size = relnet_inp_weights.shape[1]

    max_n_sentences = lsnn_sentence_core_connection_array.shape[0]

    assert lsnn_sentence_core_connection_array.shape == (max_n_sentences,)
    assert isinstance(lsnn_question_core_connection_array, LSNNCopyStruct)

    copies_for_fanout_sentences = lsnn_sentence_core_connection_array[0].cores.relay.shape[0]
    copies_for_fanout_questions = lsnn_question_core_connection_array.cores.relay.shape[0]

    lsnn_size_sentences = lsnn_sentence_core_connection_array[0].cores.lsnn[-1].end
    lsnn_size_questions = lsnn_question_core_connection_array.cores.lsnn[-1].end

    assert relnet_inp_weights.shape[0] == 2*lsnn_size_sentences + lsnn_size_questions, \
        "Relnet init layer incorrect weights size mismatch"
    assert relnet_inp_weights.shape == relnet_inp_delays.shape

    # disable write on any of these
    relnet_inp_mask = np.ones(relnet_inp_weights.shape, dtype=np.int64)  # type: np.ndarray
    relnet_inp_weights = relnet_inp_weights.view()  # type: np.ndarray
    relnet_inp_delays = relnet_inp_delays.view()  # type: np.ndarray
    lsnn_sentence_core_connection_array = lsnn_sentence_core_connection_array.view()  # type: np.ndarray
    relnet_inp_mask.setflags(write=False)
    relnet_inp_weights.setflags(write=False)
    relnet_inp_delays.setflags(write=False)
    lsnn_sentence_core_connection_array.setflags(write=False)

    # ==================================================================================================
    # Get the parameters corresponding to the optimal placement of the first layer of relational network
    # ==================================================================================================

    neurons_per_core = int(neurons_per_core)

    n_relnet_cores = int(np.ceil(relnet_size / neurons_per_core))

    # ========================================================================
    # Assigning the Cores for each a sample instance of the relational network
    # ========================================================================

    current_core_id = core_index_start

    # Assigning the cores for the sample relnet instance
    relnet_cores = []
    start_neuron_index = 0
    for i in range(n_relnet_cores):
        end_neuron_index = min(start_neuron_index + neurons_per_core, relnet_size)
        relnet_cores.append(RelnetInitCoreTuple(id=current_core_id,
                                                layer_name=layer_name,
                                                start=start_neuron_index,
                                                end=end_neuron_index,
                                                sentence1_ind=0,
                                                sentence2_ind=0))
        current_core_id += 1
        start_neuron_index = end_neuron_index

        if i == n_relnet_cores - 1:
            assert start_neuron_index == relnet_size, \
                "Something went wrong with the assignment of the relnet to cores in layer {}".format(layer_name)

    del start_neuron_index, end_neuron_index, i  # delete loop variables

    # Converting lists to tuple (immutability)
    relnet_cores = tuple(relnet_cores)

    # ===========================================================================
    # Assigning the Cores and Connections for each copy of the relational network
    # ===========================================================================

    def append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind):
        return tuple(RelnetInitCoreTuple(id=core.id + relnet_offset * n_relnet_cores,
                                         layer_name=core.layer_name,
                                         start=core.start,
                                         end=core.end,
                                         sentence1_ind=sentence1_ind,
                                         sentence2_ind=sentence2_ind)
                     for core in relnet_cores)

    def connect_input_to_relnet(input_cores: Tuple[LSNNCoreTuple], relnet_cores: Tuple[RelnetInitCoreTuple], mask, weights, delays, weight_exp):
        """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
        """
        connection_tuple = []
        for presyn_core in input_cores:
            for postsyn_core in relnet_cores:
                presyn_neuron_range = slice(presyn_core.start, presyn_core.end)
                postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
                connection_tuple.append(
                    RelnetInitConnectionTuple(presyn_core=presyn_core.id,
                                              postsyn_core=postsyn_core.id,
                                              mask=mask[presyn_neuron_range, postsyn_neuron_range],
                                              weights=weights[presyn_neuron_range, postsyn_neuron_range],
                                              delays=delays[presyn_neuron_range, postsyn_neuron_range],
                                              weight_exp=weight_exp,
                                              sentence1_ind=sentence1_ind,
                                              sentence2_ind=sentence2_ind))
        return tuple(connection_tuple)

    def connect_mask_to_relnet(mask_core, relnet_cores: Tuple[RelnetInitCoreTuple], sentence1_ind, sentence2_ind, weight_value, weight_exp):
        """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
        """
        connection_tuple = []
        weights = np.zeros((max_n_sentences, relnet_size))
        weights[np.array([sentence1_ind, sentence2_ind]), :] = weight_value  # maximum negative weight
        mask = (weights != 0).astype(np.int64)  # maximum negative weight
        delays = np.zeros((max_n_sentences, relnet_size))
        weights.setflags(write=False)
        mask.setflags(write=False)
        delays.setflags(write=False)

        for postsyn_core in relnet_cores:
            presyn_neuron_range = slice(None)
            postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
            connection_tuple.append(
                RelnetInitConnectionTuple(presyn_core=mask_core.id,
                                          postsyn_core=postsyn_core.id,
                                          mask=mask[presyn_neuron_range, postsyn_neuron_range],
                                          weights=weights[presyn_neuron_range, postsyn_neuron_range],
                                          delays=delays[presyn_neuron_range, postsyn_neuron_range],
                                          weight_exp=weight_exp,
                                          sentence1_ind=sentence1_ind,
                                          sentence2_ind=sentence2_ind))
        return tuple(connection_tuple)

    relnet_offset = 0  # Number of relnets that have been processed
    fanout_indices = np.zeros(max_n_sentences, dtype=np.int64)
    relnet_instance_array = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)
    for sentence1_ind in range(max_n_sentences):
        for sentence2_ind in range(sentence1_ind, max_n_sentences):
            sentence1_relay_fanout_copy_ind = get_almost_even_index(max_n_sentences+1, copies_for_fanout_sentences, fanout_indices[sentence1_ind])
            sentence1_lsnn_copy = lsnn_sentence_core_connection_array[sentence1_ind]  # type: LSNNCopyStruct
            sentence1_lsnn_cores = sentence1_lsnn_copy.cores.relay[sentence1_relay_fanout_copy_ind]  # type: Tuple[LSNNCoreTuple]
            fanout_indices[sentence1_ind] += 1

            sentence2_relay_fanout_copy_ind = get_almost_even_index(max_n_sentences+1, copies_for_fanout_sentences, fanout_indices[sentence2_ind])
            sentence2_lsnn_copy = lsnn_sentence_core_connection_array[sentence2_ind]  # type: LSNNCopyStruct
            sentence2_lsnn_cores = sentence2_lsnn_copy.cores.relay[sentence2_relay_fanout_copy_ind]  # type: Tuple[LSNNCoreTuple]
            fanout_indices[sentence2_ind] += 1

            question_relay_fanout_copy_ind = get_almost_even_index((max_n_sentences*(max_n_sentences+1))//2, copies_for_fanout_questions, relnet_offset)
            question_lsnn_cores = lsnn_question_core_connection_array.cores.relay[question_relay_fanout_copy_ind]  # type: Tuple[LSNNCoreTuple]

            relnet_instance_cores = append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind)
            relnet_offset += 1

            # here we perform the connections between the selected LSNN cores and the current relnet instance cores
            sentence1_to_relnet_conns = connect_input_to_relnet(sentence1_lsnn_cores,
                                                                relnet_instance_cores,
                                                                relnet_inp_mask[:lsnn_size_sentences],
                                                                relnet_inp_weights[:lsnn_size_sentences],
                                                                relnet_inp_delays[:lsnn_size_sentences],
                                                                0)
            sentence2_to_relnet_conns = connect_input_to_relnet(sentence2_lsnn_cores,
                                                                relnet_instance_cores,
                                                                relnet_inp_mask[lsnn_size_sentences:2*lsnn_size_sentences],
                                                                relnet_inp_weights[lsnn_size_sentences:2*lsnn_size_sentences],
                                                                relnet_inp_delays[lsnn_size_sentences:2*lsnn_size_sentences],
                                                                0)
            question_to_relnet_conns = connect_input_to_relnet(question_lsnn_cores,
                                                               relnet_instance_cores,
                                                               relnet_inp_mask[2*lsnn_size_sentences:],
                                                               relnet_inp_weights[2*lsnn_size_sentences:],
                                                               relnet_inp_delays[2*lsnn_size_sentences:],
                                                               0)

            mask_to_relnet_conns = connect_mask_to_relnet(input_mask_core,
                                                          relnet_instance_cores,
                                                          sentence1_ind, sentence2_ind,
                                                          input_mask_weight,
                                                          input_mask_weight_exp)

            relnet_instance_connections = RelnetInitConnectionsStruct(sentence1_to_relnet=sentence1_to_relnet_conns,
                                                                      sentence2_to_relnet=sentence2_to_relnet_conns,
                                                                      question_to_relnet=question_to_relnet_conns,
                                                                      mask_to_relnet=mask_to_relnet_conns)
            relnet_instance_array[sentence1_ind, sentence2_ind] = RelnetInitInstanceStruct(cores=relnet_instance_cores,
                                                                                           connections=relnet_instance_connections)

    current_core_id = n_relnet_cores * (max_n_sentences*(max_n_sentences+1))//2 + core_index_start
    assert relnet_instance_array[-1, -1].cores[-1].id == current_core_id - 1, \
        "There appears to be an issue with the assignment of core indices"

    return relnet_instance_array, current_core_id


def get_relnet_intermediate_placement(relnet_inp_weights, relnet_inp_delays,
                                      input_relnet_core_connection_array,
                                      # Parameters for placement
                                      neurons_per_core, max_n_sentences,
                                      # Other fundamental parameters
                                      layer_name, core_index_start):
    """
    create n_relnet_cores relnet cores -
        assign the neurons to cores.

    for each valid sentence pair:
        select fanout copy of the sentence1 lstm*
        select fanout copy of the sentence2 lstm
        select fanout copy of the question lstm

        create the connection matrices corresponding to the relnet

    * This involves the following:
      each sentence_index has a fanout counter initialized to 0
      each time a sentence is used as input,
      fanout_counter[sentence_ind] += 1

      fanout_copy_index = index_of_current_fanout % copies_for_fanout

    # each fanout copy has a counter of the number of relational networks it can forward to
    #     remaining_fanouts[i, j] = number of fanouts possible for fanout copy j of sentence i

    # # remaining fanouts is initialized as below
    # remaining_fanouts = np.ones((n_sentences, copies_for_fanout), dtype=np.int32) * (total_fanout)//copies_for_fanout
    # remaining_fanouts[:(total_fanout%copies_for_fanout)] += 1

    # we find the index of the first non-zero element in remaining_fanouts[sentence1].
    # This becomes the index of the sentence1 fanout copy
    # remaining_fanouts[sentence1, index_of_first_nonzero] -= 1  # we use up one of the fanouts 

    each core is defined by
        - an id,
        - a layer (to which it belongs),
        - and the start and end index of the neuron range that it contains

    each connection between cores is defined by
        - a presyn core id
        - a postsyn core id
        - a weight matrix
        - a delay matrix

    In the end we return an np.ndarray of type object 'lsnn_core_connection_array'

    where for each i, j

    ret[i, j] corresponds to the cores and connections corresponding to
    sentence i and fanout copy j

    ret[i, j].cores = tuple of cores containing the input neurons
    ret[i, j].connections = tuple of connections from the previous layer to the current layer
    """

    relnet_size = relnet_inp_weights.shape[1]

    assert relnet_inp_weights.shape == relnet_inp_delays.shape

    # disable write on any of these
    relnet_inp_mask = np.ones(relnet_inp_weights.shape, dtype=np.int64)  # type: np.ndarray
    relnet_inp_weights = relnet_inp_weights.view()  # type: np.ndarray
    relnet_inp_delays = relnet_inp_delays.view()  # type: np.ndarray
    input_relnet_core_connection_array = input_relnet_core_connection_array.view()  # type: np.ndarray
    relnet_inp_mask.setflags(write=False)
    relnet_inp_weights.setflags(write=False)
    relnet_inp_delays.setflags(write=False)
    input_relnet_core_connection_array.setflags(write=False)

    # ==================================================================================================
    # Get the parameters corresponding to the optimal placement of the first layer of relational network
    # ==================================================================================================

    neurons_per_core = int(neurons_per_core)

    n_relnet_cores = int(np.ceil(relnet_size / neurons_per_core))

    # ========================================================================
    # Assigning the Cores for each a sample instance of the relational network
    # ========================================================================

    current_core_id = core_index_start

    # Assigning the cores for the sample relnet instance
    relnet_cores = []
    start_neuron_index = 0
    for i in range(n_relnet_cores):
        end_neuron_index = min(start_neuron_index + neurons_per_core, relnet_size)
        relnet_cores.append(RelnetIntermediateCoreTuple(id=current_core_id,
                                                        layer_name=layer_name,
                                                        start=start_neuron_index,
                                                        end=end_neuron_index,
                                                        sentence1_ind=0,
                                                        sentence2_ind=0))
        current_core_id += 1
        start_neuron_index = end_neuron_index

        if i == n_relnet_cores - 1:
            assert start_neuron_index == relnet_size, \
                "Something went wrong with the assignment of the relnet to cores in layer {}".format(layer_name)

    del start_neuron_index, end_neuron_index, i  # delete loop variables

    # Converting lists to tuple (immutability)
    relnet_cores = tuple(relnet_cores)

    # ===========================================================================
    # Assigning the Cores and Connections for each copy of the relational network
    # ===========================================================================

    def append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind):
        return tuple(RelnetIntermediateCoreTuple(id=core.id + relnet_offset * n_relnet_cores,
                                                 layer_name=core.layer_name,
                                                 start=core.start,
                                                 end=core.end,
                                                 sentence1_ind=sentence1_ind,
                                                 sentence2_ind=sentence2_ind)
                     for core in relnet_cores)

    relnet_offset = 0  # Number of relnets that have been processed
    relnet_instance_array = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)
    for sentence1_ind in range(max_n_sentences):
        for sentence2_ind in range(sentence1_ind, max_n_sentences):

            relnet_instance_cores = append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind)
            relnet_offset += 1

            # here we perform the connections between the selected LSNN cores and the current relnet instance cores

            def connect_input_to_relnet(input_relnet_cores: Tuple[RelnetInitCoreTuple], relnet_cores: Tuple[RelnetIntermediateCoreTuple]):
                """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
                """
                connection_tuple = []
                for presyn_core in input_relnet_cores:
                    for postsyn_core in relnet_cores:
                        presyn_neuron_range = slice(presyn_core.start, presyn_core.end)
                        postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
                        connection_tuple.append(
                            RelnetIntermediateConnectionTuple(presyn_core=presyn_core.id,
                                                              postsyn_core=postsyn_core.id,
                                                              mask=relnet_inp_mask[presyn_neuron_range, postsyn_neuron_range],
                                                              weights=relnet_inp_weights[presyn_neuron_range, postsyn_neuron_range],
                                                              delays=relnet_inp_delays[presyn_neuron_range, postsyn_neuron_range],
                                                              weight_exp=0,
                                                              sentence1_ind=presyn_core.sentence1_ind,
                                                              sentence2_ind=presyn_core.sentence2_ind))
                return tuple(connection_tuple)

            input_relnet_instance_cores = input_relnet_core_connection_array[sentence1_ind, sentence2_ind].cores
            relnet_instance_connections = connect_input_to_relnet(input_relnet_instance_cores, relnet_instance_cores)

            relnet_instance_array[sentence1_ind, sentence2_ind] = RelnetIntermediateInstanceStruct(cores=relnet_instance_cores,
                                                                                                   connections=relnet_instance_connections)
    current_core_id = n_relnet_cores * (max_n_sentences*(max_n_sentences+1))//2 + core_index_start
    assert relnet_instance_array[-1, -1].cores[-1].id == current_core_id - 1, \
        "There appears to be an issue with the assignment of core indices"

    return relnet_instance_array, current_core_id


def get_translation_layer_placement(translation_layer_inp_weights, input_relnet_core_connection_array,
                                    neurons_per_core, max_n_sentences,
                                    layer_name, core_index_start):
    """
    create n_translation_cores cores
        assign neuron indices to them

    for each 0 <= sentence1_ind <= sentence2_ind < max_n_sentences:
        connect relnet core corresponding to sentence1_ind, sentence2_ind
    """

    relnet_layer_size = input_relnet_core_connection_array[0, 0].cores[-1].end
    translation_layer_size = relnet_layer_size

    assert translation_layer_inp_weights.shape == (translation_layer_size,), \
        "Size mismatch in translation layer weights"

    n_translation_cores = int(np.ceil(translation_layer_size / neurons_per_core))

    current_core_id = core_index_start

    ## Reshaping the synaptic parameter matrices
    translation_layer_inp_mask = np.diag(np.ones(translation_layer_size, dtype=np.int64))
    translation_layer_inp_weights = np.diag(translation_layer_inp_weights)
    translation_layer_inp_delays = np.zeros(translation_layer_inp_weights.shape)

    translation_layer_inp_mask.setflags(write=False)
    translation_layer_inp_weights.setflags(write=False)
    translation_layer_inp_delays.setflags(write=False)

    ## Assigning the cores of the translation layer
    translation_layer_cores = []
    start_neuron_index = 0
    for i in range(n_translation_cores):
        end_neuron_index = min(start_neuron_index + neurons_per_core, translation_layer_size)
        translation_layer_cores.append(TranslationLayerCoreTuple(id=current_core_id,
                                                                 layer_name=layer_name,
                                                                 start=start_neuron_index,
                                                                 end=end_neuron_index))
        current_core_id += 1
        start_neuron_index = end_neuron_index

        if i == n_translation_cores - 1:
            assert start_neuron_index == translation_layer_size, \
                "Something went wrong with the assignment of the relnet to cores in layer {}".format(layer_name)

    del start_neuron_index, end_neuron_index, i  # delete loop variables

    ## Assigning the connections from the relational layer

    def connect_input_to_translation_layer(input_relnet_cores: Tuple[RelnetIntermediateCoreTuple], translation_layer_cores: Tuple[TranslationLayerCoreTuple]):
        """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
        """
        connection_tuple = []
        relnet_core_cursor = 0
        tl_core_cursor = 0
        neurons_mapped = 0
        neurons_connected = np.zeros(translation_layer_size)  # used for assert

        while neurons_mapped < translation_layer_size:
            presyn_core = input_relnet_cores[relnet_core_cursor]
            postsyn_core = translation_layer_cores[tl_core_cursor]

            if presyn_core.start >= postsyn_core.end or postsyn_core.start >= presyn_core.end:
                assert False, "This situation should Never happen"

            if presyn_core.end <= postsyn_core.end:
                relnet_core_cursor += 1
            if presyn_core.end >= postsyn_core.end:
                tl_core_cursor += 1

            neurons_mapped = min(presyn_core.end, postsyn_core.end)

            presyn_neuron_range = slice(presyn_core.start, presyn_core.end)
            postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)

            connection_tuple.append(
                TranslationLayerConnectionTuple(presyn_core=presyn_core.id,
                                                postsyn_core=postsyn_core.id,
                                                mask=translation_layer_inp_mask[presyn_neuron_range, postsyn_neuron_range],
                                                weights=translation_layer_inp_weights[presyn_neuron_range, postsyn_neuron_range],
                                                delays=translation_layer_inp_delays[presyn_neuron_range, postsyn_neuron_range],
                                                weight_exp=0,
                                                sentence1_ind=presyn_core.sentence1_ind,
                                                sentence2_ind=presyn_core.sentence2_ind))

            # Assert related code
            actually_connected_start = max(presyn_core.start, postsyn_core.start)
            actually_connected_end = min(presyn_core.end, postsyn_core.end)
            actually_connected_range_presyn = slice(actually_connected_start - presyn_core.start,
                                                    actually_connected_end - presyn_core.start)
            actually_connected_range_postsyn = slice(actually_connected_start - postsyn_core.start,
                                                     actually_connected_end - postsyn_core.start)
            actuall_connected_submatrix = connection_tuple[-1].mask[actually_connected_range_presyn, actually_connected_range_postsyn]
            n_actually_connected_neurons = actually_connected_end - actually_connected_start
            assert np.all(actuall_connected_submatrix == np.eye(n_actually_connected_neurons)), \
                "Neurons that need to be connected are not getting connected"
            neurons_connected[actually_connected_start:actually_connected_end] += 1

        # assert to check the connectin algorithm
        assert np.all(neurons_connected == 1.),  "One to One Connection Failed For Some Reason"
        return tuple(connection_tuple)

    translation_layer_connections = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)
    for sentence1_ind in range(max_n_sentences):
        for sentence2_ind in range(sentence1_ind, max_n_sentences):
            translation_layer_connections[sentence1_ind, sentence2_ind] = \
                connect_input_to_translation_layer(input_relnet_core_connection_array[sentence1_ind, sentence2_ind].cores,
                                                   translation_layer_cores)

    translation_layer_core_connections = TranslationLayerInstanceStruct(cores=translation_layer_cores,
                                                                        connections=translation_layer_connections)
    return translation_layer_core_connections, current_core_id


def get_final_MLP_placement(final_MLP_inp_weights, final_MLP_inp_delays, input_core_connections,
                            neurons_per_core, layer_name, core_index_start):

    current_core_id = core_index_start

    final_MLP_size = final_MLP_inp_weights.shape[1]
    input_cores = input_core_connections.cores  # Tuple[FinalMLPCoreTuple]
    input_dim = input_cores[-1].end

    assert final_MLP_inp_weights.shape == (input_dim, final_MLP_size), \
        "Size mismatch in final MLP layer weights"

    n_final_MLP_cores = int(np.ceil(final_MLP_size / neurons_per_core))

    final_MLP_inp_mask = np.ones(final_MLP_inp_weights.shape, dtype=np.int64)
    final_MLP_inp_weights = final_MLP_inp_weights.view()
    final_MLP_inp_delays = final_MLP_inp_delays.view()
    final_MLP_inp_mask.setflags(write=False)
    final_MLP_inp_weights.setflags(write=False)
    final_MLP_inp_delays.setflags(write=False)

    ## Assigning the cores of the Final MLP layer
    final_MLP_layer_cores = []
    start_neuron_index = 0
    for i in range(n_final_MLP_cores):
        end_neuron_index = min(start_neuron_index + neurons_per_core, final_MLP_size)
        final_MLP_layer_cores.append(FinalMLPCoreTuple(id=current_core_id,
                                                       layer_name=layer_name,
                                                       start=start_neuron_index,
                                                       end=end_neuron_index))
        current_core_id += 1
        start_neuron_index = end_neuron_index

        if i == n_final_MLP_cores - 1:
            assert start_neuron_index == final_MLP_size, \
                "Something went wrong with the assignment of the perceptron to cores in layer {}".format(layer_name)

    del start_neuron_index, end_neuron_index, i  # delete loop variables

    final_MLP_layer_connections = []
    for presyn_core in input_cores:
        for postsyn_core in final_MLP_layer_cores:
            presyn_neuron_range = slice(presyn_core.start, presyn_core.end)
            postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
            final_MLP_layer_connections.append(
                FinalMLPConnectionTuple(presyn_core=presyn_core.id,
                                        postsyn_core=postsyn_core.id,
                                        mask=final_MLP_inp_mask[presyn_neuron_range, postsyn_neuron_range],
                                        weights=final_MLP_inp_weights[presyn_neuron_range, postsyn_neuron_range],
                                        delays=final_MLP_inp_delays[presyn_neuron_range, postsyn_neuron_range],
                                        weight_exp=0))

    final_MLP_layer_connections = tuple(final_MLP_layer_connections)

    final_MLP_layer_core_connections = FinalMLPInstanceStruct(cores=final_MLP_layer_cores,
                                                              connections=final_MLP_layer_connections)

    return final_MLP_layer_core_connections, current_core_id


def get_chip_localized_relay_relnet_init_placement(relnet_inp_weights, relnet_inp_delays,
                                                   lsnn_sentence_core_connection_array, lsnn_question_core_connection_array,
                                                   input_mask_core, input_mask_weight, input_mask_weight_exp,
                                                   relay_weight,
                                                   # Parameters for placement
                                                   neurons_per_core, max_n_sentences,
                                                   # Other fundamental parameters
                                                   relnet_layer_name, LSNN_sentence_layer_name, LSNN_question_layer_name,
                                                   core_index_start):

    relnet_size = relnet_inp_weights.shape[1]
    neurons_per_core = int(neurons_per_core)
    n_relnet_cores = int(np.ceil(relnet_size / neurons_per_core))

    lsnn_size_sentences = lsnn_sentence_core_connection_array[0].cores.lsnn[-1].end
    lsnn_size_questions = lsnn_question_core_connection_array.cores.lsnn[-1].end

    LSNN_sentence_relay_delays = np.zeros(lsnn_size_sentences, dtype=np.float32)
    LSNN_sentence_relay_weights = relay_weight*np.ones(lsnn_size_sentences, dtype=np.float32)
    LSNN_sentence_relay_delays.setflags(write=False)
    LSNN_sentence_relay_weights.setflags(write=False)

    LSNN_question_relay_delays = np.zeros(lsnn_size_questions, dtype=np.float32)
    LSNN_question_relay_weights = relay_weight*np.ones(lsnn_size_questions, dtype=np.float32)
    LSNN_question_relay_delays.setflags(write=False)
    LSNN_question_relay_weights.setflags(write=False)

    input_mask_to_relay_delays = np.zeros(max_n_sentences, dtype=np.float32)
    input_mask_to_relay_weights = relay_weight*np.ones(max_n_sentences, dtype=np.float32)
    input_mask_to_relay_delays.setflags(write=False)
    input_mask_to_relay_weights.setflags(write=False)

    relnet_inp_mask = np.ones(relnet_inp_weights.shape, dtype=np.int64)  # type: np.ndarray
    relnet_inp_weights = relnet_inp_weights.view()  # type: np.ndarray
    relnet_inp_delays = relnet_inp_delays.view()  # type: np.ndarray
    lsnn_sentence_core_connection_array = lsnn_sentence_core_connection_array.view()  # type: np.ndarray
    relnet_inp_mask.setflags(write=False)
    relnet_inp_weights.setflags(write=False)
    relnet_inp_delays.setflags(write=False)
    lsnn_sentence_core_connection_array.setflags(write=False)

    assert n_relnet_cores == 4, "This function is hard-coded for n_relnet_cores == 4"
    assert max_n_sentences == 20, "This function is hard-coded for max_n_sentences == 20"

    # calculating starting chip index
    # if core_index_start = N_CORES_PER_CHIP*k => chip index = k (since it is that starting index of the chip)
    # if core_index_start = N_CORES_PER_CHIP*k + l => chip index = k+1 (since we now choose the next empty chip)
    starting_chip_index = int(np.ceil(core_index_start / N_CORES_PER_CHIP))

    # ================================================================================
    # Assigning the different relnet instances to the different blocks (block == chip)
    # ================================================================================
    tuple_of_blocks = []
    n_relay_copies_sentence = np.zeros(max_n_sentences, dtype=np.int64)
    n_relay_copies_question = 0
    for i in range(0, 4):
        for j in range(i, 4):
            current_block = []
            # each (i, j refers to a single block)
            sentence1_range = range(i*5, (i+1)*5)
            sentence2_range = range(j*5, (j+1)*5)

            current_block = cart_product(sentence1_range, sentence2_range)
            current_block = tuple((s1, s2) for s1, s2 in current_block if s1 <= s2)

            unique_input_sentences = set(iter_chain(sentence1_range, sentence2_range))
            for s in unique_input_sentences:
                n_relay_copies_sentence[s] += 1
            n_relay_copies_question += 1
            tuple_of_blocks.append(current_block)
    del i, j, unique_input_sentences, sentence1_range, sentence2_range, current_block
    tuple_of_blocks = tuple(tuple_of_blocks)

    assert n_relay_copies_question == 10 and np.all(n_relay_copies_sentence == 4), \
        "Something is wrong with the relay copy count"

    # =========================================================
    # For each chip assigining the relay cores and relnet cores
    # =========================================================
    sentence_relay_copies = defaultdict(list)
    question_relay_copies = []
    sentence_lsnn_to_relay_connections = defaultdict(list)
    question_lsnn_to_relay_connections = []
    input_mask_relay_cores = []
    input_mask_to_relay_connections = []
    relnet_instance_array = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)

    def allocate_cores_for_layer(n_neurons, n_cores, current_core_index, layer_name, core_named_tuple, **extra_core_args):
        relay_core_tuple = []
        start_neuron_index = 0
        for i in range(n_cores):
            end_neuron_index = start_neuron_index + almost_even_distribution(n_neurons, n_cores, i)
            if i == n_cores - 1:
                assert end_neuron_index == n_neurons
            relay_core_tuple.append(core_named_tuple(id=current_core_index,
                                                     layer_name=layer_name,
                                                     start=start_neuron_index,
                                                     end=end_neuron_index,
                                                     **extra_core_args))  # fanout_copy_ind doesn't make much sense in this context
            current_core_index += 1
            start_neuron_index = end_neuron_index
        relay_core_tuple = tuple(relay_core_tuple)

        return relay_core_tuple, current_core_index

    def connect_one_one(presyn_core_tuple, postsyn_core_tuple, weights, delays, connection_named_tuple, **extra_connection_args):

        layer_len = presyn_core_tuple[-1].end
        assert presyn_core_tuple[-1].end == postsyn_core_tuple[-1].end == layer_len, \
            "Number of neurons must match for one-one connection"

        assert weights.ndim == delays.ndim == 1 and weights.shape[0] == delays.shape[0] == layer_len
        weights = np.diag(weights)
        delays = np.diag(delays)
        mask = np.eye(layer_len, dtype=np.int64)

        connection_tuple = []
        for presyncore in presyn_core_tuple:
            for postsyncore in postsyn_core_tuple:
                presyn_core_id = presyncore.id
                postsyn_core_id = postsyncore.id
                presyn_neuron_range = slice(presyncore.start, presyncore.end)
                postsyn_neuron_range = slice(postsyncore.start, postsyncore.end)
                common_range_start = max(presyncore.start, postsyncore.start)
                common_range_end = min(presyncore.end, postsyncore.end)

                # This conditions is because if th's a one-one connections we only
                # connect if there's a common range in the presynaptic and
                # postsynaptic neuron groups
                if common_range_end > common_range_start:
                    connection = \
                        connection_named_tuple(presyn_core=presyn_core_id,
                                               postsyn_core=postsyn_core_id,
                                               mask=mask[presyn_neuron_range, postsyn_neuron_range],
                                               weights=weights[presyn_neuron_range, postsyn_neuron_range],
                                               delays=delays[presyn_neuron_range, postsyn_neuron_range],
                                               weight_exp=0,
                                               **extra_connection_args)
                    connection_tuple.append(connection)

        connection_tuple = tuple(connection_tuple)
        return connection_tuple

    def connect_input_to_relnet(input_cores: Tuple[LSNNCoreTuple], relnet_cores: Tuple[RelnetInitCoreTuple], mask, weights, delays, weight_exp):
        """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
        """
        connection_tuple = []
        for presyn_core in input_cores:
            for postsyn_core in relnet_cores:
                presyn_neuron_range = slice(presyn_core.start, presyn_core.end)
                postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
                connection_tuple.append(
                    RelnetInitConnectionTuple(presyn_core=presyn_core.id,
                                              postsyn_core=postsyn_core.id,
                                              mask=mask[presyn_neuron_range, postsyn_neuron_range],
                                              weights=weights[presyn_neuron_range, postsyn_neuron_range],
                                              delays=delays[presyn_neuron_range, postsyn_neuron_range],
                                              weight_exp=weight_exp,
                                              sentence1_ind=sentence1_ind,
                                              sentence2_ind=sentence2_ind))
        return tuple(connection_tuple)

    def connect_mask_to_relnet(mask_core, relnet_cores: Tuple[RelnetInitCoreTuple], sentence1_ind, sentence2_ind, weight_value, weight_exp):
        """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
        """
        connection_tuple = []
        weights = np.zeros((max_n_sentences, relnet_size))
        weights[np.array([sentence1_ind, sentence2_ind]), :] = weight_value  # maximum negative weight
        mask = (weights != 0).astype(np.int64)  # maximum negative weight
        delays = np.zeros((max_n_sentences, relnet_size))
        weights.setflags(write=False)
        mask.setflags(write=False)
        delays.setflags(write=False)

        for postsyn_core in relnet_cores:
            presyn_neuron_range = slice(None)
            postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
            connection_tuple.append(
                RelnetInitConnectionTuple(presyn_core=mask_core.id,
                                          postsyn_core=postsyn_core.id,
                                          mask=mask[presyn_neuron_range, postsyn_neuron_range],
                                          weights=weights[presyn_neuron_range, postsyn_neuron_range],
                                          delays=delays[presyn_neuron_range, postsyn_neuron_range],
                                          weight_exp=weight_exp,
                                          sentence1_ind=sentence1_ind,
                                          sentence2_ind=sentence2_ind))
        return tuple(connection_tuple)

    current_chip_index = starting_chip_index
    for block in tuple_of_blocks:
        sentence_fanout = np.zeros(max_n_sentences, dtype=np.int64)
        question_fanout = 0
        current_core_index = current_chip_index * N_CORES_PER_CHIP

        # ----------------------------------------------------------------
        # Calculating the fanouts for the different sentences and question
        # ----------------------------------------------------------------
        for s1, s2 in block:
            sentence_fanout[s1] += 1
            sentence_fanout[s2] += 1
            question_fanout += 1
        del s1, s2

        # ---------------------------------------------------------
        # Finding the number of cores needed for the relay networks
        # ---------------------------------------------------------
        # sentence relay networks
        sentence_relay_network_data = []
        for sent in range(max_n_sentences):
            if sentence_fanout[sent] > 0:
                # from fanout calculate neurons_per_core of corresponding relay
                max_relay_neurons_per_core = OUTPUT_AXON_LIMIT_WITHIN_CHIP // (sentence_fanout[sent] * n_relnet_cores)
                n_relay_cores = int(np.ceil(lsnn_size_sentences / max_relay_neurons_per_core))
                relay_neurons_per_core = int(np.ceil(lsnn_size_sentences / n_relay_cores))

                sentence_relay_network_data.append((sent, n_relay_cores))
        del sent

        # question relay networks
        max_relay_neurons_per_core = OUTPUT_AXON_LIMIT_WITHIN_CHIP // (question_fanout * n_relnet_cores)
        n_relay_cores = int(np.ceil(lsnn_size_questions / max_relay_neurons_per_core))
        relay_neurons_per_core = int(np.ceil(lsnn_size_questions / n_relay_cores))
        question_n_relay_cores = n_relay_cores

        del max_relay_neurons_per_core, n_relay_cores, relay_neurons_per_core

        # Assert that given all of these cores, the block can be fit into the chip
        # If the manual optimization is correct this should not be triggered
        total_relay_cores = sum(x[1] for x in sentence_relay_network_data) + question_n_relay_cores
        assert total_relay_cores + n_relnet_cores * len(block) <= 128, "Block Cannot be fit onto chip"

        # ----------------------------------------------------------
        # Allocating the cores and connections for the relay network
        # ----------------------------------------------------------

        # sentences
        for sent, n_relay_cores in sentence_relay_network_data:

            # allocating the core ids for the sentences relay copy
            relay_core_tuple, current_core_index = \
                allocate_cores_for_layer(lsnn_size_sentences, n_relay_cores, current_core_index,
                                         '{}_relay'.format(LSNN_sentence_layer_name),
                                         LSNNCoreTuple,
                                         sentence_ind=sent,
                                         fanout_copy_ind=None)

            lsnn_to_relay_connections = connect_one_one(lsnn_sentence_core_connection_array[sent].cores.lsnn,
                                                        relay_core_tuple,
                                                        LSNN_sentence_relay_weights,
                                                        LSNN_sentence_relay_delays,
                                                        LSNNConnectionTuple,
                                                        sentence_ind=sent,
                                                        fanout_copy_ind=None)

            sentence_relay_copies[sent].append(relay_core_tuple)
            sentence_lsnn_to_relay_connections[sent].extend(lsnn_to_relay_connections)
        del sent, n_relay_cores

        # questions
        relay_core_tuple, current_core_index = \
            allocate_cores_for_layer(lsnn_size_questions, question_n_relay_cores, current_core_index,
                                     '{}_relay'.format(LSNN_question_layer_name),
                                     LSNNCoreTuple,
                                     sentence_ind=0,
                                     fanout_copy_ind=None)

        lsnn_to_relay_connections = connect_one_one(lsnn_question_core_connection_array.cores.lsnn,
                                                    relay_core_tuple,
                                                    LSNN_question_relay_weights,
                                                    LSNN_question_relay_delays,
                                                    LSNNConnectionTuple,
                                                    sentence_ind=0,
                                                    fanout_copy_ind=None)
        question_relay_copies.append(relay_core_tuple)
        question_lsnn_to_relay_connections.extend(lsnn_to_relay_connections)

        # mask neurons
        mask_relay_core = LSNNCoreTuple(id=current_core_index,
                                        layer_name='LSNN_mask_relay',
                                        start=0, end=max_n_sentences,
                                        sentence_ind=None, fanout_copy_ind=None)

        input_mask_relay_cores.append(mask_relay_core)
        input_mask_to_relay_connections.append(connect_one_one((input_mask_core,), (mask_relay_core,),
                                                               input_mask_to_relay_weights,
                                                               input_mask_to_relay_delays,
                                                               LSNNConnectionTuple,
                                                               sentence_ind=None,
                                                               fanout_copy_ind=None))

        current_core_index += 1

        del relay_core_tuple

        # ----------------------------------------------------------------------
        # Allocating cores and connections for the relational network init layer
        # ----------------------------------------------------------------------

        for sentence1_ind, sentence2_ind in block:
            # create cores for relational layer
            relnet_instance_cores, current_core_index = allocate_cores_for_layer(relnet_size, n_relnet_cores, current_core_index,
                                                                                 relnet_layer_name, RelnetInitCoreTuple,
                                                                                 sentence1_ind=sentence1_ind, sentence2_ind=sentence2_ind)

            sentence1_lsnn_cores = sentence_relay_copies[sentence1_ind][-1]
            sentence2_lsnn_cores = sentence_relay_copies[sentence2_ind][-1]
            question_lsnn_cores = question_relay_copies[-1]

            # Create connections for relational layer
            sentence1_to_relnet_conns = connect_input_to_relnet(sentence1_lsnn_cores,
                                                                relnet_instance_cores,
                                                                relnet_inp_mask[:lsnn_size_sentences],
                                                                relnet_inp_weights[:lsnn_size_sentences],
                                                                relnet_inp_delays[:lsnn_size_sentences],
                                                                0)
            sentence2_to_relnet_conns = connect_input_to_relnet(sentence2_lsnn_cores,
                                                                relnet_instance_cores,
                                                                relnet_inp_mask[lsnn_size_sentences:2*lsnn_size_sentences],
                                                                relnet_inp_weights[lsnn_size_sentences:2*lsnn_size_sentences],
                                                                relnet_inp_delays[lsnn_size_sentences:2*lsnn_size_sentences],
                                                                0)
            question_to_relnet_conns = connect_input_to_relnet(question_lsnn_cores,
                                                               relnet_instance_cores,
                                                               relnet_inp_mask[2*lsnn_size_sentences:],
                                                               relnet_inp_weights[2*lsnn_size_sentences:],
                                                               relnet_inp_delays[2*lsnn_size_sentences:],
                                                               0)

            mask_to_relnet_conns = connect_mask_to_relnet(mask_relay_core,
                                                          relnet_instance_cores,
                                                          sentence1_ind, sentence2_ind,
                                                          input_mask_weight,
                                                          input_mask_weight_exp)

            relnet_instance_connections = RelnetInitConnectionsStruct(sentence1_to_relnet=sentence1_to_relnet_conns,
                                                                      sentence2_to_relnet=sentence2_to_relnet_conns,
                                                                      question_to_relnet=question_to_relnet_conns,
                                                                      mask_to_relnet=mask_to_relnet_conns)

            assert relnet_instance_array[sentence1_ind, sentence2_ind] is None, "Relational network being reassigned"
            relnet_instance_array[sentence1_ind, sentence2_ind] = RelnetInitInstanceStruct(cores=relnet_instance_cores,
                                                                                           connections=relnet_instance_connections)

        assert (current_core_index-1) // N_CORES_PER_CHIP == current_chip_index, \
            ("We have broken across core boundaries when allocating the init layer"
             " of the relational network. The end of the universe is nigh upon us")
        current_chip_index += 1

    assert all(n_copies == len(sentence_relay_copies[sent]) for sent, n_copies in enumerate(n_relay_copies_sentence)), \
        "The number of sentence relay copies differs from the calculted number of relay copies"
    assert n_relay_copies_question == len(question_relay_copies), \
        "The number of question relay copies differs from the calculted number of relay copies"

    # ==============================================================
    # Create new LSNN core connection arrays with the relay networks
    # ==============================================================
    def get_new_lsnn_copy_struct(old_lsnn_copy_struct, relay_copies, lsnn_to_relay_connections):
        relay_copy_array = np.ndarray(len(relay_copies), dtype=object)
        for i, relay_copy_core_tuple in enumerate(relay_copies):
            relay_copy_array[i] = relay_copy_core_tuple

        current_lsnn_cores = old_lsnn_copy_struct.cores
        current_lsnn_connections = old_lsnn_copy_struct.connections
        lsnn_cores = LSNNCoresStruct(input=current_lsnn_cores.input,
                                     lsnn=current_lsnn_cores.lsnn,
                                     relay=relay_copy_array)
        lsnn_connections = LSNNConnectionsStruct(input_to_lsnn=current_lsnn_connections.input_to_lsnn,
                                                 lsnn_to_lsnn=current_lsnn_connections.lsnn_to_lsnn,
                                                 lsnn_to_relay=tuple(lsnn_to_relay_connections))

        return LSNNCopyStruct(cores=lsnn_cores,
                              connections=lsnn_connections)

    new_lsnn_sentence_core_connection_array = np.empty_like(lsnn_sentence_core_connection_array)
    for sent in range(max_n_sentences):
        new_sentence_lsnn_copy_struct = get_new_lsnn_copy_struct(lsnn_sentence_core_connection_array[sent],
                                                                 sentence_relay_copies[sent],
                                                                 sentence_lsnn_to_relay_connections[sent])

        new_lsnn_sentence_core_connection_array[sent] = new_sentence_lsnn_copy_struct

    new_lsnn_question_core_connection_array = get_new_lsnn_copy_struct(lsnn_question_core_connection_array,
                                                                       question_relay_copies,
                                                                       question_lsnn_to_relay_connections)
    new_lsnn_sentence_core_connection_array.setflags(write=False)
    relnet_instance_array.setflags(write=False)

    # Recasting code for input mask relay cores and connections
    input_mask_relay_cores = tuple(input_mask_relay_cores)
    input_mask_to_relay_connections = tuple(input_mask_to_relay_connections)

    return (new_lsnn_sentence_core_connection_array,
            new_lsnn_question_core_connection_array,
            input_mask_relay_cores,
            input_mask_to_relay_connections,
            relnet_instance_array,
            current_core_index), tuple_of_blocks


def get_chip_localized_relnet_intermediate_placement(relnet_inp_weights_list, relnet_inp_delays_list,
                                                     input_relnet_core_connection_array,
                                                     # Parameters for placement
                                                     neurons_per_core_list, max_n_sentences,
                                                     # Other fundamental parameters
                                                     layer_names_list, core_index_start):

    n_layers = len(relnet_inp_weights_list)

    assert len(relnet_inp_weights_list) == n_layers
    assert len(relnet_inp_delays_list) == n_layers, "Mismatch in number of layers specified"
    assert len(neurons_per_core_list) == n_layers, "Mismatch in number of layers specified"
    assert len(layer_names_list) == n_layers, "Mismatch in number of layers specified"

    assert all(x.shape == y.shape for x, y in zip(relnet_inp_weights_list, relnet_inp_delays_list))

    relnet_size_list = [x.shape[1] for x in relnet_inp_weights_list]
    cores_per_instance_list = [int(np.ceil(x//y)) for x, y in zip(relnet_size_list, neurons_per_core_list)]
    total_cores_per_instance = sum(cores_per_instance_list)
    n_instances_per_chip = N_CORES_PER_CHIP // total_cores_per_instance

    # calculating starting chip index
    # if core_index_start = N_CORES_PER_CHIP*k => chip index = k (since it is that starting index of the chip)
    # if core_index_start = N_CORES_PER_CHIP*k + l => chip index = k+1 (since we now choose the next empty chip)
    starting_chip_index = int(np.ceil(core_index_start / N_CORES_PER_CHIP))
    relnet_instance_array_list = []
    input_relnet_core_connection_array = input_relnet_core_connection_array.view()  # type: np.ndarray
    input_relnet_core_connection_array.setflags(write=False)

    for layer_ind in range(n_layers):
        relnet_inp_weights = relnet_inp_weights_list[layer_ind]
        relnet_inp_delays = relnet_inp_delays_list[layer_ind]
        neurons_per_core = neurons_per_core_list[layer_ind]
        layer_name = layer_names_list[layer_ind]
        relnet_size = relnet_size_list[layer_ind]

        # disable write on any of these
        relnet_inp_mask = np.ones(relnet_inp_weights.shape, dtype=np.int64)  # type: np.ndarray
        relnet_inp_weights = relnet_inp_weights.view()  # type: np.ndarray
        relnet_inp_delays = relnet_inp_delays.view()  # type: np.ndarray
        relnet_inp_mask.setflags(write=False)
        relnet_inp_weights.setflags(write=False)
        relnet_inp_delays.setflags(write=False)

        # ==================================================================================================
        # Get the parameters corresponding to the optimal placement of the first layer of relational network
        # ==================================================================================================

        neurons_per_core = int(neurons_per_core)

        n_relnet_cores = int(np.ceil(relnet_size / neurons_per_core))

        # ========================================================================
        # Assigning the Cores for each a sample instance of the relational network
        # ========================================================================

        current_core_id = sum(cores_per_instance_list[:layer_ind]) + starting_chip_index*N_CORES_PER_CHIP

        # Assigning the cores for the sample relnet instance
        relnet_cores = []
        start_neuron_index = 0
        for i in range(n_relnet_cores):
            end_neuron_index = min(start_neuron_index + neurons_per_core, relnet_size)
            relnet_cores.append(RelnetIntermediateCoreTuple(id=current_core_id,
                                                            layer_name=layer_name,
                                                            start=start_neuron_index,
                                                            end=end_neuron_index,
                                                            sentence1_ind=0,
                                                            sentence2_ind=0))
            current_core_id += 1
            start_neuron_index = end_neuron_index

            if i == n_relnet_cores - 1:
                assert start_neuron_index == relnet_size, \
                    "Something went wrong with the assignment of the relnet to cores in layer {}".format(layer_name)

        del start_neuron_index, end_neuron_index, i  # delete loop variables

        # Converting lists to tuple (immutability)
        relnet_cores = tuple(relnet_cores)

        # ===========================================================================
        # Assigning the Cores and Connections for each copy of the relational network
        # ===========================================================================

        def append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind):
            current_chip_offset = relnet_offset // n_instances_per_chip
            instance_index_in_chip = relnet_offset % n_instances_per_chip
            offset = current_chip_offset*N_CORES_PER_CHIP + instance_index_in_chip*total_cores_per_instance

            return tuple(RelnetIntermediateCoreTuple(id=core.id + offset,
                                                     layer_name=core.layer_name,
                                                     start=core.start,
                                                     end=core.end,
                                                     sentence1_ind=sentence1_ind,
                                                     sentence2_ind=sentence2_ind)
                         for core in relnet_cores)

        def connect_input_to_relnet(input_relnet_cores: Tuple[RelnetInitCoreTuple], relnet_cores: Tuple[RelnetIntermediateCoreTuple]):
            """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
            """
            connection_tuple = []
            for presyn_core in input_relnet_cores:
                for postsyn_core in relnet_cores:
                    presyn_neuron_range = slice(presyn_core.start, presyn_core.end)
                    postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
                    connection_tuple.append(
                        RelnetIntermediateConnectionTuple(presyn_core=presyn_core.id,
                                                          postsyn_core=postsyn_core.id,
                                                          mask=relnet_inp_mask[presyn_neuron_range, postsyn_neuron_range],
                                                          weights=relnet_inp_weights[presyn_neuron_range, postsyn_neuron_range],
                                                          delays=relnet_inp_delays[presyn_neuron_range, postsyn_neuron_range],
                                                          weight_exp=0,
                                                          sentence1_ind=presyn_core.sentence1_ind,
                                                          sentence2_ind=presyn_core.sentence2_ind))
            return tuple(connection_tuple)

        relnet_offset = 0  # Number of relnets that have been processed
        relnet_instance_array = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)
        for sentence1_ind in range(max_n_sentences):
            for sentence2_ind in range(sentence1_ind, max_n_sentences):

                relnet_instance_cores = append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind)
                relnet_offset += 1

                # here we perform the connections between the selected LSNN cores and the current relnet instance cores

                input_relnet_instance_cores = input_relnet_core_connection_array[sentence1_ind, sentence2_ind].cores
                relnet_instance_connections = connect_input_to_relnet(input_relnet_instance_cores, relnet_instance_cores)

                relnet_instance_array[sentence1_ind, sentence2_ind] = RelnetIntermediateInstanceStruct(cores=relnet_instance_cores,
                                                                                                       connections=relnet_instance_connections)

        # current_core_id = n_relnet_cores * (max_n_sentences*(max_n_sentences+1))//2 + core_index_start
        n_relnet_instances = (max_n_sentences*(max_n_sentences+1))//2
        final_chip_index = (n_relnet_instances-1)//n_instances_per_chip + starting_chip_index
        final_instance_ind_in_chip = (n_relnet_instances-1) % n_instances_per_chip
        assert relnet_instance_array[-1, -1].cores[-1].id == final_chip_index*N_CORES_PER_CHIP + final_instance_ind_in_chip*total_cores_per_instance + sum(cores_per_instance_list[:layer_ind+1]) - 1, \
            "There appears to be an issue with the assignment of core indices"

        relnet_instance_array.setflags(write=False)
        input_relnet_core_connection_array = relnet_instance_array
        relnet_instance_array_list.append(relnet_instance_array)

    final_core_id = final_chip_index*N_CORES_PER_CHIP + final_instance_ind_in_chip*total_cores_per_instance + sum(cores_per_instance_list)
    assert relnet_instance_array_list[-1][-1, -1].cores[-1].id == final_core_id - 1, \
        "There appears to be an issue with the assignment of core indices"

    return relnet_instance_array_list, final_core_id


def get_chip_localized_optimum_relnet_intermediate_placement(relnet_inp_weights_list, relnet_inp_delays_list,
                                                             input_relnet_core_connection_array,
                                                             # This is returned by get_chip_localized_relay_relnet_init_placement
                                                             tuple_of_blocks,
                                                             # Parameters for placement
                                                             neurons_per_core_list, max_n_sentences,
                                                             # Other fundamental parameters
                                                             layer_names_list, core_index_start):

    n_layers = len(relnet_inp_weights_list)

    assert len(relnet_inp_weights_list) == n_layers
    assert len(relnet_inp_delays_list) == n_layers, "Mismatch in number of layers specified"
    assert len(neurons_per_core_list) == n_layers, "Mismatch in number of layers specified"
    assert len(layer_names_list) == n_layers, "Mismatch in number of layers specified"

    assert all(x.shape == y.shape for x, y in zip(relnet_inp_weights_list, relnet_inp_delays_list))

    relnet_size_list = [x.shape[1] for x in relnet_inp_weights_list]
    cores_per_instance_list = [int(np.ceil(x//y)) for x, y in zip(relnet_size_list, neurons_per_core_list)]
    total_cores_per_instance = sum(cores_per_instance_list)
    n_instances_per_chip = N_CORES_PER_CHIP // total_cores_per_instance

    # calculating starting chip index
    # if core_index_start = N_CORES_PER_CHIP*k => chip index = k (since it is that starting index of the chip)
    # if core_index_start = N_CORES_PER_CHIP*k + l => chip index = k+1 (since we now choose the next empty chip)
    starting_chip_index = int(np.ceil(core_index_start / N_CORES_PER_CHIP))
    relnet_instance_array_list = []
    input_relnet_core_connection_array = input_relnet_core_connection_array.view()  # type: np.ndarray
    input_relnet_core_connection_array.setflags(write=False)

    def append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind):
        current_chip_offset = relnet_offset // n_instances_per_chip
        instance_index_in_chip = relnet_offset % n_instances_per_chip
        offset = current_chip_offset*N_CORES_PER_CHIP + instance_index_in_chip*total_cores_per_instance

        return tuple(RelnetIntermediateCoreTuple(id=core.id + offset,
                                                 layer_name=core.layer_name,
                                                 start=core.start,
                                                 end=core.end,
                                                 sentence1_ind=sentence1_ind,
                                                 sentence2_ind=sentence2_ind)
                     for core in relnet_cores)

    def connect_input_to_relnet(input_relnet_cores: Tuple[RelnetInitCoreTuple], relnet_cores: Tuple[RelnetIntermediateCoreTuple]):
        """This function connects a particular copy of the LSNN to the relnet instance using the specified weights and delays
        """
        connection_tuple = []
        for presyn_core in input_relnet_cores:
            for postsyn_core in relnet_cores:
                presyn_neuron_range = slice(presyn_core.start, presyn_core.end)
                postsyn_neuron_range = slice(postsyn_core.start, postsyn_core.end)
                connection_tuple.append(
                    RelnetIntermediateConnectionTuple(presyn_core=presyn_core.id,
                                                      postsyn_core=postsyn_core.id,
                                                      mask=relnet_inp_mask[presyn_neuron_range, postsyn_neuron_range],
                                                      weights=relnet_inp_weights[presyn_neuron_range, postsyn_neuron_range],
                                                      delays=relnet_inp_delays[presyn_neuron_range, postsyn_neuron_range],
                                                      weight_exp=0,
                                                      sentence1_ind=presyn_core.sentence1_ind,
                                                      sentence2_ind=presyn_core.sentence2_ind))
        return tuple(connection_tuple)

    for layer_ind in range(n_layers):
        relnet_inp_weights = relnet_inp_weights_list[layer_ind]
        relnet_inp_delays = relnet_inp_delays_list[layer_ind]
        neurons_per_core = neurons_per_core_list[layer_ind]
        layer_name = layer_names_list[layer_ind]
        relnet_size = relnet_size_list[layer_ind]

        # disable write on any of these
        relnet_inp_mask = np.ones(relnet_inp_weights.shape, dtype=np.int64)  # type: np.ndarray
        relnet_inp_weights = relnet_inp_weights.view()  # type: np.ndarray
        relnet_inp_delays = relnet_inp_delays.view()  # type: np.ndarray
        relnet_inp_mask.setflags(write=False)
        relnet_inp_weights.setflags(write=False)
        relnet_inp_delays.setflags(write=False)

        # ==================================================================================================
        # Get the parameters corresponding to the optimal placement of the first layer of relational network
        # ==================================================================================================

        neurons_per_core = int(neurons_per_core)

        n_relnet_cores = int(np.ceil(relnet_size / neurons_per_core))

        # ========================================================================
        # Assigning the Cores for each a sample instance of the relational network
        # ========================================================================

        current_core_id = sum(cores_per_instance_list[:layer_ind]) + starting_chip_index*N_CORES_PER_CHIP

        # Assigning the cores for the sample relnet instance
        relnet_cores = []
        start_neuron_index = 0
        for i in range(n_relnet_cores):
            end_neuron_index = min(start_neuron_index + neurons_per_core, relnet_size)
            relnet_cores.append(RelnetIntermediateCoreTuple(id=current_core_id,
                                                            layer_name=layer_name,
                                                            start=start_neuron_index,
                                                            end=end_neuron_index,
                                                            sentence1_ind=0,
                                                            sentence2_ind=0))
            current_core_id += 1
            start_neuron_index = end_neuron_index

            if i == n_relnet_cores - 1:
                assert start_neuron_index == relnet_size, \
                    "Something went wrong with the assignment of the relnet to cores in layer {}".format(layer_name)

        del start_neuron_index, end_neuron_index, i  # delete loop variables

        # Converting lists to tuple (immutability)
        relnet_cores = tuple(relnet_cores)

        # ===========================================================================
        # Assigning the Cores and Connections for each copy of the relational network
        # ===========================================================================

        relnet_offset = 0  # Number of relnets that have been processed
        relnet_instance_array = np.ndarray((max_n_sentences, max_n_sentences), dtype=object)
        for block in tuple_of_blocks:
            for (sentence1_ind, sentence2_ind) in block:
                relnet_instance_cores = append_offset_to_cores(relnet_cores, relnet_offset, sentence1_ind, sentence2_ind)
                relnet_offset += 1

                # here we perform the connections between the selected LSNN cores and the current relnet instance cores

                input_relnet_instance_cores = input_relnet_core_connection_array[sentence1_ind, sentence2_ind].cores
                relnet_instance_connections = connect_input_to_relnet(input_relnet_instance_cores, relnet_instance_cores)

                relnet_instance_array[sentence1_ind, sentence2_ind] = RelnetIntermediateInstanceStruct(cores=relnet_instance_cores,
                                                                                                       connections=relnet_instance_connections)

        # current_core_id = n_relnet_cores * (max_n_sentences*(max_n_sentences+1))//2 + core_index_start
        n_relnet_instances = (max_n_sentences*(max_n_sentences+1))//2
        final_chip_index = (n_relnet_instances-1)//n_instances_per_chip + starting_chip_index
        final_instance_ind_in_chip = (n_relnet_instances-1) % n_instances_per_chip
        assert relnet_instance_array[-1, -1].cores[-1].id == final_chip_index*N_CORES_PER_CHIP + final_instance_ind_in_chip*total_cores_per_instance + sum(cores_per_instance_list[:layer_ind+1]) - 1, \
            "There appears to be an issue with the assignment of core indices"

        relnet_instance_array.setflags(write=False)
        input_relnet_core_connection_array = relnet_instance_array
        relnet_instance_array_list.append(relnet_instance_array)

    final_core_id = final_chip_index*N_CORES_PER_CHIP + final_instance_ind_in_chip*total_cores_per_instance + sum(cores_per_instance_list)
    assert relnet_instance_array_list[-1][-1, -1].cores[-1].id == final_core_id - 1, \
        "There appears to be an issue with the assignment of core indices"

    return relnet_instance_array_list, final_core_id