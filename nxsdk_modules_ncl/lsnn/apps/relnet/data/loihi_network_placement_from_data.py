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
import pickle
import argparse

from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import get_minimum_n_cores_for_LSNN_placement_no_ws
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import get_placement_param_list_relnet_input_no_ws
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import get_placement_param_list_relnet_intermediate_no_ws
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import get_placement_param_list_translation_layer_one_one_no_ws
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import get_placement_param_list_final_MLP_no_ws
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import INPUT_AXON_LIMIT
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import OUTPUT_AXON_LIMIT
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import OUTPUT_AXON_PER_NEURON_LIMIT
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import MAX_TOTAL_NEURONS_IN_CORE
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import MAX_N_SENTENCES
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import MAX_SYNAPSES_FF
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import MAX_SYNAPSES_LSNN
from nxsdk_modules.lsnn.apps.relnet.data.loihi_placement_param_search import MAX_SYNAPSES_TRANSLATION
                                        
from nxsdk_modules.lsnn.apps.relnet.data.loihi_core_allocation_functions import get_LSNN_placement
from nxsdk_modules.lsnn.apps.relnet.data.loihi_core_allocation_functions import get_input_mask_placement
from nxsdk_modules.lsnn.apps.relnet.data.loihi_core_allocation_functions import get_relnet_init_placement
from nxsdk_modules.lsnn.apps.relnet.data.loihi_core_allocation_functions import get_relnet_intermediate_placement
from nxsdk_modules.lsnn.apps.relnet.data.loihi_core_allocation_functions import get_translation_layer_placement
from nxsdk_modules.lsnn.apps.relnet.data.loihi_core_allocation_functions import get_final_MLP_placement


def get_relational_network_placement_on_loihi(cell_data, use_cores_for_input=False):

    input_mask_weight = -100.
    input_mask_weight_exp = 7
    relay_weight = 5.0

    stories_LSNN_cell_data = cell_data['stories_LSNN_cell_data']
    queries_LSNN_cell_data = cell_data['queries_LSNN_cell_data']
    relational_function_cell_data = cell_data['relational_function_cell_data']
    quantization_cell_data = cell_data['quantization_cell_data']
    final_MLP_cell_data = cell_data['final_MLP_cell_data']
    readout_cell_data = cell_data['readout_cell_data']

    stories_LSNN_input_weights = stories_LSNN_cell_data['w_in_val']
    stories_LSNN_input_delays = stories_LSNN_cell_data['w_in_delay']
    stories_LSNN_rec_weights = stories_LSNN_cell_data['w_rec_val']
    stories_LSNN_rec_delays = stories_LSNN_cell_data['w_rec_delay']
    queries_LSNN_input_weights = queries_LSNN_cell_data['w_in_val']
    queries_LSNN_input_delays = queries_LSNN_cell_data['w_in_delay']
    queries_LSNN_rec_weights = queries_LSNN_cell_data['w_rec_val']
    queries_LSNN_rec_delays = queries_LSNN_cell_data['w_rec_delay']
    relational_function_input_weights = tuple(x['w_in_val'] for x in relational_function_cell_data)
    relational_function_input_delays = tuple(x['w_in_delay'] for x in relational_function_cell_data)
    translation_layer_input_weights = quantization_cell_data['w_in_val']  # ugghh name changes suuuck
    final_MLP_input_weights = tuple(x['w_in_val'] for x in final_MLP_cell_data)
    final_MLP_input_delays = tuple(x['w_in_delay'] for x in final_MLP_cell_data)
    readout_weights = readout_cell_data['w_in_val']

    input_dim = stories_LSNN_cell_data['n_in']
    readout_dim = readout_cell_data['n_rec']
    LSNN_sentence_size = stories_LSNN_cell_data['n_rec']
    LSNN_question_size = queries_LSNN_cell_data['n_rec']
    relnet_units = [x['n_rec'] for x in relational_function_cell_data]
    final_MLP_units = [x['n_rec'] for x in final_MLP_cell_data]

    # need this for an inconsequential assert
    stories_LSNN_rec_delays[np.eye(LSNN_sentence_size) == 1] = 0
    queries_LSNN_rec_delays[np.eye(LSNN_question_size) == 1] = 0

    # Assert that all the weights are integer valued
    assert np.all(stories_LSNN_input_weights == np.round(stories_LSNN_input_weights))
    assert np.all(queries_LSNN_input_weights == np.round(queries_LSNN_input_weights))
    assert np.all(stories_LSNN_rec_weights == np.round(stories_LSNN_rec_weights))
    assert np.all(queries_LSNN_rec_weights == np.round(queries_LSNN_rec_weights))
    assert np.all(translation_layer_input_weights == np.round(translation_layer_input_weights))
    assert all(np.all(x == np.round(x)) for x in relational_function_input_weights)
    assert all(np.all(x == np.round(x)) for x in final_MLP_input_weights)
    assert np.all(readout_weights == np.round(readout_weights))

    # Assert that all the weights are between -255, 255
    assert np.all(stories_LSNN_input_weights <= 255) and np.all(stories_LSNN_input_weights >= -255)
    assert np.all(queries_LSNN_input_weights <= 255) and np.all(queries_LSNN_input_weights >= -255)
    assert np.all(stories_LSNN_rec_weights <= 255) and np.all(stories_LSNN_rec_weights >= -255)
    assert np.all(queries_LSNN_rec_weights <= 255) and np.all(queries_LSNN_rec_weights >= -255)
    assert np.all(translation_layer_input_weights <= 255) and np.all(translation_layer_input_weights >= -255)
    assert all(np.all(x <= 255) and np.all(x >= -255) for x in relational_function_input_weights)
    assert all(np.all(x <= 255) and np.all(x >= -255) for x in final_MLP_input_weights)
    assert np.all(readout_weights <= 255) and np.all(readout_weights >= -255)

    relnet_init_neurons_per_core, relnet_init_n_cores = \
        get_placement_param_list_relnet_input_no_ws(layer_size=relnet_units[0],
                                                    LSNN_sentence_size=LSNN_sentence_size,
                                                    LSNN_question_size=LSNN_question_size)

    relnet_cores_per_copy = int(np.ceil(relnet_units[0] / relnet_init_neurons_per_core))

    (sentence_n_cores_total,
     sentence_input_neurons_per_core,
     sentence_neurons_per_core,
     sentence_relay_neurons_per_core,
     sentence_n_relay_copies) = \
        get_minimum_n_cores_for_LSNN_placement_no_ws(LSNN_size=LSNN_sentence_size,
                                                     input_dim=input_dim,
                                                     relnet_cores_per_copy=relnet_cores_per_copy,
                                                     fanout_to_relnets=MAX_N_SENTENCES+1,
                                                     n_instances=MAX_N_SENTENCES,
                                                     use_cores_for_input=use_cores_for_input,
                                                     create_masking_core=True)

    (question_n_cores_total,
     question_input_neurons_per_core,
     question_neurons_per_core,
     question_relay_neurons_per_core,
     question_n_relay_copies) = \
        get_minimum_n_cores_for_LSNN_placement_no_ws(LSNN_size=LSNN_question_size,
                                                     input_dim=input_dim,
                                                     relnet_cores_per_copy=relnet_cores_per_copy,
                                                     fanout_to_relnets=((MAX_N_SENTENCES+1)*MAX_N_SENTENCES)//2,
                                                     n_instances=1,
                                                     use_cores_for_input=use_cores_for_input,
                                                     create_masking_core=False)

    relnet_intermediate_neurons_per_core = []
    relnet_intermediate_n_cores = []
    prev_relnet_output_dim = relnet_units[0]
    for units in relnet_units[1:]:
        neurons_per_core, n_cores = get_placement_param_list_relnet_intermediate_no_ws(layer_size=units,
                                                                                       input_size=prev_relnet_output_dim)
        prev_relnet_output_dim = units

        relnet_intermediate_neurons_per_core.append(neurons_per_core)
        relnet_intermediate_n_cores.append(n_cores)
    del neurons_per_core, n_cores

    translation_layer_neurons_per_core, translation_layer_n_cores = \
        get_placement_param_list_translation_layer_one_one_no_ws(prev_relnet_output_dim)

    prev_final_MLP_output_dim = prev_relnet_output_dim  # This is the output dimension of the translation layer
    final_MLP_neurons_per_core = []
    final_MLP_n_cores = []
    for i, units in enumerate(final_MLP_units):
        neurons_per_core, n_cores = \
            get_placement_param_list_final_MLP_no_ws(layer_size=units,
                                                     input_size=prev_final_MLP_output_dim)
        prev_final_MLP_output_dim = units
        final_MLP_neurons_per_core.append(neurons_per_core)
        final_MLP_n_cores.append(n_cores)
    del neurons_per_core, n_cores, i

    readout_neurons_per_core, readout_n_cores = \
        get_placement_param_list_final_MLP_no_ws(layer_size=readout_dim,
                                                 input_size=prev_final_MLP_output_dim)

    current_core_id = 0

    lsnn_sentence_core_connection_array, current_core_id = \
        get_LSNN_placement(LSNN_rec_weights=stories_LSNN_rec_weights,
                           LSNN_rec_delays=stories_LSNN_rec_delays,
                           LSNN_rec_weight_exp=stories_LSNN_cell_data['weight_exp'],
                           LSNN_inp_weights=stories_LSNN_input_weights,
                           LSNN_inp_delays=stories_LSNN_input_delays,
                           LSNN_inp_weight_exp=stories_LSNN_cell_data['weight_exp'],
                           relay_weight=relay_weight,

                           # Network Placement parameters
                           input_neurons_per_core=sentence_input_neurons_per_core,
                           neurons_per_core=sentence_neurons_per_core,
                           relay_neurons_per_core=sentence_relay_neurons_per_core,
                           relay_copies_for_fanout=sentence_n_relay_copies,
                           n_instances=MAX_N_SENTENCES,

                           # Other basic parameters
                           layer_name='LSNNSentence',
                           core_index_start=current_core_id)

    lsnn_question_core_connection_array, current_core_id = \
        get_LSNN_placement(LSNN_rec_weights=queries_LSNN_rec_weights,
                           LSNN_rec_delays=queries_LSNN_rec_delays,
                           LSNN_rec_weight_exp=queries_LSNN_cell_data['weight_exp'],
                           LSNN_inp_weights=queries_LSNN_input_weights,
                           LSNN_inp_delays=queries_LSNN_input_delays,
                           LSNN_inp_weight_exp=queries_LSNN_cell_data['weight_exp'],
                           relay_weight=relay_weight,

                           # Network Placement parameters
                           input_neurons_per_core=question_input_neurons_per_core,
                           neurons_per_core=question_neurons_per_core,
                           relay_neurons_per_core=question_relay_neurons_per_core,
                           relay_copies_for_fanout=question_n_relay_copies,
                           n_instances=0,

                           # Other basic parameters
                           layer_name='LSNNQuestion',
                           core_index_start=current_core_id)

    input_mask_core, current_core_id = \
        get_input_mask_placement(n_instances=MAX_N_SENTENCES,
                                 core_index_start=current_core_id)

    relnet_init_core_connection_array, current_core_id = \
        get_relnet_init_placement(relnet_inp_weights=relational_function_input_weights[0],
                                  relnet_inp_delays=relational_function_cell_data[0]['n_delay'] - 1 - relational_function_input_delays[0],
                                  lsnn_sentence_core_connection_array=lsnn_sentence_core_connection_array,
                                  lsnn_question_core_connection_array=lsnn_question_core_connection_array,
                                  input_mask_core=input_mask_core,
                                  input_mask_weight=input_mask_weight,
                                  input_mask_weight_exp=input_mask_weight_exp,

                                  # Network Placement parameters
                                  neurons_per_core=relnet_init_neurons_per_core,
                                  max_n_sentences=MAX_N_SENTENCES,

                                  # Other basic parameters
                                  layer_name='RelnetLayer0',
                                  core_index_start=current_core_id)

    relnet_intermediate_core_connection_array = []
    prev_relnet_core_connection_array = relnet_init_core_connection_array
    for i in range(1, len(relnet_units)):
        prev_relnet_core_connection_array, current_core_id = \
            get_relnet_intermediate_placement(relnet_inp_weights=relational_function_input_weights[i],
                                              relnet_inp_delays=relational_function_input_delays[i],
                                              input_relnet_core_connection_array=prev_relnet_core_connection_array,
                                              neurons_per_core=relnet_intermediate_neurons_per_core[i-1],
                                              max_n_sentences=MAX_N_SENTENCES,
                                              layer_name='RelnetLayer{}'.format(i),
                                              core_index_start=current_core_id)

        relnet_intermediate_core_connection_array.append(prev_relnet_core_connection_array)

    translation_layer_core_connection, current_core_id = \
        get_translation_layer_placement(translation_layer_inp_weights=translation_layer_input_weights,
                                        input_relnet_core_connection_array=prev_relnet_core_connection_array,
                                        neurons_per_core=translation_layer_neurons_per_core,
                                        max_n_sentences=MAX_N_SENTENCES,
                                        layer_name='TranslationLayer',
                                        core_index_start=current_core_id)

    final_MLP_core_connection = []
    prev_final_MLP_core_connection = translation_layer_core_connection
    for i in range(len(final_MLP_units)):
        prev_final_MLP_core_connection, current_core_id = \
            get_final_MLP_placement(final_MLP_inp_weights=final_MLP_input_weights[i],
                                    final_MLP_inp_delays=final_MLP_input_delays[i],
                                    input_core_connections=prev_final_MLP_core_connection,
                                    neurons_per_core=final_MLP_neurons_per_core[i],
                                    layer_name='FinalMLP{}'.format(i),
                                    core_index_start=current_core_id)
        final_MLP_core_connection.append(prev_final_MLP_core_connection)

    readout_core_connection, current_core_id = \
        get_final_MLP_placement(final_MLP_inp_weights=readout_weights,
                                final_MLP_inp_delays=np.zeros(readout_weights.shape, dtype=np.int64),
                                input_core_connections=prev_final_MLP_core_connection,
                                neurons_per_core=readout_neurons_per_core,
                                layer_name='Readout'.format(i),
                                core_index_start=current_core_id)

    """
    Asserts that need to be made:

    6.  D The final core id should be equal to the total cores calculated by the original functions

    7.  D The core ids in the entire network must be sequential from 0 to the final core id

    1.  That each core has all the constraints satisfied
        1.  The number of output axons
        2.  The number of input axons
        3.  The memory limitations

    2.  That each connections has a weight matrix that matches the dimensions of the presyn and postsyn neuron range

    3.  For each layer, each weight matrix has to equal the submatrix picked as according to the core neuron range

    4.  For each core tuple, the cores must be sequential

    5.  For each core tuple, the last end must equal the layer size
    """

    """

    Calculate the optimal placement using the functions from loihi_placement_param_search.py

    Using these parameters, calculate the actual cores and connections

    create dict of cores indexed by id
    create counters for input, output axons and synapses that will be filled as we iterate through the list of synapses
    perform the relevant asserts
    """

    assert current_core_id == (relnet_init_n_cores +
                               sentence_n_cores_total +
                               question_n_cores_total +
                               1 +  # input mask neurons
                               sum(relnet_intermediate_n_cores) +
                               translation_layer_n_cores +
                               sum(final_MLP_n_cores) +
                               readout_n_cores),  "There appears to be some error in the total number of cores created"

    core_array = np.ndarray(current_core_id, dtype=object)
    neuron_count = np.zeros(current_core_id, dtype=np.int64)
    output_axon_count = np.zeros(current_core_id, dtype=np.int64)
    input_axon_count = np.zeros(current_core_id, dtype=np.int64)
    synapse_count = np.zeros(current_core_id, dtype=np.int64)

    core_tuple_list = []
    layer_name_to_size_map = {
        'LSNNSentence_input': input_dim,
        'LSNNQuestion_input': input_dim,
        'LSNNSentence_recurrent': LSNN_sentence_size,
        'LSNNQuestion_recurrent': LSNN_question_size,
        'LSNNSentence_relay': LSNN_sentence_size,
        'LSNNQuestion_relay': LSNN_question_size,
        'LSNN_mask': MAX_N_SENTENCES,
        'TranslationLayer': relnet_units[-1],
        'Readout': readout_dim,
    }
    for i, x in enumerate(relnet_units):
        layer_name_to_size_map['RelnetLayer{}'.format(i)] = x
    for i, x in enumerate(final_MLP_units):
        layer_name_to_size_map['FinalMLP{}'.format(i)] = x

    layer_name_to_weight_matrices_map = {
        'LSNNSentence_input': (np.ones(stories_LSNN_input_weights.shape, dtype=np.int64),  # mask
                               stories_LSNN_input_weights,  # weights
                               stories_LSNN_input_delays),  # delays
        'LSNNSentence_recurrent': (np.ones(stories_LSNN_rec_weights.shape, dtype=np.int64),
                                   stories_LSNN_rec_weights,
                                   stories_LSNN_rec_delays),
        'LSNNSentence_relay': (np.eye(LSNN_sentence_size, dtype=np.int64),
                               5*np.eye(LSNN_sentence_size, dtype=np.float32),
                               np.zeros((LSNN_sentence_size, LSNN_sentence_size))),
        'LSNNQuestion_input': (np.ones(queries_LSNN_input_weights.shape, dtype=np.int64),
                               queries_LSNN_input_weights,
                               queries_LSNN_input_delays),
        'LSNNQuestion_recurrent': (np.ones(queries_LSNN_rec_weights.shape, dtype=np.int64),
                                   queries_LSNN_rec_weights,
                                   queries_LSNN_rec_delays),
        'LSNNQuestion_relay': (np.eye(LSNN_question_size, dtype=np.int64),
                               relay_weight*np.eye(LSNN_question_size, dtype=np.float32),  # FIXME: This hardcoded shit needs to go
                               np.zeros((LSNN_question_size, LSNN_question_size))),
        'RelnetLayer0_sentence1': (np.ones(relational_function_input_weights[0][:LSNN_sentence_size].shape, dtype=np.int64),
                                   relational_function_input_weights[0][:LSNN_sentence_size],
                                   relational_function_cell_data[0]['n_delay'] - 1 - relational_function_input_delays[0][:LSNN_sentence_size]),
        'RelnetLayer0_sentence2': (np.ones(relational_function_input_weights[0][LSNN_sentence_size:2*LSNN_sentence_size].shape, dtype=np.int64),
                                   relational_function_input_weights[0][LSNN_sentence_size:2*LSNN_sentence_size],
                                   relational_function_cell_data[0]['n_delay'] - 1 - relational_function_input_delays[0][LSNN_sentence_size:2*LSNN_sentence_size]),
        'RelnetLayer0_question': (np.ones(relational_function_input_weights[0][2*LSNN_sentence_size:].shape, dtype=np.int64),
                                  relational_function_input_weights[0][2*LSNN_sentence_size:],
                                  relational_function_cell_data[0]['n_delay'] - 1 - relational_function_input_delays[0][2*LSNN_sentence_size:]),
        # This case is handled specially as the connections connect from a different subset of the core each time
        # 'RelnetLayer0_mask': (np.ones(relational_function_input_weights[0][2*LSNN_sentence_size:].shape, dtype=np.int64),
        #                       relational_function_input_weights[0][2*LSNN_sentence_size:],
        #                       relational_function_cell_data[0]['n_delay'] - 1 - relational_function_input_delays[0][2*LSNN_sentence_size:]),
        'TranslationLayer': (np.eye(relnet_units[-1], dtype=np.int64),
                             np.diag(translation_layer_input_weights),
                             np.zeros((relnet_units[-1], relnet_units[-1]), dtype=np.int64)),
    }
    for i, (w, d) in enumerate(zip(relational_function_input_weights[1:], relational_function_input_delays[1:])):
        layer_name_to_weight_matrices_map['RelnetLayer{}'.format(i+1)] = \
            (np.ones(relational_function_input_weights[i+1].shape, dtype=np.int64),
             relational_function_input_weights[i+1],
             relational_function_input_delays[i+1])

    for i, (w, d) in enumerate(zip(final_MLP_input_weights, final_MLP_input_delays)):
        layer_name_to_weight_matrices_map['FinalMLP{}'.format(i)] = \
            (np.ones(final_MLP_input_weights[i].shape, dtype=np.int64),
             final_MLP_input_weights[i],
             final_MLP_input_delays[i])

    def add_cores_and_core_tuple(tuple_of_cores):
        assert len(tuple_of_cores) > 0

        core_id_arr = np.array([core.id for core in tuple_of_cores])
        assert len(tuple_of_cores) == 1 or np.all(core_id_arr[1:] - core_id_arr[:-1] == 1)  # assert sequential core id array
        assert tuple_of_cores[-1].end == layer_name_to_size_map[tuple_of_cores[0].layer_name]  # assert all neurons covered

        for core in tuple_of_cores:
            assert core_array[core.id] is None, "It Appears that a core id has been repeated"
            assert all(hasattr(core, x) for x in {'id', 'layer_name', 'start', 'end'}), \
                "The core passed is not a valid core tuple"
            core_array[core.id] = core
            neuron_count[core.id] = core.end - core.start

        core_tuple_list.append((tuple_of_cores, tuple_of_cores[0].layer_name))

    # lsnn_question_core_connection_array
    for core_conn_data in lsnn_sentence_core_connection_array.flatten():
        add_cores_and_core_tuple(core_conn_data.cores.input)
        add_cores_and_core_tuple(core_conn_data.cores.lsnn)
        for relay_core_tuple in core_conn_data.cores.relay:
            add_cores_and_core_tuple(relay_core_tuple)

    add_cores_and_core_tuple(lsnn_question_core_connection_array.cores.input)
    add_cores_and_core_tuple(lsnn_question_core_connection_array.cores.lsnn)
    for relay_core_tuple in lsnn_question_core_connection_array.cores.relay:
        add_cores_and_core_tuple(relay_core_tuple)

    # Add core of the masking neurons
    add_cores_and_core_tuple((input_mask_core,))

    # relnet_init_core_connection_array
    for core_conn_data in relnet_init_core_connection_array.flatten():
        if core_conn_data is not None:
            add_cores_and_core_tuple(core_conn_data.cores)

    for relnet_layer_array in relnet_intermediate_core_connection_array:
        for core_conn_data in relnet_layer_array.flatten():
            if core_conn_data is not None:
                add_cores_and_core_tuple(core_conn_data.cores)

    add_cores_and_core_tuple(translation_layer_core_connection.cores)

    for layer_core_connection in final_MLP_core_connection:
        add_cores_and_core_tuple(layer_core_connection.cores)

    add_cores_and_core_tuple(readout_core_connection.cores)

    assert all(x is not None for x in core_array), \
        "It appears there might be core ids missing"

    # Flatten all connections

    connection_tuple_list = []

    def add_connection_tuple(tuple_of_connections, layer_name):
        assert len(tuple_of_connections) > 0
        assert all(all(hasattr(connection, x) for x in {'presyn_core', 'postsyn_core', 'mask', 'weights', 'delays'})
                   for connection in tuple_of_connections), \
            "The connection_tuple passed is not a valid connection tuple"
        connection_tuple_list.append((tuple_of_connections, layer_name))
        for conn in tuple_of_connections:
            # Update the resource usage statistics for the relevant cores
            if conn.presyn_core != conn.postsyn_core:
                n_axons = np.count_nonzero(np.sum(conn.mask, axis=1))
                output_axon_count[conn.presyn_core] += n_axons
                input_axon_count[conn.postsyn_core] += n_axons
            synapse_count[conn.postsyn_core] += np.count_nonzero(conn.mask)

            # Assert compliance with the weight matrices
            presyn_core_slice = slice(core_array[conn.presyn_core].start, core_array[conn.presyn_core].end)
            postsyn_core_slice = slice(core_array[conn.postsyn_core].start, core_array[conn.postsyn_core].end)

            # Assert the weight and delay matrix compliance
            if layer_name == 'RelnetLayer0_mask':
                source_weight_matrix = np.zeros((MAX_N_SENTENCES, relnet_units[0]), dtype=np.float32)  # FIXME
                source_weight_matrix[np.array([conn.sentence1_ind, conn.sentence2_ind]), :] = input_mask_weight
                source_delay_matrix = np.zeros((MAX_N_SENTENCES, relnet_units[0]), dtype=np.int64)
            else:
                source_mask_matrix, source_weight_matrix, source_delay_matrix = layer_name_to_weight_matrices_map[layer_name]

            assert np.all(conn.weights == source_weight_matrix[presyn_core_slice, postsyn_core_slice])
            assert np.all(conn.delays == source_delay_matrix[presyn_core_slice, postsyn_core_slice])

    for core_conn_data in lsnn_sentence_core_connection_array.flatten():
        add_connection_tuple(core_conn_data.connections.input_to_lsnn, 'LSNNSentence_input')
        add_connection_tuple(core_conn_data.connections.lsnn_to_lsnn, 'LSNNSentence_recurrent')
        add_connection_tuple(core_conn_data.connections.lsnn_to_relay, 'LSNNSentence_relay')

    add_connection_tuple(lsnn_question_core_connection_array.connections.input_to_lsnn, 'LSNNQuestion_input')
    add_connection_tuple(lsnn_question_core_connection_array.connections.lsnn_to_lsnn, 'LSNNQuestion_recurrent')
    add_connection_tuple(lsnn_question_core_connection_array.connections.lsnn_to_relay, 'LSNNQuestion_relay')

    for core_conn_data in relnet_init_core_connection_array.flatten():
        if core_conn_data is not None:
            add_connection_tuple(core_conn_data.connections.sentence1_to_relnet, 'RelnetLayer0_sentence1')
            add_connection_tuple(core_conn_data.connections.sentence2_to_relnet, 'RelnetLayer0_sentence2')
            add_connection_tuple(core_conn_data.connections.question_to_relnet, 'RelnetLayer0_question')
            add_connection_tuple(core_conn_data.connections.mask_to_relnet, 'RelnetLayer0_mask')

    for i, relnet_layer_array in enumerate(relnet_intermediate_core_connection_array):
        for core_conn_data in relnet_layer_array.flatten():
            if core_conn_data is not None:
                add_connection_tuple(core_conn_data.connections, 'RelnetLayer{}'.format(i+1))

    for connection_tuple in translation_layer_core_connection.connections.flatten():
        if connection_tuple is not None:
            add_connection_tuple(connection_tuple, 'TranslationLayer')

    for i, layer_core_connection in enumerate(final_MLP_core_connection):
        add_connection_tuple(layer_core_connection.connections, 'FinalMLP{}'.format(i))

    ## here is where we get serious with the asserts
    input_core_ids_sentence = np.concatenate([np.array([y.id for y in x.cores.input]) for x in lsnn_sentence_core_connection_array.ravel()])
    input_core_ids_question = np.array([y.id for y in lsnn_question_core_connection_array.cores.input])
    input_core_ids = np.concatenate([input_core_ids_sentence, input_core_ids_question])

    if use_cores_for_input:
        actual_output_axon_limit = OUTPUT_AXON_LIMIT
    else:
        actual_output_axon_limit = OUTPUT_AXON_LIMIT*np.ones(current_core_id, dtype=np.int64)
        actual_output_axon_limit[input_core_ids] = 2**23

    fanout_count = output_axon_count//neuron_count
    assert np.all(neuron_count <= MAX_TOTAL_NEURONS_IN_CORE)
    assert np.all(output_axon_count <= actual_output_axon_limit)
    assert np.all(fanout_count <= OUTPUT_AXON_PER_NEURON_LIMIT)
    assert np.all(input_axon_count <= INPUT_AXON_LIMIT)

    n_lsnn_cores = sentence_n_cores_total + question_n_cores_total + 1  # +1 for the mask core
    n_relnet_cores = relnet_init_n_cores + sum(relnet_intermediate_n_cores)
    n_ff_neuron_layer_cores = sum(final_MLP_n_cores)

    print("sentence LSNN neuron count: ", neuron_count[:25])
    print("sentence LSNN output axon count: ", output_axon_count[:25])
    print("sentence LSNN fanout: ", fanout_count[:25])
    print("sentence LSNN input axon count: ", input_axon_count[:25])

    print("question LSNN neuron count: ", neuron_count[sentence_n_cores_total:][:25])
    print("question LSNN output axon count: ", output_axon_count[sentence_n_cores_total:][:25])
    print("question LSNN fanout: ", fanout_count[sentence_n_cores_total:][:25])
    print("question LSNN input axon count: ", input_axon_count[sentence_n_cores_total:][:25])

    print("relational net neuron count: ", neuron_count[n_lsnn_cores:][:25])
    print("relational net output axon count: ", output_axon_count[n_lsnn_cores:][:25])
    print("relational net fanout: ", fanout_count[n_lsnn_cores:][:25])
    print("relational net input axon count: ", input_axon_count[n_lsnn_cores:][:25])

    core_cursor = 0
    assert np.all(synapse_count[core_cursor:core_cursor+n_lsnn_cores] <= MAX_SYNAPSES_LSNN)
    core_cursor += n_lsnn_cores
    assert np.all(synapse_count[core_cursor:core_cursor+n_relnet_cores] <= MAX_SYNAPSES_FF)
    core_cursor += n_relnet_cores
    assert np.all(synapse_count[core_cursor:core_cursor+translation_layer_n_cores] <= MAX_SYNAPSES_TRANSLATION)
    core_cursor += translation_layer_n_cores
    assert np.all(synapse_count[core_cursor:] <= MAX_SYNAPSES_FF)

    relevant_relnet_core_id_list = []
    for sentence1 in range(MAX_N_SENTENCES):
        for sentence2 in range(sentence1, MAX_N_SENTENCES):
            current_relnet_copy = relnet_init_core_connection_array[sentence1, sentence2]
            current_relnet_cores = current_relnet_copy.cores
            relevant_relnet_core_id_list.extend([core.id for core in current_relnet_cores])
    relevant_relnet_core_id_list = np.array(relevant_relnet_core_id_list)
    total_relnet_incoming_synapses = np.sum(synapse_count[relevant_relnet_core_id_list])
    print("The total number of connections made to the first layer of the relational network is {}".format(total_relnet_incoming_synapses))

    return ((lsnn_sentence_core_connection_array, sentence_n_cores_total),
            (lsnn_question_core_connection_array, question_n_cores_total),
            (input_mask_core, 1),
            (relnet_init_core_connection_array, relnet_init_n_cores),
            (relnet_intermediate_core_connection_array, relnet_intermediate_n_cores),
            (translation_layer_core_connection, translation_layer_n_cores),
            (final_MLP_core_connection, final_MLP_n_cores),
            (readout_core_connection, readout_n_cores))


def main(data_file_path):

    with open(data_file_path, 'rb') as fin:
        data_dict = pickle.load(fin)
        plot_data = data_dict['plot_data']
        cell_data = data_dict['cell_data']

    stories_LSNN_cell_data = cell_data['stories_LSNN_cell_data']
    queries_LSNN_cell_data = cell_data['queries_LSNN_cell_data']
    relational_function_cell_data = cell_data['relational_function_cell_data']
    quantization_cell_data = cell_data['quantization_cell_data']
    final_MLP_cell_data = cell_data['final_MLP_cell_data']
    readout_cell_data = cell_data['readout_cell_data']

    # Assert all the voltages when multiplied by the voltage scale factor are approximately integral
    stories_LSNN_voltages = plot_data['embedded_input_stories_V']
    queries_LSNN_voltages = plot_data['embedded_input_queries_V']
    relational_function_voltages = plot_data['relational_function_voltages']
    translation_layer_voltages = plot_data['quantization_voltages']
    final_MLP_voltages = plot_data['final_MLP_voltages']
    readout_voltages = plot_data['final_output']

    stories_LSNN_voltage_scaling = stories_LSNN_cell_data['voltage_scaling_factor']
    queries_LSNN_voltage_scaling = queries_LSNN_cell_data['voltage_scaling_factor']
    relational_function_voltage_scaling = tuple(x['voltage_scaling_factor'] for x in relational_function_cell_data)
    final_MLP_voltage_scaling = tuple(x['voltage_scaling_factor'] for x in final_MLP_cell_data)
    readout_voltage_scaling = readout_cell_data['voltage_scaling_factor']
    translation_layer_voltage_scaling = quantization_cell_data['voltage_scaling_factor']

    # def is_close_to_integral(ndarray):
    #     return np.all(np.abs(ndarray - np.round(ndarray)) < 1)

    # assert is_close_to_integral(stories_LSNN_voltages*stories_LSNN_voltage_scaling)
    # assert is_close_to_integral(queries_LSNN_voltages*queries_LSNN_voltage_scaling)
    # assert is_close_to_integral(translation_layer_voltages*translation_layer_voltage_scaling)
    # assert all(is_close_to_integral(x*scale) for x, scale in zip(relational_function_voltages, relational_function_voltage_scaling))
    # assert all(is_close_to_integral(x*scale) for x, scale in zip(final_MLP_voltages, final_MLP_voltage_scaling))
    # assert is_close_to_integral(readout_voltages*readout_voltage_scaling)

    # assert np.all(np.abs(np.round(readout_voltages*readout_voltage_scaling)) < 2**24)

    ((lsnn_sentence_core_connection_array, sentence_n_cores_total),
     (lsnn_question_core_connection_array, question_n_cores_total),
     (input_mask_core, _),
     (relnet_init_core_connection_array, relnet_init_n_cores),
     (relnet_intermediate_core_connection_array, relnet_intermediate_n_cores),
     (translation_layer_core_connection, translation_layer_n_cores),
     (final_MLP_core_connection, final_MLP_n_cores),
     (readout_core_connection, readout_n_cores)) = get_relational_network_placement_on_loihi(cell_data)


if __name__ == '__main__':
    import ipdb

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', action='store', type=str, required=True)
    arg_namespace = parser.parse_args()

    ipdb.set_trace()
    with ipdb.launch_ipdb_on_exception():
        main(arg_namespace.data_file)
