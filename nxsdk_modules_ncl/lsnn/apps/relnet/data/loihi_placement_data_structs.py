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

from collections import namedtuple

# =================================================================================
# Define the Various structs that contain data regarding LSNN cores and connections
# =================================================================================
LSNNCoreTuple = namedtuple('LSNNCoreTuple', ['id', 'layer_name', 'start', 'end', 'sentence_ind', 'fanout_copy_ind'])
LSNNCoreTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a core

    :param id: This is the id of the core. All cores in a particular network
        should have unique id's ranging from core_index_start to ...
    :param layer_name: This is the layer whose neurons are implemented by this
        core
    :param start: This is the starting index of the range of neurons that are
        simulated by this core
    :param end: This is the ending index of the range of neurons that are
        simulated by this core

    :param sentence_ind: This is extra information containing the sentence
        index to which the current core belongs
    :param fanout_copy_ind: This is extra information containing the fanout
        copy index to which the current core belongs
"""

LSNNConnectionTuple = namedtuple('LSNNConnectionTuple', ['presyn_core', 'postsyn_core', 'mask', 'weights', 'delays', 'weight_exp', 'sentence_ind', 'fanout_copy_ind'])
LSNNConnectionTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a connection between two cores

    :param presyn_core: This is the id of the core from which the synapses connect.
    :param postsyn_core: This is the id of the core to which the synapses connect.
    :param mask: This is the binary connection mask n_presyn x n_postsyn
    :param weights: This is the matrix of weights of size n_presyn x n_postsyn
    :param delays: This is the matrix of delays of size n_presyn x n_postsyn

    :param sentence_ind: This is extra information containing the sentence
        index to which the current core belongs
    :param fanout_copy_ind: This is extra information containing the fanout
        copy index to which the current core belongs
"""

LSNNCoresStruct = namedtuple('LSNNCoresStruct', ['input', 'lsnn', 'relay'])
LSNNCoresStruct.__doc__ = """
    This is a struct that contains the input cores and lsnn cores for a particular copy of the LSNN

    :param input: Tuple of input cores (each of which is an instance of CoreTuple)
    :param lsnn: Tuple of lsnn cores (each of which is an instance of CoreTuple)
    :param relay: Tuple of cores used for the relay network (each of which is an instance of CoreTuple)
"""

LSNNConnectionsStruct = namedtuple('LSNNConnectionsStruct', ['input_to_lsnn', 'lsnn_to_lsnn', 'lsnn_to_relay'])
LSNNConnectionsStruct.__doc__ = """
    This is a struct that contains the connections pertaining to a particular copy of the lsnn.

    :param input_to_lsnn: Tuple of connections from the input cores to the lsnn cores (each of which is an instance of ConnectionTuple)
    :param lsnn_to_lsnn: Tuple of recurrent connections within the lsnn cores (each of which is an instance of ConnectionTuple)
    :param lsnn_to_relay: Tuple of connections from LSNN to relay (each of which is an instance of ConnectionTuple)
"""

LSNNCopyStruct = namedtuple('LSNNCopyStruct', ['cores', 'connections'])
LSNNCopyStruct.__doc__ = """
    This is a struct that contains all the information pertaining to a particular
    copy of the LSNN

    :param cores: an instance of LSNNCoresStruct detailing all the input and
        lsnn cores for the particular copy of the LSNN
    :param connections: an instance of LSNNConnectionsStruct detailing all the
        input and lsnn connections for the particular copy of the LSNN
"""

# ===========================================================================================
# Define the Various structs that contain data regarding relnet initial cores and connections
# ===========================================================================================
RelnetInitCoreTuple = namedtuple('RelnetInitCoreTuple', ['id', 'layer_name', 'start', 'end', 'sentence1_ind', 'sentence2_ind'])
RelnetInitCoreTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a core

    :param id: This is the id of the core. All cores in a particular network
        should have unique id's ranging from core_index_start to ...
    :param layer_name: This is the layer whose neurons are implemented by this
        core
    :param start: This is the starting index of the range of neurons that are
        simulated by this core
    :param end: This is the ending index of the range of neurons that are
        simulated by this core

    :param sentence1_ind: This is extra information containing the sentence
        index of sentence1 used as input to this relational network copy
        simulated by the current core
    :param sentence2_ind: This is extra information containing the sentence
        index of sentence2 used as input to this relational network copy
        simulated by the current core
"""

RelnetInitConnectionTuple = namedtuple('RelnetInitConnectionTuple', ['presyn_core', 'postsyn_core', 'mask', 'weights', 'delays', 'weight_exp', 'sentence1_ind', 'sentence2_ind'])
RelnetInitConnectionTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a connection between two cores

    :param presyn_core: This is the id of the core from which the synapses connect.
    :param postsyn_core: This is the id of the core to which the synapses connect.
    :param mask: This is the binary connection mask n_presyn x n_postsyn
    :param weights: This is the matrix of weights of size n_presyn x n_postsyn
    :param delays: This is the matrix of delays of size n_presyn x n_postsyn

    :param sentence1_ind: This is extra information containing the sentence
        index of sentence1 used as input to this relational network copy
        simulated by the current core
    :param sentence2_ind: This is extra information containing the sentence
        index of sentence2 used as input to this relational network copy
        simulated by the current core
"""

RelnetInitConnectionsStruct = namedtuple('RelnetInitConnectionsStruct', ['sentence1_to_relnet', 'sentence2_to_relnet', 'mask_to_relnet', 'question_to_relnet'])
RelnetInitConnectionsStruct.__doc__ = """
    This is a struct that contains the connections pertaining to a particular copy of the lsnn.

    :param input_to_lsnn: Tuple of input cores (each of which is an instance of ConnectionTuple)
    :param lsnn_to_lsnn: Tuple of lsnn cores (each of which is an instance of ConnectionTuple)
"""

RelnetInitInstanceStruct = namedtuple('RelnetInitInstanceStruct', ['cores', 'connections'])
RelnetInitInstanceStruct.__doc__ = """
    This is a struct that contains all the information pertaining to a particular
    instance of the relational network connectivity

    :param cores: a tuple of RelnetInitCoreTuple detailing the cores that
        simulate the relnet instance
    :param connections: an instance of RelnetInitConnectionsStruct detailing
        all the connections for the particular instance of the relnet
"""

# ================================================================================================
# Define the Various structs that contain data regarding relnet intermediate cores and connections
# ================================================================================================
RelnetIntermediateCoreTuple = namedtuple('RelnetIntermediateCoreTuple', ['id', 'layer_name', 'start', 'end', 'sentence1_ind', 'sentence2_ind'])
RelnetIntermediateCoreTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a core

    :param id: This is the id of the core. All cores in a particular network
        should have unique id's ranging from core_index_start to ...
    :param layer_name: This is the layer whose neurons are implemented by this
        core
    :param start: This is the starting index of the range of neurons that are
        simulated by this core
    :param end: This is the ending index of the range of neurons that are
        simulated by this core

    :param sentence1_ind: This is extra information containing the sentence
        index of sentence1 used as input to this relational network copy
        simulated by the current core
    :param sentence2_ind: This is extra information containing the sentence
        index of sentence2 used as input to this relational network copy
        simulated by the current core
"""

RelnetIntermediateConnectionTuple = namedtuple('RelnetIntermediateConnectionTuple', ['presyn_core', 'postsyn_core', 'mask', 'weights', 'delays', 'weight_exp', 'sentence1_ind', 'sentence2_ind'])
RelnetIntermediateConnectionTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a connection between two cores

    :param presyn_core: This is the id of the core from which the synapses connect.
    :param postsyn_core: This is the id of the core to which the synapses connect.
    :param mask: This is the binary connection mask n_presyn x n_postsyn
    :param weights: This is the matrix of weights of size n_presyn x n_postsyn
    :param delays: This is the matrix of delays of size n_presyn x n_postsyn

    :param sentence1_ind: This is extra information containing the sentence
        index of sentence1 used as input to this relational network copy
        simulated by the current core
    :param sentence2_ind: This is extra information containing the sentence
        index of sentence2 used as input to this relational network copy
        simulated by the current core
"""

RelnetIntermediateInstanceStruct = namedtuple('RelnetIntermediateInstanceStruct', ['cores', 'connections'])
RelnetIntermediateInstanceStruct.__doc__ = """
    This is a struct that contains all the information pertaining to a particular
    instance of the relational network connectivity

    :param cores: a tuple of RelnetIntermediateCoreTuple detailing the cores that
        simulate the relnet instance
    :param connections: a tuple of RelnetIntermediateConnectionTuple detailing
        all the connections for the particular instance of the relnet
"""

# ==================================================================================================
# Define the Various structs that contain data regarding the translation layer cores and connections
# ==================================================================================================
TranslationLayerCoreTuple = namedtuple('TranslationLayerCoreTuple', ['id', 'layer_name', 'start', 'end'])
TranslationLayerCoreTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a core

    :param id: This is the id of the core. All cores in a particular network
        should have unique id's ranging from core_index_start to ...
    :param layer_name: This is the layer whose neurons are implemented by this
        core
    :param start: This is the starting index of the range of neurons that are
        simulated by this core
    :param end: This is the ending index of the range of neurons that are
        simulated by this core
"""

TranslationLayerConnectionTuple = namedtuple('TranslationLayerConnectionTuple', ['presyn_core', 'postsyn_core', 'mask', 'weights', 'delays', 'weight_exp', 'sentence1_ind', 'sentence2_ind'])
TranslationLayerConnectionTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a connection between two cores

    :param presyn_core: This is the id of the core from which the synapses connect.
    :param postsyn_core: This is the id of the core to which the synapses connect.
    :param mask: This is the binary connection mask n_presyn x n_postsyn
    :param weights: This is the matrix of weights of size n_presyn x n_postsyn
    :param delays: This is the matrix of delays of size n_presyn x n_postsyn

    :param sentence1_ind: This is extra information containing the sentence
        index of sentence1 used as input to the relational network whose output
        neuron forms the presynaptic neuron of the current synapse.
    :param sentence2_ind: This is extra information containing the sentence
        index of sentence2 used as input to the relational network whose output
        neuron forms the presynaptic neuron of the current synapse.
"""

TranslationLayerInstanceStruct = namedtuple('TranslationLayerInstanceStruct', ['cores', 'connections'])
TranslationLayerInstanceStruct.__doc__ = """
    This is a struct that contains all the information pertaining to a particular
    instance of the relational network connectivity

    :param cores: a tuple of TranslationLayerCoreTuple detailing the cores that
        simulate the translation layer
    :param connections: an np.ndarray of size max_n_sentences x max_n_sentences.
        connections[sentence1_ind, sentence2_ind] contains a tuple of
        TranslationLayerConnectionTuple instances which represent the connections
        from that relational network to the translation layer.
"""

# ============================================================================================
# Define the Various structs that contain data regarding final mlp layer cores and connections
# ============================================================================================
FinalMLPCoreTuple = namedtuple('FinalMLPCoreTuple', ['id', 'layer_name', 'start', 'end'])
FinalMLPCoreTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a core

    :param id: This is the id of the core. All cores in a particular network
        should have unique id's ranging from core_index_start to ...
    :param layer_name: This is the layer whose neurons are implemented by this
        core
    :param start: This is the starting index of the range of neurons that are
        simulated by this core
    :param end: This is the ending index of the range of neurons that are
        simulated by this core
"""

FinalMLPConnectionTuple = namedtuple('FinalMLPConnectionTuple', ['presyn_core', 'postsyn_core', 'mask', 'weights', 'delays', 'weight_exp'])
FinalMLPConnectionTuple.__doc__ = """
    Tuple that represents all the necessary information pertaining to a connection between two cores

    :param presyn_core: This is the id of the core from which the synapses connect.
    :param postsyn_core: This is the id of the core to which the synapses connect.
    :param mask: This is the binary connection mask n_presyn x n_postsyn
    :param weights: This is the matrix of weights of size n_presyn x n_postsyn
    :param delays: This is the matrix of delays of size n_presyn x n_postsyn
"""

FinalMLPInstanceStruct = namedtuple('FinalMLPInstanceStruct', ['cores', 'connections'])
FinalMLPInstanceStruct.__doc__ = """
    This is a struct that contains all the information pertaining to a particular
    instance of the relational network connectivity

    :param cores: a tuple of FinalMLPCoreTuple detailing the cores that
        simulate the relnet instance
    :param connections: a tuple of FinalMLPConnectionTuple instances
        which represent the connections from the previous layer to this one
"""
