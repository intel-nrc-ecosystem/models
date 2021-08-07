/*
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
*/
#include <stdlib.h>
#include "initsnip.h"


void init_snip(runState *s) {
    int paramChannelID = getChannelID("nxinit_input_params");

    // Retrieve size parameters
    readChannel(paramChannelID, &num_input, 1);
    readChannel(paramChannelID, &img_size, 1);
    readChannel(paramChannelID, &batch_size, 1);
    readChannel(paramChannelID, &cores, 1);

    if (batch_size != BATCH_SIZE) {
        printf("\nSeqMnist: ERROR - Incompatible batchSize received.\n");
        readChannel(paramChannelID, &batch_size, 1); // Make execution stall!
    }
    if (num_input != NUM_INPUTS) {
        printf("\nSeqMnist: ERROR - numInput received.\n");
        readChannel(paramChannelID, &num_input, 1); // Make execution stall!
    }

    // Retrieve thresholds for spike generator
    readChannel(getChannelID("nxinit_thresholds"), thresholds, NUM_THRESHOLDS);
    thresholds[0] = 1;

    // Retrieve input port map
    int inputPortChannelID = getChannelID("nxinit_input_ports");

    for (int i = 0; i < num_input; i++) {
        readChannel(inputPortChannelID, &inputPortMap[i].boardId, 1);
        readChannel(inputPortChannelID, &inputPortMap[i].chipId, 1);
        readChannel(inputPortChannelID, &inputPortMap[i].coreId, 1);
        readChannel(inputPortChannelID, &inputPortMap[i].regId, 1);

    }

    // Retrieve output neuron map
    int outputNeuronChannelId = getChannelID("output_neurons");

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        readChannel(outputNeuronChannelId, &outputNeuronMap[i].boardId, 1);
        readChannel(outputNeuronChannelId, &outputNeuronMap[i].chipId, 1);
        readChannel(outputNeuronChannelId, &outputNeuronMap[i].coreId, 1);
        readChannel(outputNeuronChannelId, &outputNeuronMap[i].regId, 1);

    }
}
