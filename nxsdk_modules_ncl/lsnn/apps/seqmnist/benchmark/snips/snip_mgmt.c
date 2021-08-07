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

#include "snip_mgmt.h"

extern ADDRESS_MAP outputNeuronMap[NUM_OUTPUTS];
extern int cores;
static int classifications[BATCH_SIZE];
static int numNeuronsPerCore = 1<<10;
static int numDendAccumPerCore = 1<<13;


int do_mgmt(runState *rs) {
    if (rs->time_step % (TIMESTEPS_PER_IMAGE) == 0) {
        return 1;
    } else {
        return 0;
    }
}

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}

void snip_mgmt(runState *rs) {
    int logicalCoreId;
    CoreId coreId;
    NeuronCore *nc;

    // Compute time and image id within batch
    int tBatch = ((rs->time_step-1) % TIMESTEPS_PER_BATCH + 1);
    int imgId = (tBatch-1)/TIMESTEPS_PER_IMAGE;

    // Read outputs and classify
    int v;
    int maxV = -8388608;
    int maxVId = maxV;
    for(int i=0; i<NUM_OUTPUTS; i++) {
        // Find physical address of output neuron
        logicalCoreId = 0; //outputNeuronMap[i].coreId;
        logicalToPhysicalCoreId(logicalCoreId, &coreId);
        nc = NEURON_PTR(coreId);

        // Read  membrane potential and  store neuron id if strongest
        v = nc->cx_state[outputNeuronMap[i].regId].V;
        if (v > maxV) {
            maxVId = i;
            maxV = v;
        }

     }
    classifications[imgId] = maxVId;

    // Send classifications back to super host for all images in batch
    if (tBatch == TIMESTEPS_PER_BATCH) {
        writeChannel(getChannelID("classifications"), classifications, BATCH_SIZE);
    }

    // Reset cxState, cxMetaState and dendAccum registers at end of each image
    // Re-initialize all 4 PHASE fields to IDLE (2)
    int cxms = (((((2 << 5) | 2) << 5) | 2) << 5) | 2;

    for(int i=0; i<cores; i++) {
        // Find physical address of logical cores
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);

        // Reset neuro cores
        nx_fast_init32(&nc->cx_state, 2*numNeuronsPerCore, 0);
        nx_fast_init32(&nc->cx_meta_state, numNeuronsPerCore/4, cxms);
        nx_fast_init32(&nc->dendrite_accum, numDendAccumPerCore, 0);
    }
}