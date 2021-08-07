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

#include "readout_snip_mgmt.h"
#include <stdlib.h>

ADDRESS_MAP outputNeuronMap[NUM_OUTPUTS];
static int classifications[500];
static int current_iteration = -1;
static int numNeuronsPerCore = 1<<10;
static int numDendAccumPerCore = 1<<13;
static CoreId coreId;
static NeuronCore *nc;
static int readout_core_ids[128];
char readout_channel_name[20];
CxProfileCfg tmp_cx_profile_cfg;
SharedCfg tmp_shared_cfg;
static CoreId coreids [128];
static NeuronCore *ncs[128];

int do_mgmt(runState *rs) {

    if (((int)(rs->time_step) - 2) % (TIMESTEPS_PER_SAMPLE) == 0) {
        // This offset of 2 is because of the offset between the input and the LSNN (plus allowing relay neurons to be disabled at the beginning)
        current_iteration++;
    }
    
    if (rs->time_step == 1) {
        return 1;
    }
    
    if (rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_START_READOUT) {
        return 1;
    }
    
    if (rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_RESET_OFFSET) {
        return 1;
    }
    
    return 0;
}

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}

/* Function to clear the values of the membrane voltage */
void ClearVoltage() {
    nx_fast_init32(&nc->cx_state, numNeuronsPerCore, 0);
}

/*Function to change voltage decay*/
void ChangeVoltageDecay(int decay, int offset) {
    CxProfileCfg tmp_cx_profile_cfg = nc->cx_profile_cfg[0];
    tmp_cx_profile_cfg.Decay_v = decay;
    nc->cx_profile_cfg[0] = tmp_cx_profile_cfg;
    
    // need to also set offset, otherwise decay might be 1
    SharedCfg tmp_shared_cfg = nc->dendrite_shared_cfg;
    tmp_shared_cfg.DmOffsets = offset;
    nc->dendrite_shared_cfg = tmp_shared_cfg;
}

uint16_t myChipId() {
  static bool valid=0;
  ChipId myChipId = nx_my_chipid();
  if (!valid) {
    valid=1;
    ChipId min=nx_min_chipid();
    ChipId max=nx_max_chipid();
    uint16_t m=0;
    uint16_t x, y, z;
    for (x=min.x; x<=max.x; x++) {
      for (y=min.y; y<=max.y; y++) {
        for (z=min.z; z<=max.z; z++) {
          if (nx_chipid(x,y,z).id == myChipId.id) {
            return m;
          }
          m++;
        }
      }
    }
  }
  printf("ERROR: Illegal chip id.");
  exit(1);
}

void run_mgmt(runState *rs) {

    //printf("######## Timestep readout: %d \n", rs->time_step);
    
    if (rs->time_step == 1) {
        
        int chip_id = myChipId();
        sprintf(readout_channel_name,"readout_id_ch_%d", chip_id);
        
        // Retrieve lsnn_core_ids
        readChannel(getChannelID(readout_channel_name), readout_core_ids, 1);
        
        logicalToPhysicalCoreId(23, &coreId);
        nc = NEURON_PTR(coreId);

        
        for (int i=0;i<128;i++) {
            //printf("%d, ", readout_core_ids[i]);

            int logical_id = readout_core_ids[i];
            if (logical_id != -1) {

                //printf("######## Setting readout to max decay at: %d \n", rs->time_step);

                logicalToPhysicalCoreId(logical_id, &coreId);
                nc = NEURON_PTR(coreId);
                ChangeVoltageDecay(4095, 1);
                
                // Retrieve output neuron map
                int outputNeuronChannelId = getChannelID("output_neurons");

                for (int i = 0; i < NUM_OUTPUTS; i++) {
                    readChannel(outputNeuronChannelId, &outputNeuronMap[i].boardId, 1);
                    readChannel(outputNeuronChannelId, &outputNeuronMap[i].chipId, 1);
                    readChannel(outputNeuronChannelId, &outputNeuronMap[i].coreId, 1);
                    readChannel(outputNeuronChannelId, &outputNeuronMap[i].regId, 1);
                }
            }
        }
        //printf("\n");
        
        for (int i=0; i<128; ++i) {
            coreids[i] = nx_nth_coreid(i);
            ncs[i] = NEURON_PTR(coreids[i]);
        }
    }
    
    if (rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_START_READOUT) {
        
        for (int i=0;i<128;i++) {
            int logical_id = readout_core_ids[i];
            if (logical_id != -1) {
                //printf("######## Starting readout integration at: %d \n", rs->time_step);
                logicalToPhysicalCoreId(logical_id, &coreId);
                nc = NEURON_PTR(coreId);
                ChangeVoltageDecay(0, 0);
            }
        }
    }
    
    
    // write classification and reset readout neurons
    if (current_iteration >= 0 && rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_RESET_OFFSET) {
        
        int v;
        int maxV = -8388608;
        int maxVId = -1;
        
        //printf("######## Starting readout write at: %d \n", rs->time_step);
        int initlogicalCoreId = outputNeuronMap[0].coreId;
        logicalToPhysicalCoreId(initlogicalCoreId, &coreId);
        nc = NEURON_PTR(coreId);
        
        for(int i=0; i<NUM_OUTPUTS; i++) {
            //printf("######## looping at: %d \n", i);
            // Find physical address of output neuron
            int logicalCoreId = outputNeuronMap[i].coreId;
            if (initlogicalCoreId != logicalCoreId) 
                logicalToPhysicalCoreId(logicalCoreId, &coreId);
                nc = NEURON_PTR(coreId);

            // Read  membrane potential and  store neuron id if strongest
            v = nc->cx_state[outputNeuronMap[i].regId].V;
            if (v > maxV) {
                maxVId = i;
                maxV = v;
            }
        }
        //printf("######## Starting write at: %d \n", rs->time_step);
        //printf("######## maxVId: %d with maxV: %d \n", maxVId, maxV);
        classifications[current_iteration] = maxVId;
        
        if (current_iteration + 1 == 250) {
            // Send classifications back to super host for all images in batch
            writeChannel(getChannelID("classifications"), classifications, 1);
        }
        
        // Reset cxState, cxMetaState and dendAccum registers at end of each image
        DendriteAccumEntry accum = (DendriteAccumEntry) {.accum=0};
        
        CxState state = (CxState) {.U=0, .V=0};
        
        // Re-initialize all 4 PHASE fields to IDLE (2)
        MetaState cxms = (MetaState) {.Phase0=2, .SomaOp0=2, 
                                      .Phase1=2, .SomaOp1=2,
                                      .Phase2=2, .SomaOp2=2,
                                      .Phase3=2, .SomaOp3=2};
                                      
        nx_fast_init_multicore(ncs[0]->cx_state, 2*numNeuronsPerCore, sizeof(uint32_t), sizeof(uint32_t), &state, coreids, 24);
        nx_fast_init_multicore(ncs[0]->cx_meta_state, (numNeuronsPerCore + 3) / 4, sizeof(uint32_t), sizeof(uint32_t), &cxms, coreids,24);
        nx_fast_init_multicore(ncs[0]->dendrite_accum, numDendAccumPerCore, sizeof(uint32_t), sizeof(uint32_t), &accum, coreids, 24);
        
        ChangeVoltageDecay(4095, 1);
        
    }
}