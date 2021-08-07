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

#include "snip_reset.h"
#include <stdlib.h>

static int current_iteration = -1;
static int numNeuronsPerCore = 1<<10;
static int numDendAccumPerCore = 1<<13;
static NeuronCore *nc;
static int reset_core_ids[128];
char reset_channel_name[20];
CoreId coreids [128];
NeuronCore *ncs[128];
static int chip_id;

int do_mgmt(runState *rs) {

    if (((int)(rs->time_step) - 2) % (TIMESTEPS_PER_SAMPLE) == 0) {
        // This offset of 2 is because of the offset between the input and the LSNN (plus allowing relay neurons to be disabled at the beginning)
        current_iteration++;
    } 
    
    if (rs->time_step == 1) {
        return 1;
    }
    
    if (rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_RESET_OFFSET) {
        return 1;
    }
    
    return 0;
}

/* Function to enable updates. Used to enable spiking.*/
void EnableUpdates() {
    nc->num_updates = (UpdateCfg) {
        .num_updates = (1 + 3) / 4,
        .num_stdp    = 0
    };
}

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
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

    //printf("######## RESET Timestep: %d \n", rs->time_step);
    
    if (rs->time_step == 1) {
        
        chip_id = myChipId();
        sprintf(reset_channel_name,"reset_id_ch_%d", chip_id);
        
        // Retrieve reset_core_ids
        readChannel(getChannelID(reset_channel_name), reset_core_ids, 1);
        
        /*
        for (int i=0;i<128;i++) {
               printf("%d, ", reset_core_ids[i]);
        }
        printf("\n");
        */
        for (int i=0; i<128; ++i) {
            coreids[i] = nx_nth_coreid(i);
            ncs[i] = NEURON_PTR(coreids[i]);
        }
    }

    
    // reset all cores which are not LSNN/Relay or readout
    if (current_iteration >= 0 && rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_RESET_OFFSET) {
        
        /*
        for (int i=0;i<128;i++) {
            
            int logical_id = reset_core_ids[i];
            if (logical_id != -1) {

                logicalToPhysicalCoreId(logical_id, &coreId);
                nc = NEURON_PTR(coreId);

                // Reset cxState, cxMetaState and dendAccum registers at end of each image

                // Re-initialize all 4 PHASE fields to IDLE (2)
                int cxms = (((((2 << 5) | 2) << 5) | 2) << 5) | 2;

                // Reset neuro cores
                nx_fast_init32(&nc->cx_state, 2*numNeuronsPerCore, 0);
                nx_fast_init32(&nc->cx_meta_state, (numNeuronsPerCore + 3) / 4, cxms);
                nx_fast_init32(&nc->dendrite_accum, numDendAccumPerCore, 0);
            }
        }*/
        
        //printf("######## RESET Timestep: %d \n", rs->time_step);
        
        DendriteAccumEntry accum = (DendriteAccumEntry) {.accum=0};
        
        CxState state = (CxState) {.U=0, .V=0};
        
        // Re-initialize all 4 PHASE fields to IDLE (2)
        MetaState cxms = (MetaState) {.Phase0=2, .SomaOp0=2, 
                                      .Phase1=2, .SomaOp1=2,
                                      .Phase2=2, .SomaOp2=2,
                                      .Phase3=2, .SomaOp3=2};
        
        //UpdateCfg upt = (UpdateCfg) {.num_updates = 0, .num_stdp    = 0};

        
        nx_fast_init_multicore(ncs[0]->cx_state, 2*numNeuronsPerCore, sizeof(uint32_t), sizeof(uint32_t), &state, coreids, 128);
        nx_fast_init_multicore(ncs[0]->cx_meta_state, (numNeuronsPerCore + 3) / 4, sizeof(uint32_t), sizeof(uint32_t), &cxms, coreids,128);
        nx_fast_init_multicore(ncs[0]->dendrite_accum, numDendAccumPerCore, sizeof(uint32_t), sizeof(uint32_t), &accum, coreids, 128);
        
        if (chip_id != 20 && chip_id != 0 && chip_id != 0) {
            //nx_fast_init_multicore(ncs[0]->num_updates, 1, sizeof(uint32_t), sizeof(uint32_t), &upt, coreids, 128);
        }
    }
}