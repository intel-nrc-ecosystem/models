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
#include <stdlib.h>

static int current_iteration = -1;
static int numNeuronsPerCore = 1<<10;
static int numDendAccumPerCore = 1<<13;
static CoreId coreId;
static NeuronCore *nc;
static int lsnn_core_ids[128];
static int relay_core_ids[128];
char lsnn_channel_name[20];
char relay_channel_name[20];
CoreId coreids [128];
NeuronCore *ncs[128];
static int lsnn = 0;
static int relay = 0;
static int chip_id;

int do_mgmt(runState *rs) {
    
    if (rs->time_step == 1) {
        return 1;
    }

    if (((int)(rs->time_step) - 2) % (TIMESTEPS_PER_SAMPLE) == 0) {
        // This offset of 2 is because of the offset between the input and the LSNN (plus allowing relay neurons to be disabled at the beginning)
        current_iteration++;
    } 
    
    if ((rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration)) == TIMESTEP_START_RELAY) {
        return 1;
    } 
    
    if (rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_STOP_LSNN) {
        return 1;
    }
    
    // if ((rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration)) == TIMESTEP_STOP_LSNN) {
    //     return 1;
    // }
    
    return 0;
}

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}

/* Function to disable updates. Used to disable spiking.*/
void DisableUpdates() {
    nc->num_updates = (UpdateCfg) {
        .num_updates = 0,
        .num_stdp    = 0
    };
}

/* Function to enable updates. Used to enable spiking.*/
void EnableUpdates() {
    nc->num_updates = (UpdateCfg) {
        .num_updates = (numNeuronsPerCore + 3) / 4,
        .num_stdp    = 0
    };
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
    
    //printf("######## Timestep mgmt: %d \n", rs->time_step);
    
    if (rs->time_step == 1) {
        
        chip_id = myChipId();
        sprintf(lsnn_channel_name,"lsnn_id_ch_%d", chip_id);
        sprintf(relay_channel_name,"relay_id_ch_%d", chip_id);
        
        // Retrieve lsnn_core_ids
        readChannel(getChannelID(lsnn_channel_name), lsnn_core_ids, 1);
        
        // Retrieve relay_core_ids
        readChannel(getChannelID(relay_channel_name), relay_core_ids, 1);
       
        /*
        for (int i=0;i<128;i++) {
               printf("%d, ", lsnn_core_ids[i]);
        }
        printf("\n");
        
        for (int i=0;i<128;i++) {
               printf("%d, ", relay_core_ids[i]);
        }
        //printf("\n");
        */
        
        
        for (int i=0; i<128; ++i) {
            coreids[i] = nx_nth_coreid(i);
            ncs[i] = NEURON_PTR(coreids[i]);
            
            if (lsnn_core_ids[i] != -1) {
                lsnn = 1;
            }
            if (relay_core_ids[i] != -1) {
                relay = 1;
            }
        }

        //disable the relay neurons to not send spikes into the FF neurons
        //printf("#### Disabling started at %d\n", rs->time_step);
        for (int i=0;i<128;i++) {
            int logical_id = relay_core_ids[i];
            if (logical_id != -1) {
                // Find physical address of logical cores
                logicalToPhysicalCoreId(logical_id, &coreId);
                nc = NEURON_PTR(coreId);
                DisableUpdates();
                //printf("Disabled %d\n", logical_id);
            }
        }
    }
    
    //first and every 400ths time step
    if (current_iteration >= 0 && rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration) == TIMESTEP_STOP_LSNN) {
        //disable the relay neurons to not send spikes into the FF neurons
        //printf("#### Disabling started at %d\n", rs->time_step);
        for (int i=0;i<128;i++) {
            int logical_id = relay_core_ids[i];
            if (logical_id != -1) {
                // Find physical address of logical cores
                logicalToPhysicalCoreId(logical_id, &coreId);
                nc = NEURON_PTR(coreId);
                DisableUpdates();
                //printf("Disabled %d\n", logical_id);
            }
        }
        
        // Stop and reset LSNN blocks
        //printf("####### Reset LSNN at %d\n", rs->time_step);
        // Reset cxState, cxMetaState and dendAccum registers of the LSNN at this point
        // Re-initialize all 4 PHASE fields to IDLE (2)
        /*
        int cxms = (((((2 << 5) | 2) << 5) | 2) << 5) | 2;

        for (int i=0;i<128;i++) {
            int logical_id = lsnn_core_ids[i];
            if (logical_id != -1) {
                // Find physical address of logical cores
                
                logicalToPhysicalCoreId(logical_id, &coreId);
                nc = NEURON_PTR(coreId);

                // Reset neuro cores
                nx_fast_init32(&nc->cx_state, 2*numNeuronsPerCore, 0);
                nx_fast_init32(&nc->cx_meta_state, (numNeuronsPerCore + 3) / 4, cxms);
                nx_fast_init32(&nc->dendrite_accum, numDendAccumPerCore, 0);
            }
        }*/
        
        if (lsnn == 1) {
        
            DendriteAccumEntry accum = (DendriteAccumEntry) {.accum=0};

            CxState state = (CxState) {.U=0, .V=0};

            // Re-initialize all 4 PHASE fields to IDLE (2)
            MetaState cxms = (MetaState) {.Phase0=2, .SomaOp0=2, 
                                          .Phase1=2, .SomaOp1=2,
                                          .Phase2=2, .SomaOp2=2,
                                          .Phase3=2, .SomaOp3=2};


            nx_fast_init_multicore(ncs[0]->cx_state, 2*numNeuronsPerCore, sizeof(uint32_t), sizeof(uint32_t), &state, coreids, 64);
            nx_fast_init_multicore(ncs[0]->cx_meta_state, (numNeuronsPerCore + 3) / 4, sizeof(uint32_t), sizeof(uint32_t), &cxms, coreids, 64);
            nx_fast_init_multicore(ncs[0]->dendrite_accum, numDendAccumPerCore, sizeof(uint32_t), sizeof(uint32_t), &accum, coreids, 64);
        }
    }

    if (current_iteration >= 0 && (rs->time_step - (TIMESTEPS_PER_SAMPLE * current_iteration)) == TIMESTEP_START_RELAY) {
            
        //printf("#### Enabling started at %d\n", rs->time_step);

        
            
        //activate the relay neurons to send spikes into the FF neurons
        /*
        for (int i=0;i<128;i++) {
            int logical_id = relay_core_ids[i];

            if (logical_id != -1) {
                printf("#### <-- chip fast_init at  %d\n", rs->time_step);
                // Find physical address of logical cores
                logicalToPhysicalCoreId(logical_id, &coreId);
                nc = NEURON_PTR(coreId);

                //clear dendritic accumulator to clear spikes which are "on flight"
                nx_fast_init32(&nc->dendrite_accum, numDendAccumPerCore, 0);

                EnableUpdates();
                nx_flush_core(coreId);
                //printf("Enabled %d\n", logical_id);
            }
        }
        */
       
        if (lsnn != 1 && relay == 1) {
            //printf("#### <-- chip multi_init at  %d\n", rs->time_step);
            //UpdateCfg upt = (UpdateCfg) {.num_updates = (128 + 3) / 4, .num_stdp    = 0};
            DendriteAccumEntry accum = (DendriteAccumEntry) {.accum=0};
            nx_fast_init_multicore(ncs[0]->dendrite_accum, numDendAccumPerCore, sizeof(uint32_t), sizeof(uint32_t), &accum, coreids, 16);
            //nx_fast_init_multicore(&(ncs[0]->num_updates), 1, sizeof(uint32_t), sizeof(uint32_t), &upt, coreids, 16);
            
            for(int i=0; i<16; i++) {
                NeuronCore* myCore = NEURON_PTR(coreids[i]);
                myCore->num_updates = (UpdateCfg) {.num_updates = (128 + 3) / 4,
                                             .num_stdp    = 0};
            }
        }
        
        
    }
}