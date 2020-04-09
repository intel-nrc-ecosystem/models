/*

Copyright © 2018 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
*/

#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "snip_reset.h"

static int numNeuronsPerCore = 1024;
static int NUM_Y_TILES = 5;

int tImgStart = 0;
int tImgEnd = 0;

extern int numCores;
extern int resetInterval;
extern int enableReset;

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId) {
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}
/*
int do_write_input(runState *RunState) {
    if(enableReset && (RunState->time_step % resetInterval == 0)) {
        return 0;
    } else {
        return 0;
    }
}

void write_input(runState *RunState) {

    LOG("NxTF: Writing input at runState->time_step=%d...\n", RunState->time_step);
    
    // Input injection code goes here
    
    volatile TimeState dummy;
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        dummy = nc->time;
    }
}
*/
int do_reset(runState *RunState) {
    if (enableReset && (RunState->time_step == 1 || (RunState->time_step - 1) % resetInterval == 0)) {
        return 1;
    } else {
        return 0;
    }
}

void reset(runState *RunState) {

    NeuronCore *nc;
    CoreId coreId;

    LOG("NxTF: Resetting cores at runState->time_step=%d...\n", RunState->time_step);

    CxState cxs = (CxState) {.U=0, .V=0};
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
//        LOG("numCores=%d, core=%d, coreID=%d\n", numCores, i, coreId);
        nx_fast_init64(nc->cx_state, numNeuronsPerCore, *(uint64_t*)&cxs);
    }

    LOG("NxTF: Done resetting cx_state. %d\n", RunState->time_step);

    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
    }

    LOG("NxTF: Done resetting dendrite_accum. %d\n", RunState->time_step);

    MetaState ms = (MetaState) {.Phase0=2, .SomaOp0=3,
                                .Phase1=2, .SomaOp1=3,
                                .Phase2=2, .SomaOp2=3,
                                .Phase3=2, .SomaOp3=3};
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->cx_meta_state, numNeuronsPerCore/4, *(uint32_t*)&ms);
    }

    LOG("NxTF: Done resetting cx_meta_state. %d\n", RunState->time_step);

    volatile TimeState dummy;
    for(int i=0; i<numCores; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        dummy = nc->time;
    }

    if (LOGGING) {tImgEnd = clock();}
    LOG("NxTF: Runtime per img = %dms, Avg. runtime per step = %dus time_step %d\n",
        (tImgEnd-tImgStart)/1000, (tImgEnd-tImgStart)/resetInterval, RunState->time_step);
}


