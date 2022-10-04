/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2019-2021 Intel Corporation.

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

#include "myreset2.h"
#include "nxsdk.h"
#include "array_sizes.h" 

// ------------------------------------------------ LK

#define LAYER_RESET_INTERVAL 269 // 269
#define NUM_RESET_CORES 128 // 58

// copied from slayer/.../myspking2.c
int do_reset(runState *s) {
    int rst;
    // if((s->time_step - 1) % LAYER_RESET_INTERVAL == 169) {
    if((s->time_step - 1) % LAYER_RESET_INTERVAL == 169 || (s->time_step - 1) % LAYER_RESET_INTERVAL == 260) {
        rst = 1;
    }
    else {
        rst = 0;
    }
    return rst;
}

// copied from /ioSnips/iomgmt.c
void reset_cores_fx(int core_start_id, int num_cores_to_reset) {
    // TODO put into non-spiking mode -> reset everyting -> put back into spiking mode
    CxState cxs = (CxState) {.U=-50000, .V=-200000};
    // CxState cxs = (CxState) {.U=0, .V=0};
    int phaseReset = 4; // 0
    int somaReset = 3; // 3
    MetaState ms = (MetaState) {
        .Phase0=phaseReset, .SomaOp0=somaReset,
        .Phase1=phaseReset, .SomaOp1=somaReset,
        .Phase2=phaseReset, .SomaOp2=somaReset,
        .Phase3=phaseReset, .SomaOp3=somaReset};
    
    NeuronCore* nc;
    for(int core_id=core_start_id; core_id<(core_start_id+num_cores_to_reset); ++core_id) {
        nc = NEURON_PTR(nx_nth_coreid(core_id));
        for(int cx_id=0; cx_id<1024; ++cx_id) {
            nc->cx_state[cx_id] = cxs;
            nc->cx_meta_state[cx_id] = ms;
        }
    }
}

// copied from /ioSnips/iomgmt.c
void set_cores_fx(int core_start_id, int num_cores_to_reset) {
    CxState cxs = (CxState) {.U=0, .V=0};
    MetaState ms = (MetaState) {
        .Phase0=2, .SomaOp0=3,
        .Phase1=2, .SomaOp1=3,
        .Phase2=2, .SomaOp2=3,
        .Phase3=2, .SomaOp3=3};
    
    NeuronCore* nc;
    for(int core_id=core_start_id; core_id<(core_start_id+num_cores_to_reset); ++core_id) {
        nc = NEURON_PTR(nx_nth_coreid(core_id));
        for(int cx_id=0; cx_id<1024; ++cx_id) {
            nc->cx_state[cx_id] = cxs;
            nc->cx_meta_state[cx_id] = ms;
        }
    }
}

void layer_reset_mgmt_fx(runState *s) {
    if((s->time_step - 1) % LAYER_RESET_INTERVAL == 169) {
        reset_cores_fx(0, NUM_RESET_CORES);
    }
    else {
        set_cores_fx(0, NUM_RESET_CORES);
    }
}

// ------------------------------------------------