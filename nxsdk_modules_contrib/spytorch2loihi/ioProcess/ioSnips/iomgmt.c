/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2021-2022 Intel Corporation.

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
#include <time.h>
#include <unistd.h>
#include "array_sizes.h"
#include "iomgmt.h"

static int chipid = -1;
static int lmtid = -1;
static int mgmt_state = 0;

#ifdef USE_BIAS_INPUT
static char input_channel_name[20];
static int input_channel_id = -1;
static char input_data[INPUT_NUM_PACKED] = {0};
static CxCfg input_cx_cfg;
static bool bias_input_mgmt = false;
#endif

#ifdef USE_VOLTAGE_OUTPUT
static char output_channel_name[20];
static int output_channel_id = -1;
static int output[NUM_OUTPUTS] = {0};
static int output_core_ids[NUM_OUTPUTS] = OUTPUT_CORE_IDS;
static int output_cx_ids[NUM_OUTPUTS] = OUTPUT_CX_IDS;
static bool voltage_output_mgmt = false;
#endif

#ifdef USE_LAYER_RESET
static CoreId reset_core_ids[NUM_RESET_CORES] = {0};
static int reset_cores[NUM_RESET_CORES] = LAYER_RESET_CORES;
static int layer_core_start[NUM_LAYERS] = LAYER_CORE_START;
static int layer_chip_start[NUM_LAYERS] = LAYER_CHIP_START;
static int layer_chip_end[NUM_LAYERS] = LAYER_CHIP_END;
static int num_cores[NUM_LAYERS] = NUM_CORES_IN_LAYER;
static bool layer_reset_mgmt = false;
#endif

int do_run_mgmt(runState *s) {
    // setup at first time step
    if (s->time_step==1){
        // get chip id
        for(int i=0; i<nx_num_chips(); ++i)
            if(nx_nth_chipid(i).id == nx_my_chipid().id) {
                chipid = i;
                break;
            }
        // get lmt id
        for(int i=0; i<3; ++i)
            if(nx_coreid_lmt(i).id == nx_my_coreid().id) {
                lmtid = i;
                break;
            }
        // get spike channel
        #ifdef USE_BIAS_INPUT
            sprintf(input_channel_name, "bias_input_%d_%d", chipid, lmtid);
            input_channel_id = getChannelID(input_channel_name);
            if(input_channel_id != -1)
                printf("Bias input channel %s successfully established.\n", input_channel_name);
            // else
            //     printf("Invalid channel ID for %s\n", input_channel_name);

            // read a sample CxCfg from what is previously available
            input_cx_cfg = NEURON_PTR(nx_nth_coreid(0))->cx_cfg[0];
        #endif

        #ifdef USE_VOLTAGE_OUTPUT
            sprintf(output_channel_name, "voltage_output_%d_%d", chipid, lmtid);
            output_channel_id = getChannelID(output_channel_name);
            if(output_channel_id != -1)
                printf("Voltage output channel %s successfully established.\n", output_channel_name);
            // else
            //     printf("Invalid channel ID for %s\n", output_channel_name);
        #endif

        #ifdef USE_LAYER_RESET
            if(lmtid == LAYER_RESET_LMT) {
                // create logical core to physical core map   
                for(int i=0; i<NUM_RESET_CORES; ++i) {
                    reset_core_ids[i] = nx_nth_coreid(reset_cores[i]);
                }

                // correct information in reset table for layers that span multiple cores
                for(int layer=0; layer<NUM_LAYERS; ++layer) {
                    if(layer_chip_start[layer] != layer_chip_end[layer]) { // layer spans multiple chip
                        int chip = layer_chip_start[layer];
                        int cores_remaining = num_cores[layer];
                        int core_start = layer_core_start[layer];
                        int layer_num_cores = 0;

                        while(chip <= chipid && cores_remaining > 0) {
                            if(cores_remaining < (128 - core_start))    layer_num_cores = cores_remaining;
                            else                                        layer_num_cores = 128 - core_start;

                            if(chip == chipid) {
                                layer_chip_start[layer] = chipid;
                                layer_chip_end[layer] = chipid;
                                layer_core_start[layer] = core_start;
                                num_cores[layer] = layer_num_cores;
                            }

                            cores_remaining -= layer_num_cores;
                            core_start = 0; // core starts at 0 in the next chip
                            chip++;
                        }
                    }
                }

                // // reorder layer reset cores
                // for(int layer=0; layer<NUM_LAYERS; ++layer) {
                //     if(layer_chip_start[layer] == layer_chip_end[layer]) { // this should always be satisfied after table modification
                //         if(layer_chip_start[layer] == chipid) {
                //             for(int i=1; i<num_cores[layer]/2; i+=2) { // reorder 0, 1, 2, ..., n-2, n-1 to 0, n-1, 2, ..., n-2, 1
                //                 if(chipid == 0) printf("%d, %d\n", i, num_cores[layer] - i);
                //                 CoreId temp = reset_core_ids[layer_core_start[layer] + i];
                //                 reset_core_ids[layer_core_start[layer] + num_cores[layer] - i] = reset_core_ids[layer_core_start[layer] + i];
                //                 reset_core_ids[layer_core_start[layer] + num_cores[layer] - i] = temp;
                //             }                            
                //         }
                //     } 
                // }
            }
        #endif
    }

    // management functions
    #ifdef NUM_TIME_STEPS
        if((s->time_step) >= NUM_TIME_STEPS)    printf("time: %d\n", s->time_step);
        if((s->time_step) >= NUM_TIME_STEPS)    return 0;
    #endif

    mgmt_state = 0;

    #ifdef USE_BIAS_INPUT
    if(input_channel_id != -1)
        if((s->time_step - 1) % BIAS_INPUT_INTERVAL == 0) { // time always starts at 1
            bias_input_mgmt = true;
            mgmt_state = 1;
        }
    #endif

    #ifdef USE_VOLTAGE_OUTPUT
    if(output_channel_id != -1)
        if((s->time_step - 1) % VOLTAGE_OUTPUT_INTERVAL == VOLTAGE_OUTPUT_OFFSET % VOLTAGE_OUTPUT_INTERVAL) {
            voltage_output_mgmt = true;
            mgmt_state = 1;
        }
    #endif

    #ifdef USE_LAYER_RESET
    if(lmtid == LAYER_RESET_LMT)
        if((s->time_step - 1) % LAYER_RESET_INTERVAL < NUM_LAYERS) {
            layer_reset_mgmt = true;
            mgmt_state = 1;
        }
    #endif

    // printf("time: %d\n", s->time_step);

    return mgmt_state;
}

#ifdef USE_BIAS_INPUT
void bias_input_mgmt_fx(runState *s) {
    // static int tic, toc;
    // tic = timestamp();
    int index = lmtid;
    int cx_id, core_id;
    for(int packet=0; packet<INPUT_NUM_PACKETS; ++packet) {
        readChannel(input_channel_id, (int*)input_data, INPUT_NUM_PACKED/4/16);

        for(int i=0; i<INPUT_NUM_PACKED; ++i) {
            if(index < NUM_INPUTS) {
                cx_id = index % INPUT_NEURONS_PER_CORE;
                core_id = index / INPUT_NEURONS_PER_CORE;
                input_cx_cfg.Bias = input_data[i];
                NEURON_PTR(nx_nth_coreid(core_id))->cx_cfg[cx_id] = input_cx_cfg;
            }
            index+=NUM_BIAS_INPUT_SNIPS;
        }
    }
    // toc = timestamp();
    // printf("At time %d, took %d us to read bias.\n", s->time_step, (toc - tic)/400); // clock frequency is 400Mhz
}
#endif

#ifdef USE_VOLTAGE_OUTPUT
void voltage_output_mgmt_fx(runState *s) {
    for(int i=0; i<NUM_OUTPUTS; ++i) {
        output[i] = NEURON_PTR(nx_nth_coreid(output_core_ids[i]))->cx_state[output_cx_ids[i]].V;
    }

    writeChannel(output_channel_id, output, NUM_OUTPUTS);
}
#endif

#ifdef USE_LAYER_RESET
void reset_cores_fx(int core_start_id, int num_cores_to_reset) {
    int neurons_per_core = 1024; // for now
    NeuronCore* nc = NEURON_PTR(reset_core_ids[core_start_id]);
    CxState cxs = (CxState) {.U=0, .V=0};
    MetaState ms = (MetaState) {
            .Phase0=2, .SomaOp0=3,
            .Phase1=2, .SomaOp1=3,
            .Phase2=2, .SomaOp2=3,
            .Phase3=2, .SomaOp3=3};
    
    nx_fast_init_multicore(
        nc->cx_state,
        neurons_per_core,
        sizeof(CxState),
        sizeof(CxState),
        &cxs,
        reset_core_ids + core_start_id,
        num_cores_to_reset);
    
    // Do not reset dendriteaccum. Use the information present.
    // nx_fast_init_multicore(
    //     nc->dendrite_accum,
    //     neurons_per_core * 8192 / 1024,
    //     sizeof(DendriteAccumEntry),
    //     sizeof(DendriteAccumEntry),
    //     0,
    //     reset_core_ids + core_start_id,
    //     num_cores_to_reset);

    nx_fast_init_multicore(
        nc->cx_meta_state,
        neurons_per_core/4,
        sizeof(MetaState),
        sizeof(MetaState),
        &ms,
        reset_core_ids + core_start_id,
        num_cores_to_reset);

    for(int core=core_start_id; core<(core_start_id+num_cores_to_reset); ++core)
        nx_flush_core(reset_core_ids[core]);

    // for (int ii=0; ii < num_cores; ii++) {
    //     nx_flush_core(core_map[start_core+ii]);
    // }
}

void layer_reset_mgmt_fx(runState *s) {
    int layer = (s->time_step - 1) % LAYER_RESET_INTERVAL;

    if(layer_chip_start[layer] == layer_chip_end[layer]) { // this should always be satisfied after table modification
        if(layer_chip_start[layer] == chipid) {
            // layer cores are completely in this same chip
            reset_cores_fx(layer_core_start[layer], num_cores[layer]);
        }
    } 
}
#endif

void run_mgmt(runState *s) {
    #ifdef USE_BIAS_INPUT
        if(bias_input_mgmt==true)   bias_input_mgmt_fx(s);
        bias_input_mgmt = false;
    #endif

    #ifdef USE_VOLTAGE_OUTPUT
        if(voltage_output_mgmt==true)   voltage_output_mgmt_fx(s);
        voltage_output_mgmt = false;
    #endif

    #ifdef USE_LAYER_RESET
        if(layer_reset_mgmt==true)  layer_reset_mgmt_fx(s);
        layer_reset_mgmt = false;
    #endif
}
