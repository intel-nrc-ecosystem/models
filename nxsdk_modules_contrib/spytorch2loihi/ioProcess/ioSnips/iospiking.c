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
#include "iospiking.h"

static int chipid = -1;
static int lmtid = -1;
static int spiking_state = 0;

#ifdef PROFILE_TIME
static char timer_channel_name[20];
static int timer_channel_id = -1;
// static int total_time = 0;
static int t_st = -1;
static int t_en = -1;
static int time_log[NUM_PROFILE_TIME_STEPS];
static bool profile_time = false;
#endif

int do_run_spiking(runState *s) {
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

        #ifdef PROFILE_TIME
            t_st = timestamp();
            sprintf(timer_channel_name, "time_log_%d_%d", chipid, lmtid);
            timer_channel_id = getChannelID(timer_channel_name);
            if(timer_channel_id != -1)
                printf("Time log output channel %s successfully established.\n", timer_channel_name);
        #endif
    }

    spiking_state = 0;

    #ifdef PROFILE_TIME
    if(chipid==0 && lmtid==0) {  
        if(s->time_step < NUM_PROFILE_TIME_STEPS) {
            profile_time = true;
            spiking_state = 1;
        }
    }
    #endif

    return spiking_state;
}

void run_spiking(runState *s) {
    #ifdef PROFILE_TIME
    t_en = timestamp(); 
    time_log[s->time_step-1] = t_en - t_st;
    t_st = t_en;
    if(s->time_step == NUM_PROFILE_TIME_STEPS-1) {
        writeChannel(timer_channel_id, time_log, NUM_PROFILE_TIME_STEPS);
    }
    
    profile_time = false;
    #endif
}