/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2018-2021 Intel Corporation.

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
#include "spiking.h"
#include <time.h>
#include <unistd.h>

int spikeCountCx = 0;
int channelIDST = -1;
int channelIDACK = -1;
int noSolution = -1;
int falsePositive = -2;
int probe_id = 0; //  ID of the spike_probe, if only summation is probed it will be 0
int condition = 0;
int lfsr_offset = 0; 
int Dummy = 0;

int do_spiking(runState *s) {// Run SNIP right from the start of simulation and get channel ID
if (s->time_step==1) {
    channelIDST = getChannelID("nxsummlmt");
    channelIDACK = getChannelID("nxstacknow");
    }
    if (s->time_step < lfsr_offset+2) {
    SPIKE_COUNT[(s->time_step-1)&3][0x20+probe_id] = 0;
    return 0;
    }
    else if (Dummy==0){
    return 1;
    }
    else if (Dummy==1){
    return 0;
    }
    else {
    return 0;
    }
}

// Read spike counter, accumulate value in spikeCountCx, reset counter to zero and write spikeCountCx to channel to host
void run_spiking(runState *s) {
    if (SPIKE_COUNT[(s->time_step)&3][0x20+probe_id]==1) {
                  printf("SPIKE ON SNIP t=%d and count = %d\n", s->time_step,SPIKE_COUNT[(s->time_step-1)&3][0x20+probe_id]);
        SPIKE_COUNT[(s->time_step)&3][0x20+probe_id] = 0;   // Clear LMT spike counters to prevent overflow
        if (s->time_step>lfsr_offset+2){
        int solutionTime=s->time_step-1;
        printf("LMT is writing to channel %d \n", solutionTime);
        writeChannel(channelIDST, &solutionTime, 1);            // Write solution time to channel
        readChannel(channelIDACK, &Dummy, 1);
        printf("\n LMT read acknowledgement %d \n", Dummy);
        if (Dummy==0){
            printf("\n LMT will write handshake %d \n", Dummy);
             writeChannel(channelIDST, &falsePositive, 1);            // Write solution time to channel
            }
        else {
            }
        }
        }
    else if (s->time_step == s->total_steps && Dummy==0) {
        Dummy=3;
        printf("\n LMT will write -1 to channel \n");
        writeChannel(channelIDST, &noSolution, 1);             // Write -1 to channel if no solution registered
        printf("\n LMT wrote -1 to channel \n");
        }
}
