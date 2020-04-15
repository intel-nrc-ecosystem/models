// Copyright(c) 2019-2020 Intel Corporation All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the
//     distribution.
//   * Neither the name of Intel Corporation nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "nxsdk.h"
#include "initsnip.h"
#include "constants.h"
#include "common.h"


static int channelID = INVALID_CHANNEL_ID;

void initChannel() {
    if(channelID == INVALID_CHANNEL_ID) {
        channelID = getChannelID("nxinit");
        if(channelID == INVALID_CHANNEL_ID) {
          printf("ERROR: Invalid channelID for nxinit\n");
        }
    }
}


void readInputs() {
    int inputData[NUM_MCS];
    readChannel(channelID, &inputData, NUM_MCS);
    for (int i = 0; i < NUM_MCS; i++) {
        int bias = inputData[i];
        nxCompartmentGroup[MCAD_CXGRP_ID][i].Bias = bias;
        //LOG("BIAS for MC[%d]=%d \n",i,bias);
    }
}

void dumpConstants() {
    LOG("NUM_CORES = %d \n", NUM_CORES);
    LOG("NUM_MCS_PER_CORE = %d \n", NUM_MCS_PER_CORE);
    LOG("NUM_GCS_PER_CORE = %d \n", NUM_GCS_PER_CORE);
    LOG("NUM_MCS = %d \n", NUM_MCS);
    LOG("NUM_GCS = %d \n", NUM_GCS);
    LOG("MCAD_CXGRP_ID = %d \n", MCAD_CXGRP_ID);
    LOG("MCSOMA_CXGRP_ID = %d \n", MCSOMA_CXGRP_ID);
}

void initParamsAndInputs(runState *s) {
    //dumpConstants();
    initChannel();
    readInputs();
    thetaState = POSITIVE_THETA;
    mode = TRAINING;
    if (mode == TRAINING) LOG("INIT: TRAINING MODE \n");
    tbeginCurrState = s->time_step;
    LOG("DONE INIT: tstep %d \n", tbeginCurrState);
}
