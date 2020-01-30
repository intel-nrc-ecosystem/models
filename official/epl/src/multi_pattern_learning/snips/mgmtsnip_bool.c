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

#include "mgmtsnip.h"

#define FINISHED_TRAINING 1
static int mgmtChannelId = INVALID_CHANNEL_ID;
static int mcInputsChannelId = INVALID_CHANNEL_ID;
static int spikeCounterChannelId = INVALID_CHANNEL_ID;
static int statusChannelId = INVALID_CHANNEL_ID;
//static int mcSomaSpikeCounters[NUM_MCS];
static int excSynFmtId = -1;
static int numTest = 0;
static int total_steps;


int doMgmt(runState *s) {

    if (USE_LMT_SPIKE_COUNTERS==1 && s->time_step==1){
        spikeCounterChannelId = getChannelID("nxspkcntr");
        total_steps = RUN_TIME; //s->total_steps;
        //printf("*****TOTAL TIME STEPS = %d \n", total_steps);
    }

    if (s->time_step == 1) {
        statusChannelId = getChannelID("status");
    }

    int stateDuration;
    int timeElapsed = s->time_step - tbeginCurrState;
    int runCommand = 0;
    /*LOG("MGMT %d: stateDuration %d, timeElapsed %d\n", s->time_step, stateDuration,
    timeElapsed);*/
    command = DO_NOTHING;
    if (mode == TRAINING) {
        // Positive theta cycle
        if (thetaState == POSITIVE_THETA) {
            stateDuration = POSITIVE_THETA_PERIOD_TRAIN;
            if (timeElapsed == stateDuration - NO_LEARNING_PERIOD) {
                command = DISABLE_LEARNING;
                runCommand = 1;
            }
            else if (timeElapsed == stateDuration) {
                command = SWITCH_TO_NEGATIVE_THETA;
                 runCommand = 1;
            }
        }
        else {
            // Negative theta cycle
            if (timeElapsed == NEGATIVE_THETA_PERIOD) {
                command = CHANGE_MODE_AND_SWITCH_TO_POSITIVE_THETA;
                 runCommand = 1;
            }
        }

    }

    if (mode == TESTING) {
        // Positive theta cycle
        if (thetaState == POSITIVE_THETA) {
            if (timeElapsed == POSITIVE_THETA_PERIOD_TEST) {
                command = SWITCH_TO_NEGATIVE_THETA;
                 runCommand = 1;
            }
        }
        else {
            // Negative theta cycle
            if (timeElapsed == NEGATIVE_THETA_PERIOD) {
                command = SWITCH_TO_POSITIVE_THETA;
                 runCommand = 1;
            }
        }

    }

    if (USE_LMT_SPIKE_COUNTERS) return 1;
    else return runCommand;
}

int readAndResetSpikeCounter(int probeId, int time_step) {
    int idx = 0x20 + probeId;
    int t = time_step - 1;
    if (SPIKE_COUNT[t&3][idx] >= 1) {
        SPIKE_COUNT[t&3][idx] = 0;
        //printf("t=%d: cxID=%d spiked at  \n", t-1, probeId);
        return 1;
    }
    else {
        SPIKE_COUNT[t&3][idx] = 0;
        return 0;
    }
}

void updateSpikeCounters(int time_step) {
    LOG("TIMESTEP = %d:updating spike counters...\n", time_step);
    if (time_step == 1) return;
    int t = time_step - 2;
    int hasSpiked;
    int marker = total_steps + 10;
    writeChannel(spikeCounterChannelId, &t, 1);

    for (int cxId=0; cxId < NUM_MCS; cxId++) {
        hasSpiked = readAndResetSpikeCounter(cxId, time_step);
        if (hasSpiked) writeChannel(spikeCounterChannelId, &cxId, 1);
    }
    if (time_step == total_steps) marker++;
    writeChannel(spikeCounterChannelId, &marker, 1);
}

void initMCInputsChannel() {
    if(mcInputsChannelId == INVALID_CHANNEL_ID) {
        mcInputsChannelId = getChannelID("nxmgmt_mc_inputs");
        if(mcInputsChannelId == INVALID_CHANNEL_ID) {
            printf("ERROR: Invalid channelID for nxmgmt_mc_inputs \n");
        }
    }
}

void initMgmtChannel() {
    if(mgmtChannelId == INVALID_CHANNEL_ID) {
        mgmtChannelId = getChannelID("nxmgmt");
        if(mgmtChannelId == INVALID_CHANNEL_ID)
            printf("ERROR: Invalid channelID for nxinit\n");
    }
}

void changeMCToGCWeights(int coreId, bool turnOffLearning) {
    //CoreId core = nx_nth_coreid(coreId);
    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
    LOG("MC>GC: coreId=%d, turnOffLearning=%d\n", coreId, turnOffLearning);

    int numAxons;
    readChannel(mgmtChannelId,&numAxons,1);
    int inputAxonId;

    DiscreteMapEntry SynMapEntry;
    SupperUnpackState supper;
    Synapse syn_unpack;
    int SIZE = NUM_GCS_PER_CORE * NUM_MC_TO_GC_DELAYS; // times conn_prob
    //LOG("##### SIZE=%d \n", SIZE);
    Synapse synArray[SIZE];

    for (int n=0; n < numAxons; n++) {
        readChannel(mgmtChannelId,&inputAxonId,1);
        //LOG("MC->GC: Core Id=%d, Input axonId=%d\n", coreId, inputAxonId);
        int synMapEntryID = inputAxonId;
        int synIdx = 0;
        SynMapEntry = nc->synapse_map[synMapEntryID].discreteMapEntry;
        supper = initSupperUnpackState(SynMapEntry.Ptr, SynMapEntry.Len,
                                                    SynMapEntry.CxBase);
        while(synMemWordsToSyn(nc, &supper,  &syn_unpack) == 1) {
            synArray[synIdx] = syn_unpack;
            /*LOG("CIdx=%d, Wgt=%d, Dly=%d \n", synArray[synIdx].CIdx,
                            synArray[synIdx].Wgt, synArray[synIdx].Dly);*/
            if (turnOffLearning == false) {
                if (syn_unpack.Wgt == 201) synArray[synIdx].Wgt = 0;
            }
            else synArray[synIdx].LrnEn = 0;
            synIdx++;
        }
        //Write back to memory;returns synMemLen
        int numSyns = synIdx;
        //LOG("**** MC->GC: MAX NUM SYNS = %d *****\n", numSyns);
        //if (turnOffLearning == false) {
        (synArray, numSyns, nc,
                            SynMapEntry.Ptr, SynMapEntry.CxBase);
                           // }
    }
}

void changeGCToMCWeights(int coreId, bool turnOffLearning) {
    //CoreId core = nx_nth_coreid(coreId);
    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
    //LOG("coreId=%d\n", coreId);
    LOG("GC>MC: coreId=%d, turnOffLearning=%d\n", coreId, turnOffLearning);

    int numAxons;
    readChannel(mgmtChannelId,&numAxons,1);
    int inputAxonId;

    DiscreteMapEntry SynMapEntry;
    SupperUnpackState supper;
    Synapse syn_unpack;
    Synapse synArray[2*NUM_MCS_PER_CORE];

    for (int n=0; n < numAxons; n++) {
        readChannel(mgmtChannelId,&inputAxonId,1);
        /*LOG("GC->MC EXC: Core Id=%d, Input axonId=%d\n", coreId,
                    inputAxonId);*/
        int synMapEntryID = inputAxonId;
        int synIdx = 0;
        SynMapEntry = nc->synapse_map[synMapEntryID].discreteMapEntry;
        supper = initSupperUnpackState(SynMapEntry.Ptr, SynMapEntry.Len,
                                                    SynMapEntry.CxBase);
        while(synMemWordsToSyn(nc, &supper,  &syn_unpack) == 1) {
            synArray[synIdx] = syn_unpack;
           /* LOG("OLD: SynFmtId = %d, CIdx=%d, Wgt=%d, Dly=%d \n",
                synArray[synIdx].synFmtId, synArray[synIdx].CIdx,
                            synArray[synIdx].Wgt, synArray[synIdx].Dly);*/
            //if (syn_unpack.Wgt == 201) synArray[synIdx].Wgt = 0;
            if (turnOffLearning == false) {
                if (excSynFmtId == -1) {
                    int fid = syn_unpack.synFmtId;
                    if (nc->synapse_fmt[fid].FanoutType == F_EXC) {
                        excSynFmtId = fid;
                        //LOG("******* EXC FANOUT ID: %d \n", excSynFmtId);
                    }
                }

                if (syn_unpack.Dly >= 0) {
                    if (syn_unpack.Dly > 40) synArray[synIdx].Dly = 40;
                    if (syn_unpack.synFmtId == excSynFmtId) {
                        synArray[synIdx].Wgt = 4;
                    }
                    else {
                        synArray[synIdx].Wgt = -7;
                        synArray[synIdx].Dly -= 1;
                    }
                }
            }
            else synArray[synIdx].LrnEn = 0;
            LOG("NEW:SynFmtId = %d, CIdx=%d, Wgt=%d, Dly=%d, LrnEn=%d\n",
                synArray[synIdx].synFmtId, synArray[synIdx].CIdx,
                synArray[synIdx].Wgt, synArray[synIdx].Dly,
                synArray[synIdx].LrnEn);
            synIdx++;
        }


        /*Write back to memory;returns synMemLen*/
        int numSyns = synIdx;
        //LOG("**** GC->MC: MAX NUM SYNS = %d \n", numSyns);
        //if (turnOffLearning == false) {
        synToSynMemWords(synArray, numSyns, nc,
                            SynMapEntry.Ptr, SynMapEntry.CxBase);
                            //}
    }
}

void changeMCToGCConnections(bool turnOffLearning) {
    int numCores, coreId;
    readChannel(mgmtChannelId,&numCores,1);
    LOG("NUM CORES=%d\n", numCores);
    for (int i=0; i < numCores; i++) {
        readChannel(mgmtChannelId,&coreId,1);
        //LOG("coreId=%d\n", coreId);
        changeMCToGCWeights(coreId, turnOffLearning);
    }
}

void changeGCToMCExcConnections(bool turnOffLearning) {
    int numCores, coreId;
    readChannel(mgmtChannelId,&numCores,1);
    LOG("NUM CORES=%d\n", numCores);
    for (int i=0; i < numCores; i++) {
        readChannel(mgmtChannelId,&coreId,1);
        //LOG("coreId=%d\n", coreId);
        changeGCToMCWeights(coreId, turnOffLearning);
    }

}


void switchToInference(runState *s) {

    LOG("MGMT %d: switch to inference \n", s->time_step);
    //nxCompartmentGroup[MCAD_CXGRP_ID].DisableLearning;
    changeMCToGCConnections(false);
    changeGCToMCExcConnections(false);
}

void changeMode(runState *s) {

    if (mgmtChannelId == INVALID_CHANNEL_ID) initMgmtChannel();
    readChannel(mgmtChannelId,&mode,1);
    LOG("MODE = %d \n", mode);

    switch(mode) {
        case TESTING:
            switchToInference(s); break;
    }
    switchToPositiveTheta(s);
}

void applyInputs() {
    if (mcInputsChannelId == INVALID_CHANNEL_ID) initMCInputsChannel();
    int biasArray[NUM_MCS];
    LOG("waiting for test input \n");
    readChannel(mcInputsChannelId, &biasArray, NUM_MCS);
    for (int i = 0; i < NUM_MCS; i++) {
        int bias = biasArray[i];
        nxCompartmentGroup[MCAD_CXGRP_ID][i].Bias = bias;
        //LOG("BIAS for MC[%d]=%d \n",i,bias);
    }
}

void switchToPositiveTheta(runState *s) {
    nxCompartmentGroup[MCSOMA_CXGRP_ID].Vth = 2;
    thetaState = POSITIVE_THETA;
    if (numTest < NUM_TEST_SAMPLES) {
        LOG("TEST: APPLY INPUT (%d) \n", numTest);
        applyInputs();
    }
    int status = FINISHED_TRAINING;
    if (numTest == 0) writeChannel(statusChannelId, &status, 1);
    //LOG("TEST: APPLY INPUT - numtest(%d) \n", numTest);
    tbeginCurrState = s->time_step;
    numTest++;
}

void switchToNegativeTheta(runState *s) {
    //LOG("Reset MC AD Bias and Voltages \n");
    nxCompartmentGroup[MCAD_CXGRP_ID].V = 0;
    nxCompartmentGroup[MCAD_CXGRP_ID].Bias = 0;
    //LOG("Reset MC Soma Vth=255 \n");
    nxCompartmentGroup[MCSOMA_CXGRP_ID].Vth = 255;
    thetaState = NEGATIVE_THETA;
    tbeginCurrState = s->time_step;
}


void disableLearning() {
    //LOG("Disabling learning \n");
    nxCompartmentGroup[MCAD_CXGRP_ID].DisableLearning;
    for (int i=0; i < NUM_PATTERNS; i++) {
        int gid = gcGrpIdsPerPattern[i];
        nxCompartmentGroup[gid].DisableLearning;
    }
}

void runMgmt(runState *s) {
    LOG("\n MGMT %d: BEGIN command=%s \n", s->time_step, command2strings[command]);
    switch(command) {
        case DISABLE_LEARNING:
            disableLearning(); break;
        case SWITCH_TO_POSITIVE_THETA:
            switchToPositiveTheta(s); break;
        case SWITCH_TO_NEGATIVE_THETA:
            switchToNegativeTheta(s); break;
        case CHANGE_MODE_AND_SWITCH_TO_POSITIVE_THETA:
            changeMode(s); break;
        case DO_NOTHING:
            break;
    }
    if (USE_LMT_SPIKE_COUNTERS) updateSpikeCounters(s->time_step);
    LOG("\n MGMT %d: END command=%s \n", s->time_step, command2strings[command]);
}
