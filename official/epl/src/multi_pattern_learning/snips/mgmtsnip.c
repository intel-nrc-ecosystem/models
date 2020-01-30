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
static int numPatternsLearned = 0;
//static int mcSomaNumSTDP = 0;

void enableLearning(void);
void changeGCToMCLearning(int patternIdx, bool disable);

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
    if (USE_LMT_SPIKE_COUNTERS) command = DO_NOTHING;
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
                numPatternsLearned++;
                if (numPatternsLearned < NUM_PATTERNS) {
                    command = SWITCH_TO_POSITIVE_THETA_TRAINING;
                }
                else command = CHANGE_MODE_AND_SWITCH_TO_POSITIVE_THETA;
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
                command = SWITCH_TO_POSITIVE_THETA_TESTING;
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

void applyInputs() {
    if (mcInputsChannelId == INVALID_CHANNEL_ID) initMCInputsChannel();
    int biasArray[NUM_MCS];
    LOG("reading MCAD input biases... \n");
    readChannel(mcInputsChannelId, &biasArray, NUM_MCS);
    for (int i = 0; i < NUM_MCS; i++) {
        int bias = biasArray[i];
        nxCompartmentGroup[MCAD_CXGRP_ID][i].Bias = bias;
        //LOG("BIAS for MC[%d]=%d \n",i,bias);
    }
}

void switchToPositiveThetaTest(runState *s) {
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
    nxCompartmentGroup[MCSOMA_CXGRP_ID].DisableLearning;
    for (int i = 0; i < NUM_PATTERNS; i++) {
        int gid = gcGrpIdsPerPattern[i];
        nxCompartmentGroup[gid].DisableLearning;
    }
}

void initMgmtChannel() {
    if(mgmtChannelId == INVALID_CHANNEL_ID) {
        mgmtChannelId = getChannelID("nxmgmt_input_axon_ids");
        if(mgmtChannelId == INVALID_CHANNEL_ID)
            printf("ERROR: Invalid channelID for nxinit\n");
    }
}

void changeMCToGCWeights(int coreId) {
    //CoreId core = nx_nth_coreid(coreId);
    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
    LOG("coreId=%d\n", coreId);
    int numAxons = NUM_MCS;
    int inputAxonId;
    DiscreteMapEntry SynMapEntry;
    SupperUnpackState supper;
    Synapse syn_unpack;
    Synapse synArray[128*NUM_MC_TO_GC_DELAYS]; // *conn_prob

    for (int n=0; n < numAxons; n++) {
        inputAxonId = 2 * n;
        LOG("MC->GC: Core Id=%d, Input axonId=%d\n", coreId, inputAxonId);
        int synMapEntryID = inputAxonId;
        int synIdx = 0;
        SynMapEntry = nc->synapse_map[synMapEntryID].discreteMapEntry;
        supper = initSupperUnpackState(SynMapEntry.Ptr, SynMapEntry.Len,
                                                    SynMapEntry.CxBase);
        while(synMemWordsToSyn(nc, &supper,  &syn_unpack) == 1) {
            synArray[synIdx] = syn_unpack;
            LOG("OLD:CIdx=%d, Wgt=%d, Dly=%d \n", synArray[synIdx].CIdx,
                            synArray[synIdx].Wgt, synArray[synIdx].Dly);
            if (syn_unpack.Wgt == 201) synArray[synIdx].Wgt = 0;
            LOG("NEW:CIdx=%d, Wgt=%d, Dly=%d \n", synArray[synIdx].CIdx,
                            synArray[synIdx].Wgt, synArray[synIdx].Dly);
            synIdx++;
        }
        //Write back to memory;returns synMemLen
        int numSyns = synIdx;
        LOG("**** MC->GC: MAX NUM SYNS = %d *****\n", numSyns);
        synToSynMemWords(synArray, numSyns, nc, SynMapEntry.Ptr, SynMapEntry.CxBase);
    }
}

void changeGCToMCWeights(int coreId) {
    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
    LOG("coreId=%d\n", coreId);
    int numAxons = NUM_GCS;
    int inputAxonId;
    DiscreteMapEntry SynMapEntry;
    SupperUnpackState supper;
    Synapse syn_unpack;
    Synapse synArray[2];
    int baseAxonId = NUM_MCS + 1;
    for (int n=0; n < numAxons; n++) {
        inputAxonId = baseAxonId + 2 * n;
        LOG("GC->MC EXC: Core Id=%d, Input axonId=%d\n", coreId, inputAxonId);
        int synMapEntryID = inputAxonId;
        int synIdx = 0;
        SynMapEntry = nc->synapse_map[synMapEntryID].discreteMapEntry;
        supper = initSupperUnpackState(SynMapEntry.Ptr, SynMapEntry.Len,
                                                    SynMapEntry.CxBase);
        while(synMemWordsToSyn(nc, &supper,  &syn_unpack) == 1) {
            synArray[synIdx] = syn_unpack;
            LOG("OLD: SynFmtId = %d, CIdx=%d, Wgt=%d, Dly=%d \n",
                synArray[synIdx].synFmtId, synArray[synIdx].CIdx,
                            synArray[synIdx].Wgt, synArray[synIdx].Dly);
            if (excSynFmtId == -1) {
                int fid = syn_unpack.synFmtId;
                if (nc->synapse_fmt[fid].FanoutType == F_EXC) {
                    excSynFmtId = fid;
                    LOG("******* EXC FANOUT ID: %d \n", excSynFmtId);
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
            LOG("NEW:SynFmtId = %d, CIdx=%d, Wgt=%d, Dly=%d \n",
                synArray[synIdx].synFmtId, synArray[synIdx].CIdx,
                            synArray[synIdx].Wgt, synArray[synIdx].Dly);
            synIdx++;
        }


        /*Write back to memory;returns synMemLen*/
        int numSyns = synIdx;
        LOG("**** GC->MC: MAX NUM SYNS = %d \n", numSyns);
        synToSynMemWords(synArray, numSyns, nc, SynMapEntry.Ptr, SynMapEntry.CxBase);
    }
}

void changeMCToGCConnections() {
    for (int coreId = GC_CORE_ID_BEGIN; coreId <= GC_CORE_ID_END; coreId++) {
        //LOG("coreId=%d\n", coreId);
        changeMCToGCWeights(coreId);
    }
}

void changeGCToMCExcConnections() {
        //LOG("coreId=%d\n", coreId);
        int coreId = 2;
        changeGCToMCWeights(coreId);
}

void switchToInference(runState *s) {
    LOG("MGMT %d: switch to inference \n", s->time_step);
    disableLearning();
    changeMCToGCConnections();
    changeGCToMCExcConnections();

    for (int i = 0; i < NUM_PATTERNS; i++) {
        int gid = gcGrpIdsPerPattern[i];
        nxCompartmentGroup[gid].Vth = 600;
    }
}


void changeMode(runState *s) {

    /*if (mgmtChannelId == INVALID_CHANNEL_ID) initMgmtChannel();
    readChannel(mgmtChannelId,&mode,1);
    LOG("MODE = %d \n", mode);*/
    mode = TESTING;
    switch(mode) {
        case TESTING:
            switchToInference(s); break;
        // just to supress compiler warnings; will not encounter this case
        case TRAINING:
            break;
    }
    switchToPositiveThetaTest(s);

}

void switchToPositiveThetaTrain(runState *s) {
    LOG("*** Continuing training *** \n");
    nxCompartmentGroup[MCSOMA_CXGRP_ID].Vth = 2;
    int GCGRP_PREV_PATTERN = gcGrpIdsPerPattern[numPatternsLearned-1];
    int GCGRP_NEXT_PATTERN = gcGrpIdsPerPattern[numPatternsLearned];
    nxCompartmentGroup[GCGRP_PREV_PATTERN].Vth = 131071;
    nxCompartmentGroup[GCGRP_NEXT_PATTERN].Vth = 600;
    thetaState = POSITIVE_THETA;
    LOG("TRAIN: APPLY INPUT (%d) \n", numPatternsLearned);
    applyInputs();
    enableLearning();
    tbeginCurrState = s->time_step;
}

void enableLearning() {
    int currentPatternIdx = numPatternsLearned;
    changeGCToMCLearning(currentPatternIdx, false);
}

void changeGCToMCLearning(int patternIdx, bool disable) {
    LOG(">>>>>>>>>>> patternIdx = %d, disable=%d \n", patternIdx, disable);
     // TODO: remove magic number
    int coreId = 2;
    NeuronCore *nc = NEURON_PTR(nx_nth_coreid(coreId));
    StdpPreProfileCfg pp1;
    int baseInpAxonId0 =  NUM_MCS + 1;
    int baseInpAxonId = baseInpAxonId0 + (2 * NUM_GCS_PER_PATTERN * patternIdx);
    LOG("base axonID=%d \n", baseInpAxonId);
    int inputAxonId, profileId;
    int offset = disable? 1 : -1;
    for (int i = 0; i < NUM_GCS_PER_PATTERN; i++) {
        inputAxonId = baseInpAxonId + (2 * i);
        LOG("CoreId=%d, Input axonId=%d\n", coreId, inputAxonId);
        int synMapEntryID = inputAxonId+1;
        profileId = nc->synapse_map[synMapEntryID].preTraceEntry1.StdpPreProfileCfg;
        nc->synapse_map[synMapEntryID].preTraceEntry1.StdpPreProfileCfg =
                                profileId + offset;
        profileId = nc->synapse_map[synMapEntryID].preTraceEntry1.StdpPreProfileCfg;
        pp1 = nc->stdp_pre_profile_cfg[profileId];
        LOG("New> StdpPreProfileCfg(%d), UpdateAlways(%d) \n", profileId ,pp1.UpdateAlways);
    }
}

void disableLearning2() {
    int currentPatternIdx = numPatternsLearned;
    int gid = gcGrpIdsPerPattern[currentPatternIdx];
    nxCompartmentGroup[gid].DisableLearning;
    changeGCToMCLearning(currentPatternIdx, true);
}

void runMgmt(runState *s) {
    LOG("\n MGMT %d: BEGIN command=%s \n", s->time_step, command2strings[command]);
    switch(command) {
        case DISABLE_LEARNING:
            disableLearning2(); break;
        case SWITCH_TO_POSITIVE_THETA_TRAINING:
            switchToPositiveThetaTrain(s); break;
        case SWITCH_TO_POSITIVE_THETA_TESTING:
            switchToPositiveThetaTest(s); break;
        case SWITCH_TO_NEGATIVE_THETA:
            switchToNegativeTheta(s); break;
        case CHANGE_MODE_AND_SWITCH_TO_POSITIVE_THETA:
            changeMode(s); break;
        case SWITCH_TO_INFERENCE:
        case DO_NOTHING:
            break;
    }
    if (USE_LMT_SPIKE_COUNTERS) updateSpikeCounters(s->time_step);

    LOG("\n MGMT %d: END command=%s \n", s->time_step, command2strings[command]);
}
