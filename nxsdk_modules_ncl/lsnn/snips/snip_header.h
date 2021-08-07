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

#pragma once
#include "nxsdk.h"
#include "basic_sim_params.h"

#define NUM_Y_TILES 5

#define NUM_OUTPUTS 181

#define TIMESTEPS_PER_SAMPLE (LSNN_TIME_LENGTH - RELAY_TIME_LENGTH + SIM_TIME_LENGTH + 10)
#define TIMESTEP_START_RELAY (LSNN_TIME_LENGTH - RELAY_TIME_LENGTH + 1)
#define TIMESTEP_STOP_LSNN (LSNN_TIME_LENGTH + 2)

#define TIMESTEP_START_READOUT (LSNN_TIME_LENGTH - RELAY_TIME_LENGTH + SIM_TIME_LENGTH - READOUT_TIME_LENGTH + 12)
#define TIMESTEP_RESET_OFFSET (LSNN_TIME_LENGTH - RELAY_TIME_LENGTH + SIM_TIME_LENGTH + 11)

typedef struct {
    int boardId;
    int chipId;
    int coreId;
    int regId;
} ADDRESS_MAP;

int do_mgmt(runState *rs);
void run_mgmt(runState *rs);

uint16_t myChipId();
void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId);
