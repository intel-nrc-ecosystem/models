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

#ifndef CONSTANTS_H 
#define CONSTANTS_H 

#define INVALID_CHANNEL_ID -1 
#define NUM_CORES 72 
#define NUM_MCS_PER_CORE 1 
#define NUM_GCS_PER_CORE 3 
#define NUM_MCS (NUM_CORES * NUM_MCS_PER_CORE) 
#define NUM_GCS (NUM_CORES * NUM_GCS_PER_CORE) 
#define NUM_MC_TO_GC_DELAYS 1 
#define MCAD_CXGRP_ID 1 
#define MCSOMA_CXGRP_ID 2 
#define GAMMA_CYCLE_DURATION 40 
#define NUM_GAMMA_CYCLES_TRAIN 45 
#define NUM_GAMMA_CYCLES_TEST 5 
#define POSITIVE_THETA_PERIOD_TRAIN (GAMMA_CYCLE_DURATION * NUM_GAMMA_CYCLES_TRAIN) 
#define POSITIVE_THETA_PERIOD_TEST (GAMMA_CYCLE_DURATION * NUM_GAMMA_CYCLES_TEST) 
#define NEGATIVE_THETA_PERIOD 200 
#define NO_LEARNING_PERIOD 20 
#define NUM_TEST_SAMPLES 11 
#define USE_LMT_SPIKE_COUNTERS 1 
#define RUN_TIME 6400 

#endif 
