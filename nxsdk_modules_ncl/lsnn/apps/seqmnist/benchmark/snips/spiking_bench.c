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

#include "spiking.h"
#include <time.h>

extern int num_input;
extern int img_size;
extern int batch_size;
extern int cores;
extern ADDRESS_MAP inputPortMap[NUM_INPUTS];

static unsigned char img_data[IMAGE_SIZE * BATCH_SIZE];
static int tBatch;
static int current_image_idx;
static int current_pixel_idx;
static CoreId coreId;
static int pixelChannelId = -1;
static int packed_data[PACKED_MODE_SIZE] = {0};

static int measurements[PACKED_MODE_SIZE] = {0};
static int processTimeRead = 0;
static int processTimeProcess = 0;
static int tStart = 0;
static int tEnd = 0;


int do_spiking(runState *s) {
    // Inject spikes every time step
    return 1;
}

void run_spiking(runState *rs) {
    // Update batch time and indices
    tBatch = ((rs->time_step-1) % TIMESTEPS_PER_BATCH + 1);
    current_image_idx = ((tBatch-1) / TIMESTEPS_PER_IMAGE);
    current_pixel_idx = ((tBatch-1) % TIMESTEPS_PER_IMAGE);

    // Calculate physical coreId only once
    if (rs->time_step == 1) {
        logicalToPhysicalCoreId(inputPortMap[2].coreId, &coreId);
        pixelChannelId = getChannelID("nxspk_img_data");
    }

    // Read new batch of images at beginning of batch
    if (tBatch == 1) {
        tStart = clock();
        read_new_batch_of_images();
        tEnd = clock();
        processTimeRead += tEnd - tStart;
    }

    // Write benchmark results
    if (rs->time_step > 83999) {
        measurements[0]= processTimeRead;
        measurements[1]= (int) (processTimeProcess);
        writeChannel(getChannelID("measurements_spk"), measurements, 1);
    }

    // Generate and inject spikes
    tStart = clock();
    process_image(rs->time_step);
    tEnd = clock();
    processTimeProcess += tEnd - tStart;
}

void read_new_batch_of_images() {
    for(int i =0; i< (IMAGE_SIZE * BATCH_SIZE / PACKED_MODE_SIZE);i++) {
        readChannel(pixelChannelId, packed_data, 1);
        for(int k=i*PACKED_MODE_SIZE, j=0; k<(i*PACKED_MODE_SIZE+PACKED_MODE_SIZE); k++, j++)
            img_data[k] = packed_data[j];
     }
}

void process_image(int time){

    if (current_pixel_idx < IMAGE_SIZE) {
        threshold_line_scan_spikes_inject(time);
    } else {
        output_cue_duration_spikes_inject(time);
    }
}

void threshold_line_scan_spikes_inject(int time) {
    int axonId;
    for (int t = 1; t < NUM_THRESHOLDS; t++) {
        int threshold = thresholds[t];
        if (t != NUM_THRESHOLDS - 1) {
            axonId = inputPortMap[2*t].regId;
            if(get_onset_spikes(threshold)) {
                inject_spike(axonId, time, (2*t));
            }
            axonId = inputPortMap[2*t+1].regId;
            if(get_offset_spikes(threshold)) {
                inject_spike(axonId, time, (2*t+1));
            }
        } else {
            axonId = inputPortMap[2*t].regId;
            if(get_touch_spikes(threshold)) {
                inject_spike(axonId, time, (2*t));
            }
        }
    }
}

void output_cue_duration_spikes_inject(int time) {
    int axonId = inputPortMap[NUM_INPUTS-1].regId;
    inject_spike(axonId, time, (NUM_INPUTS-1));
}

void inject_spike(int axonId, int time, int port) {
    if (axonId == -1) {
        printf("\nSqIC: ERROR - Trying to send spikes to illegal spikeInputPort id=%d\n", port);
    } else {
        nx_send_discrete_spike(time, coreId, axonId);
    }
}

int get_onset_spikes(int threshold) {
    if (current_pixel_idx == LAST_PIXEL) return 0;

    int current_idx = current_image_idx * IMAGE_SIZE + current_pixel_idx;
    int next_idx = current_idx + 1;

    if (img_data[current_idx] < threshold && img_data[next_idx] >= threshold) return 1;
    return 0;

}

int get_offset_spikes(int threshold) {
    if (current_pixel_idx == LAST_PIXEL) return 0;

    int current_idx = current_image_idx * IMAGE_SIZE + current_pixel_idx;
    int next_idx = current_idx + 1;

    if (img_data[current_idx] >= threshold && img_data[next_idx] < threshold) return 1;
    return 0;

}

int get_touch_spikes(int threshold) {
    int current_idx = current_image_idx * IMAGE_SIZE + current_pixel_idx;

    if (img_data[current_idx] == threshold) return 1;
    else return 0;
}
