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

#include "myspiking0.h"
#include "nxsdk.h"
#include "array_sizes.h" 

static int time = 0;
static CoreId core;
static uint16_t axon;
static int spike_index = 0;
static int channelID;
static bool advance_time;
static CoreId core_map[128];
static SpikesIn spikes_in[spikes_per_packet];

int do_spiking0(runState *s) {
    return 1;
}

void run_spiking0(runState *s) {

    time = s->time_step;
    
    // initialize
    if (time==1){
        channelID = getChannelID("spikeAddresses0");
        if(channelID == -1) {
              printf("Invalid channelID for spikeAddresses0\n");
        }
        
        for(int ii=0; ii<128; ii++)
            core_map[ii] = nx_nth_coreid(ii);
        //dummy as though we've just finished a packet to force reading a new packet
        spike_index = spikes_per_packet; 
    }
    
    advance_time = false;
    
    while (!advance_time) // until we see the command to move to the next timestep
    {
        // if we reached the end of the packet, get another packet
        if (spike_index == spikes_per_packet)
        {
            readChannel(channelID, &spikes_in[0], spikes_per_packet/16);
            spike_index = 0;
        }
        
        axon = (1<<14) | spikes_in[spike_index].axon;
        advance_time = axon == (3<<13); //the condition for advancing time
            
        // only inject spikes, not "advance time" messages
        if(!advance_time)
        {
            core = core_map[spikes_in[spike_index].core];
            nx_send_remote_event(time, nx_my_chipid(), core, axon);
        }
        spike_index = spike_index + 1;
    }
}
