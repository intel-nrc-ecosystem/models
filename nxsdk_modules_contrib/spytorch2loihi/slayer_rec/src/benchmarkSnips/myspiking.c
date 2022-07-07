/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2021 Intel Corporation.

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

#include "myspiking.h"
#include "nxsdk.h"
#include "array_sizes.h" 

static int time = 0;
static CoreId core;
static uint16_t axon;
static int spike_index = 0;
static int spike_channel;
static bool advance_time;
static CoreId core_map[128];
#ifndef STREAM_ONCE
    static SpikesIn spikes_in[spikes_per_packet];
#else
    static SpikesIn spikes_in[num_packets * spikes_per_packet];
#endif
static char spike_channel_name[20];
static int chipid = -1;
static int lmtid = -1;

int do_spiking(runState *s) 
{   return 1;   }

void run_spiking(runState *s) 
{
    time = s->time_step;
    // printf("%d\n", time);
    
    // initialize
    if (time==1){
        // get chip id
        for(int i=0; i<nx_num_chips(); ++i)
            if(nx_nth_chipid(i).id == nx_my_chipid().id)
            {
                chipid = i;
                break;
            }
        // get lmt id
        for(int i=0; i<3; ++i)
            if(nx_coreid_lmt(i).id == nx_my_coreid().id)  
            {
                lmtid = i;
                break;
            }
        // get spike channel
        sprintf(spike_channel_name, "spikeAddresses_%d_%d", chipid, lmtid);
        // printf("Spike Channel: %s\n", spike_channel_name);
        spike_channel = getChannelID(spike_channel_name);
        if(spike_channel == -1) 
        {
            printf("Invalid spike_channel for %s\n", spike_channel_name);
        }
        // get physical coremap
        for(int i=0; i<128; i++)
            core_map[i] = nx_nth_coreid(i);
        
        #ifndef STREAM_ONCE
            //dummy as though we've just finished a packet to force reading a new packet
            spike_index = spikes_per_packet; 
        #else
            for(int i=0; i<num_packets; ++i)
                readChannel(spike_channel, spikes_in + i*spikes_per_packet, spikes_per_packet/16);
            // printf("Finished reading %d spikes\n", spikes_per_packet * num_packets);
        #endif
    }
    
    advance_time = false;
    
    // if(chipid==0 && lmtid==0 && time>7707590)    printf("%d\n", time);
    while (!advance_time) // until we see the command to move to the next timestep
    {
        #ifndef STREAM_ONCE
            // if we reached the end of the packet, get another packet
            if (spike_index == spikes_per_packet)
            {
                readChannel(spike_channel, &spikes_in[0], spikes_per_packet/16);
                spike_index = 0;
            }
        #else
            if (spike_index == spikes_per_packet * num_packets)
                spike_index = 0;
        #endif

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

    // if(chipid==0 && lmtid==0 && time%100==0)    printf("%d%c", time, time%2000==0 ? '\n' : ' ');
    // if(chipid==0 && lmtid==0 && time>7706000)    printf("%d%c", time, time%10==0 ? '\n' : ' ');
    // if(chipid==0 && lmtid==0 && time>7707590)    printf("Complete\n");
}
