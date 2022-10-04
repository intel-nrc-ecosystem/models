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

#include <stdlib.h>
#include <string.h>
#include "runmgmt.h"
#include <time.h>
#include <unistd.h>

static int count = 0;
static int channelID = -1;
static int32_t spike_counts[num_packed*16] = {0}; 
static char channel_name[20];
static int chipid = -1;

int do_run_mgmt(runState *s) {
        
    if (s->time_step==1)
    {
        // get chip id
        for(int i=0; i<nx_num_chips(); ++i)
            if(nx_nth_chipid(i).id == nx_my_chipid().id)
            {
                chipid = i;
                break;
            }
        sprintf(channel_name, "nxspkcntr_%d", chipid);
        printf("Counter Channel: %s\n", channel_name);
        channelID = getChannelID(channel_name);
    }

    if ((count == timesteps_per_sample-1) || (count%512 == 0)) 
    {
        return 1; //read the spike counters
    }
    else
    {
        count = count + 1;
        return 0; //otherwise do nothing this time
    }
}

void run_mgmt(runState *s) 
{

    for(int probe_id = 0; probe_id<num_classes; probe_id++) 
    {
            //------- to probe on every timestep, use this:
            //spike_counts[probe_id] += SPIKE_COUNT[(s->time_step-1)&3][0x20+probe_id]; 
            //SPIKE_COUNT[(s->time_step-1)&3][0x20+probe_id] = 0;
        
            //but rather probe occasionally and use this:
            for(int ii=0; ii<4; ii++)
            {
                spike_counts[probe_id] += SPIKE_COUNT[ii][0x20+probe_id]; 
                SPIKE_COUNT[ii][0x20+probe_id] = 0;
            }
    }

    if (count == timesteps_per_sample-1) 
    {
        count = 0;
        // Write the spike counter value back to the channel and reset our spike counts
        writeChannel(channelID, spike_counts, 1);
        for(int probe_id = 0; probe_id<num_classes; probe_id++)
            spike_counts[probe_id] = 0;

        // printf("Wrote back spikes\n");
    }
    else
    {
        count = count + 1;
    }
}
