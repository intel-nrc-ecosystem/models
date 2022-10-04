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

#include "gesture.h"

static int channelID = -1;
static CoreId core;
static uint32_t axon;
 
int do_dvs_snip_injection(runState *s){

    if(s->time_step==1){
      channelID = getChannelID(DVS_LIVE_RECEIVE_NAME);
      if(channelID == -1) 
          printf("Invalid channelID for nxdvs_data\n");
    }

    return 1;
}



void dvs_snip_injection(runState *s) {
    uint32_t us = 1000;
    uint64_t durationTicks = us * TICKS_PER_MICROSECOND;
    
    uint64_t now = timestamp();
    uint64_t deadline = now + durationTicks;

    DvsData data;

    while(now < deadline)
    {
          uint8_t avail = probeChannel(channelID);
          for (int ii = 0; ii < avail; ++ii)
          {
                readChannel(channelID,&data,1); 
                
                for (int ii = 0; ii < DVS_LIVE_SPIKES_PER_MESSAGE; ++ii) 
                {
                      
                  uint8_t x = data.spikes[ii].x;
                  uint8_t y = data.spikes[ii].y;
                  uint8_t pol = (data.polarity & 1);
                  data.polarity >>= 1;

                  // check range
                  if (x < (WIDTH + 56) && y < (HEIGHT + 26))
                  {
                    if (x >= 56 && y >= 26) 
                    {
                      // compute core/axon
                      uint32_t index = (x-56)*(HEIGHT*2)+(127-(y-26))*2+pol; //flip x and y
                      uint32_t logicalCore = index/COMPARTMENTS_PER_CORE;
                      
                      core = nx_nth_coreid(logicalCore);
                      axon  = index % COMPARTMENTS_PER_CORE;

                      //send spike
                      nx_send_discrete_spike(s->time_step, core, axon);
                    }
                  }
                }
          }
      now = timestamp();
    }
}
