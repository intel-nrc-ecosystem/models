/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2018 Intel Corporation.

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
#include "time.h"
#include "snip_init.h"

int numCores;
int resetInterval;
int enableReset;

char channelName[20];

uint16_t myChipId() {
  static bool valid=0;
  static uint16_t num_chips=0;
  ChipId myChipId = nx_my_chipid();
  if (!valid) {
    valid=1;
    ChipId min=nx_min_chipid();
    ChipId max=nx_max_chipid();
    uint16_t m=0;
    for (uint16_t x=min.x; x<=max.x; x++) {
      for (uint16_t y=min.y; y<=max.y; y++) {
        for (uint16_t z=min.z; z<=max.z; z++) {
          if (nx_chipid(x,y,z).id == myChipId.id) {
            return m;
          }
          m++;
        }
      }
    }
  }
  printf("ERROR: Illegal chip id.");
  exit(1);
}

void init_1(runState *s) {
    LOG("NxTF: Initializing...\n");
    sprintf(channelName,"channel_init_ch%d_lmt0", myChipId());
    LOG("NxTF:     channelName=%s\n", channelName);

    int channelID = getChannelID(channelName);
    LOG("NxTF:     channelId=%d\n", channelID);
    if (channelID == -1) {
      LOG("Invalid channelName %s\n", channelName);
      return;
    }

    readChannel(channelID, &numCores, 1);
    readChannel(channelID, &resetInterval, 1);
    readChannel(channelID, &enableReset, 1);

    LOG("NxTF:     numCores=%d, resetInterval=%d, enableReset=%d\n",
        numCores, resetInterval, enableReset);
}



