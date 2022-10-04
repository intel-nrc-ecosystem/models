/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2018-2021 Intel Corporation.

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

#ifndef DVS_SNIP_H
#define DVS_SNIP_H
#include "nxsdk.h"

#define COMPARTMENTS_PER_CORE 1024
#define WIDTH 128
#define HEIGHT 128

/*
typedef struct __attribute__((packed)) {
  uint8_t x;
  uint8_t y;
} DvsSpike;

typedef struct __attribute__((packed)) {
  DvsSpike spikes[DVS_SPIKES_PER_MESSAGE];
  uint32_t polarity;
} DvsData;
*/
int do_dvs_snip_injection(runState *s);
void dvs_snip_injection(runState *s);

#endif
