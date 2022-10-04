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
// #define INPUT_START_CORE 0
// #define INPUT_NEURONS_PER_CORE 256

int do_spiking(runState *s);
void run_spiking(runState *s);

typedef struct __attribute__((packed)) {
  uint16_t axon;
  uint8_t core, chip;
  } SpikesIn;