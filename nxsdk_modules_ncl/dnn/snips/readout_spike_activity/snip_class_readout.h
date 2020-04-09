/*

Copyright © 2020 Intel Corporation.

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

#ifndef NXSDK_MODULES_DNN_SNIPS_READOUT_SPIKE_ACTIVITY_SNIP_CLASS_READOUT_H_
#define NXSDK_MODULES_DNN_SNIPS_READOUT_SPIKE_ACTIVITY_SNIP_CLASS_READOUT_H_

#include "utils.h"

int do_readout(runState *RunState);
void readoutSpikes(runState *RunState);
void readoutVoltage(runState *RunState);
void readout(runState *RunState);

#endif  // NXSDK_MODULES_DNN_SNIPS_READOUT_SPIKE_ACTIVITY_SNIP_CLASS_READOUT_H_
