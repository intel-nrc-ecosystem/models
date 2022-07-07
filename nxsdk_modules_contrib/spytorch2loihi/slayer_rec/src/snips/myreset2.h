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

// ------------------------------------------------ LK

int do_reset(runState *s);
void reset_cores_fx(int core_start_id, int num_cores_to_reset);
void set_cores_fx(int core_start_id, int num_cores_to_reset);
void layer_reset_mgmt_fx(runState *s);

// ------------------------------------------------