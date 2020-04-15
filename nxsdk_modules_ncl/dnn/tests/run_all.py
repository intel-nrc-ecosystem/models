#
# Copyright © 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

"""Run all DNN tests."""

from nxsdk_modules_ncl.dnn.tests import test_dnn_compiler, test_dnn_tutorials, test_softreset
from nxsdk_modules_ncl.dnn.tests import test_dnn_mapper
from nxsdk_modules_ncl.dnn.tests import test_dnn_partitioner



if __name__ == '__main__':
    test_dnn_compiler.main()
    test_dnn_mapper.main()
    test_dnn_partitioner.main()
    test_dnn_tutorials.main()
    test_softreset.main()
