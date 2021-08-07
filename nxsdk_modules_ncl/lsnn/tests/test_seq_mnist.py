# Copyright © 2018-2021 Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#   * Neither the name of Intel Corporation nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest
from test import support

from nxsdk_modules.lsnn.apps.seqmnist.seq_mnist import runSequentialMnist


class TestSMNIST(unittest.TestCase):
    """Unit test for sMNIST module."""

    def test_full(self):
        """Workload test."""

        expectation = [8, 7, 1, 7, 7, 2, 1, 9, 5, 4, 5, 6, 5, 2, 0, 7, 6, 4, 6,
                       4, 5, 7, 8, 9, 8, 7, 4, 3, 9, 4, 2, 2, 9, 7, 6, 9, 1, 2,
                       8, 3, 0, 6, 4, 0, 8, 7, 5, 2, 9, 8, 4, 3, 4, 0, 7, 0, 7,
                       4, 6, 1, 3, 7, 2, 8, 8, 0, 1, 0, 5, 1, 5, 1, 4, 9, 4, 8,
                       7, 3, 7, 8, 9, 1, 4, 4, 0, 2, 5, 7, 7, 5, 3, 0, 6, 2, 4,
                       1, 6, 8, 6, 9]

        expName = 'v25_94per'
        sqic = runSequentialMnist(expName, numSamples=100, batchSize=10)

        c =sqic.classifications

        # print(list(c))

        self.assertSequenceEqual(expectation, list(c))


if __name__ == '__main__':
    support.run_unittest(TestSMNIST)
