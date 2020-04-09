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

"""Unit test for ComposableDNN"""

import unittest

from nxsdk.composable.model import Model
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN
from nxsdk_modules_ncl.dnn.src.dnn_layers import NxInputLayer, NxConv2D, NxAveragePooling2D, NxFlatten, NxDense, NxModel


class TestComposableDNN(unittest.TestCase):
    """Unit test for ComposableDNN"""
    def setUpDNN(self) -> ComposableDNN:
        """Sets up a DNN"""
        # Specify input shape of network.
        inputShape = (16, 16, 3)

        #################
        # BUILD NETWORK #
        #################

        inputLayer = NxInputLayer(inputShape)

        x = NxConv2D(4, (3, 3))(inputLayer.input)
        x = NxAveragePooling2D()(x)
        x = NxFlatten()(x)
        x = NxDense(10)(x)

        DNNModel = NxModel(inputLayer.input, x)

        composableDNNModel = ComposableDNN(model=DNNModel, num_steps_per_img=100)
        return composableDNNModel

    def testComposableDNN(self):
        """Test composable dnn and run the pipeline"""
        composableDNNModel = self.setUpDNN()
        model = Model("dnn_pipeline")
        model.add(composableDNNModel)
        model.compile()

    def testComposableDNNWithInScopeVersion(self):
        """Tests the DNN Composable with the in-scope version"""
        with Model("dnn_pipeline") as model:
            dnn = self.setUpDNN()
            model.compile()
