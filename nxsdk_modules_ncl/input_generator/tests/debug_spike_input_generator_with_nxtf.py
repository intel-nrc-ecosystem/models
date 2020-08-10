import os
import numpy as np
import matplotlib.pyplot as plt

from nxsdk_modules_ncl.dnn.src.dnn_layers import NxInputLayer, NxModel, \
    NxDense, ProbableStates, InputModes
from nxsdk.composable.model import Model as ComposableModel
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN
from nxsdk_modules_ncl.input_generator.spike_input_generator import \
    SpikeInputGenerator


os.environ['SLURM'] = '1'
# os.environ['PARTITION'] = 'nahuku08'

compartment_kwargs = {'vThMant': 1,
                      'compartmentVoltageDecay': 4095,
                      'compartmentCurrentDecay': 4095,
                      'functionalState': 0,
                      'enableSomaTrace': True,
                      'refractoryDelay': 0,
                      }

num_input_neurons = 1024
num_output_neurons = 1

input_layer = NxInputLayer(num_input_neurons,
                           compartmentKwargs=compartment_kwargs,
                           inputMode=InputModes.AEDAT)

layer = NxDense(num_output_neurons,
                compartmentKwargs=compartment_kwargs)(input_layer.input)

nxmodel = NxModel(input_layer.input, layer)

nxmodel.summary()

nxmodel.compile()

num_steps_per_img = num_input_neurons + 1

nxmodel_composable = ComposableDNN(nxmodel, num_steps_per_img)
input_generator = SpikeInputGenerator(name='SpikeGen')
input_generator.connect(nxmodel_composable)
input_generator.processes.inputEncoder.executeAfter(
    nxmodel_composable.processes.reset)
model = ComposableModel('debug')
model.add(nxmodel_composable)
model.add(input_generator)

model.compile()

p = ProbableStates.ACTIVITY
probes_spikes_input = [neuron.probe(p) for neuron in nxmodel.layers[0]]
p = ProbableStates.VOLTAGE
probes_voltage_input = [neuron.probe(p) for neuron in nxmodel.layers[0]]
probes_voltage_output = [neuron.probe(p) for neuron in nxmodel.layers[-1]]

model.start(nxmodel.board)

inputs = []
for n in range(num_input_neurons):
    inputs.append((n, n + 1))
input_generator.encode(inputs)
model.run(num_steps_per_img)
model.finishRun()

spikes_input = np.stack([p.data[-num_steps_per_img:]
                         for p in probes_spikes_input], 1)

voltage_input = np.stack([p.data[-num_steps_per_img:]
                          for p in probes_voltage_input], 1)

for p in probes_voltage_input:
    p.plot()
plt.show()
probes_voltage_output[0].plot()
plt.show()

model.disconnect()
