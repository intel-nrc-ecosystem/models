"""Script to run a SLAYER-trained SNN on Loihi, using the NxTF API."""

from collections import OrderedDict

import os
import numpy as np

from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition
from nxsdk.composable.model import Model as ComposableModel
from nxsdk_modules.slayer.src.slayer2loihi import Slayer2Loihi
from nxsdk_modules.slayer.tutorials.nmnist.nmnistDataset import NmnistDataset
from nxsdk_modules_ncl.dnn.composable.composable_dnn import ComposableDNN
from nxsdk_modules_ncl.input_generator.spike_input_generator import \
    SpikeInputGenerator
from nxsdk_modules_ncl.dnn.src.dnn_layers import NxInputLayer, NxModel, \
    NxDense, NxFlatten, InputModes
from nxsdk_modules_ncl.snntoolbox.nx_backend import print_performance, \
    save_performance_stats
from snntoolbox.simulation.plotting import plot_execution_time_probe, \
    plot_energy_probe, plot_power_probe


# SETTINGS #
############

# Define board to use for running SNN.
os.environ['SLURM'] = '1'
os.environ['PARTITION'] = 'nahuku32_2h'
os.environ['BOARD'] = 'ncl-ext-ghrd-01'

# Enable to measure energy and execution time.
probe_performance = True
buffer_size = 1024
bin_size = 32
if probe_performance:
    assert os.environ['BOARD'] == 'ncl-ext-ghrd-01', \
        "For performance measurements use board 'ncl-ext-ghrd-01'"

# Define compartment and connection properties, as used during SLAYER trining..
compartment_kwargs = {'vThMant': 80,
                      'compartmentVoltageDecay': 128,
                      'compartmentCurrentDecay': 1024,
                      'vMinExp': 23,
                      'biasExp': 0,
                      'functionalState': 0,
                      'refractoryDelay': 1
                      }
connection_kwargs = {'numDelayBits': 0,
                     'numTagBits': 0,
                     'numWeightBits': 8,
                     'synapseEncoding': 'dense1',
                     'weightExponent': 0,
                     'delay': 0,
                     'enableDelay': 0,
                     'signMode': 1,
                     'weightLimitExp': 0
                     }

# Input layer uses lower threshold and faster decay than hidden layers
# (this layer just propagates DVS events through).
compartment_kwargs_input = compartment_kwargs.copy()
compartment_kwargs_input['vThMant'] = 1
compartment_kwargs_input['compartmentVoltageDecay'] = 4095
compartment_kwargs_input['compartmentCurrentDecay'] = 4095

num_to_test = 200

# SLAYER uses reset to zero for the membrane potentials after spike generation.
reset_mode = 'hard'

# Resetting network states globally between NMNIST digits is not needed here
# because our neurons are leaky. Turning off reset speeds up inference.
reset_between_samples = False

# The input layer uses two channels for ON and OFF events in the DVS stream.
input_shape = (34, 34, 2)
num_input_neurons = int(np.prod(input_shape))
input_shape_flat = (1, num_input_neurons, 1)
num_classes = 10

# DATASET #
###########

dataset = NmnistDataset('/nfs/ncl/datasets/NMNIST_test')

# Number of timesteps to run each digit for.
num_steps_per_img = dataset.sampleLength

# WEIGHTS #
###########

# Get SLAYER weights.
path_weights = os.path.join(Slayer2Loihi.getModels(), '02_NMNIST', 'Trained')

# MODEL #
#########

# INPUT LAYER
input_layer = NxInputLayer(input_shape_flat,
                           resetMode=reset_mode,
                           compartmentKwargs=compartment_kwargs_input,
                           connectionKwargs=connection_kwargs,
                           inputMode=InputModes.AEDAT)

# The flatten layer is just needed to remove the dummy axes (1, ..., 1) in the
# input layer.
layer = NxFlatten()(input_layer.input)

# HIDDEN LAYER
name = 'NMNISTFc1'
weights = np.load(os.path.join(path_weights, name + '.npy'))

# SLAYER helper function to determine how to encode weights efficiently.
weights, num_weight_bits, weight_exponent, sign_mode = \
    Slayer2Loihi.optimizeWeightBits(weights)
# Apply optimized connection settings.
conn_kwargs = connection_kwargs.copy()
conn_kwargs['numWeightBits'] = int(num_weight_bits)
conn_kwargs['weightExponent'] = int(weight_exponent)
conn_kwargs['signMode'] = sign_mode

# Weight matrix needs to be transposed when going from SLAYER (Pytorch) to
# NxTF (Keras)
weights = weights.transpose()

# Switching from x-y-p co-ordinates to a fully connected layer requires
# re-ordering
idxs = np.arange(num_input_neurons)
permutation = np.ravel(np.reshape(idxs, input_shape, 'C'), 'F')
inverse_permutation = np.arange(len(permutation))[np.argsort(permutation)]
weights = weights[inverse_permutation]

num_neurons = 512

# SLAYER does not use biases.
biases = np.zeros(num_neurons)

layer = NxDense(num_neurons,
                weights=[weights, biases],
                compartmentKwargs=compartment_kwargs,
                connectionKwargs=conn_kwargs,
                resetMode=reset_mode,
                name=name)(layer)

# OUTPUT LAYER
name = 'NMNISTFc2'
weights = np.load(os.path.join(path_weights, name + '.npy'))
weights, num_weight_bits, weight_exponent, sign_mode = \
    Slayer2Loihi.optimizeWeightBits(weights)
conn_kwargs = connection_kwargs.copy()
conn_kwargs['numWeightBits'] = int(num_weight_bits)
conn_kwargs['weightExponent'] = int(weight_exponent)
conn_kwargs['signMode'] = sign_mode
weights = weights.transpose()

num_neurons = num_classes
layer = NxDense(num_neurons,
                weights=[weights, np.zeros(num_neurons)],
                compartmentKwargs=compartment_kwargs,
                connectionKwargs=conn_kwargs,
                resetMode=reset_mode,
                name=name)(layer)

# COMPILATION #
###############

# Compile complete model.
nxmodel = NxModel(input_layer.input, layer)
nxmodel.summary()
nxmodel.compile()  # Keras compile function. Not yet invoking NxTF compiler.

# Use the composablility interface of NxSDK to wrap our NxTF model. Allows
# more efficient input injection, output readout, and resetting via snips.
nxmodel_composable = ComposableDNN(nxmodel, num_steps_per_img,
                                   enable_reset=reset_between_samples)
model = ComposableModel('nmnist_mlp')
input_generator = SpikeInputGenerator(name='SpikeGen', packetSize=256,
                                      numSnipsPerChip=3, queueSize=512)

model.add(nxmodel_composable)
model.add(input_generator)
input_generator.connect(nxmodel_composable)
input_generator.processes.inputEncoder.executeAfter(
    nxmodel_composable.processes.reset)

# Call NxTF compiler to map model onto Loihi.
model.compile()

# PROBES #
##########

# Create performance probe.
if probe_performance:
    condition = PerformanceProbeCondition(
        tStart=buffer_size*bin_size+1,
        tEnd=num_to_test*num_steps_per_img, bufferSize=buffer_size,
        binSize=bin_size)
    performance_probe = nxmodel.board.probe(ProbeParameter.ENERGY, condition)
else:
    performance_probe = None

# INPUT #
#########

print("Preparing input spikes.")
inputs_encoded = OrderedDict()
for i in range(num_to_test):
    y = dataset[i].y
    x = dataset[i].x
    t = dataset[i].t + i * num_steps_per_img + 1
    p = dataset[i].p
    # Encode pixel-address, timestamp, and polarity as single number.
    addr = x * input_shape[0] * input_shape[2] + y * input_shape[2] + p
    # Apply encoding expected by SpikeInputGenerator.
    inp_encoded = input_generator.prepare_encoding(np.column_stack([addr, t]))
    # Group the encoded spikes of each sample by chip and embedded processor.
    for chip_id, inputs_per_chip in inp_encoded.items():
        inputs_encoded.setdefault(chip_id, OrderedDict())
        for cpu_id, inputs_per_cpu in inputs_per_chip.items():
            inputs_encoded[chip_id].setdefault(cpu_id, [])
            inputs_encoded[chip_id][cpu_id].extend(inputs_per_cpu)

# Start run.
model.start(nxmodel.board)
model.run(num_steps_per_img * num_to_test, aSync=True)

# Inject input.
print("Sending encoded input spikes.")
input_generator.send_inputs(inputs_encoded)

# Read output.
outputs = nxmodel_composable.readout_channel.read(num_to_test)
accuracy = np.mean(np.equal(outputs, dataset.labels[:len(outputs)]))
print("Accuracy after {} samples: {:.2%}.".format(num_to_test, accuracy))

model.finishRun()
model.disconnect()

# Show performance results.
if probe_performance:
    outdir = '.'
    stats = model.board.energyTimeMonitor.powerProfileStats
    print_performance(stats, num_steps_per_img)
    save_performance_stats(stats, outdir, num_steps_per_img)
    plot_execution_time_probe(outdir, performance_probe)
    plot_energy_probe(outdir, performance_probe)
    plot_power_probe(outdir, performance_probe)
