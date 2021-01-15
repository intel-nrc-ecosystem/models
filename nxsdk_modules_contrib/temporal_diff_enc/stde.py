'''
This is a Loihi implementation of the spiking elementary motion detector (sEMD)/ temporal difference encoder (sTDE)
It converts a temporal difference between 2 spikes from different sources into a firing rate/ number of spikes

The sEMD has been introduced and used by:
- Milde, M. B., Bertrand, O. J., Ramachandran, H., Egelhaaf, M., & Chicca, E. (2018). Spiking elementary motion detector in neuromorphic systems. Neural computation, 30(9), 2384-2417.
- D'Angelo, G., Janotte, E., Schoepe, T., O'Keeffe, J., Milde, M. B., Chicca, E., & Bartolozzi, C. (2020). Event-based eccentric motion detection exploiting time difference encoding. Frontiers in Neuroscience, 14, 451.

This file was mainly written at the Telluride Neuromorphic Workshop 2019
by Alpha Renner (alpren@ini.uzh.ch)
Contributors:
Elisabetta Chicca
Garrick Orchard
Andreas Wild
Mike Davies

Version 1.3
Updated for nxsdk version 0.9.5

'''

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import nxsdk.api.n2a as nx
from nxsdk.utils.plotutils import plotRaster

def calculate_mant_exp(value, precision=8):
    """
    This function calculates the exponent and mantissa from a desired value. Can e.g. be used for weights.
    :param value: the value for which you want to calculate mantissa and exponent
    :param precision: the allowed precision in bits for the mantissa
    :return: mantissa, exponent
    """
    des_value = value
    exponent = 0
    while abs(value) > 2 ** precision:
        value /= 2
        exponent += 1
    if not int(value) * 2 ** exponent == des_value:
        print('desired value of normalized max weight:', des_value)
        print('actual value:', int(value) * 2 ** exponent, 'mantissa:', int(value), 'exponent:', exponent)
    return int(value), exponent


class STDE_group(object):
    def __init__(self, params, net=None, name=None):
        """
        The STDE_group contains the sTDE neurons.
        One neuron consists of 4 compartments that are connected as follows:
        
                    D (main/soma)
                    |
                    C (current)
                   / \\
        (trigger) A   B (facilitator)
                 
        A is the gate and lets B's current pass whenever it spikes.
        C receives B's current on its voltage variable and decays
        D receives C's voltage and integrates it, so C is basically a second current input to D
        
        The two inputs are called trigger and facilitator following Milde (2018)
        
        params are
        'tau_fac': current tau of facilitator input
        'tau_trigg': current tau of trigger input
        'tau_v': voltage tau of TDE Neuron
        'tau_c': current tau of TDE Neuron
        'weight_fac': amplitude of the facilitator spike
        'do_probes' : can be 'all', 'spikes' or None
        'num_neurons' : number of sTDE neurons that are created

        """
        if net is None:
            net = nx.NxNet()
            
        self.net = net

        self.num_neurons = params['num_neurons']
        self.neurongroups = {}
        self.probes = {}
        self.spikegens = {}

        weight_fac, exponent_fac = calculate_mant_exp(params['weight_fac']/params['tau_fac'],7)

        # Create auxiliary compartments
        cpA = nx.CompartmentPrototype(
            vThMant=1,
            compartmentCurrentDecay=int(1 / 1 * 2 ** 12),
            compartmentVoltageDecay=4095,
            # thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.NO_SPIKE_AND_PASS_V_LG_VTH_TO_PARENT
        )

        cpB = nx.CompartmentPrototype(
            vThMant=100,
            compartmentCurrentDecay=int(1 / params['tau_fac'] * 2 ** 12),
            compartmentVoltageDecay=4095
        )

        cpC = nx.CompartmentPrototype(
            vThMant=1000,
            compartmentCurrentDecay=int(1 / 1 * 2 ** 12),
            compartmentVoltageDecay=int(1 / params['tau_trigg'] * 2 ** 12),
            thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.NO_SPIKE_AND_PASS_V_LG_VTH_TO_PARENT
        )

        # Create main compartment
        cpD = nx.CompartmentPrototype(
            vThMant=100,
            compartmentCurrentDecay=int(1 / params['tau_v'] * 2 ** 12),
            compartmentVoltageDecay=int(1 / params['tau_c'] * 2 ** 12),
        )

        # build compartment tree
        cpC.addDendrites(prototypeA=cpA, prototypeB=cpB, joinOp=nx.COMPARTMENT_JOIN_OPERATION.PASS)
        cpD.addDendrite(prototype=[cpC], joinOp=nx.COMPARTMENT_JOIN_OPERATION.ADD)

        num_neurons = params['num_neurons']
        neuronPrototype = nx.NeuronPrototype(cpD)
        neurongroup = net.createNeuronGroup(prototype=neuronPrototype, size=num_neurons)

        sgpA = net.createSpikeGenProcess(numPorts=num_neurons)
        sgpB = net.createSpikeGenProcess(numPorts=num_neurons)
        # sgpC = net.createSpikeGenProcess(numPorts=1)

        connProto = nx.ConnectionPrototype(weight=weight_fac, weightExponent=exponent_fac)

        sgpA.connect(neurongroup.dendrites[0].dendrites[1], prototype=connProto, connectionMask=sp.sparse.identity(num_neurons))
        sgpB.connect(neurongroup.dendrites[0].dendrites[0], prototype=connProto, connectionMask=sp.sparse.identity(num_neurons))

        spikegens = [sgpA, sgpB]

        if params['do_probes'] == 'all':
            (uA, vA, sA) = neurongroup.dendrites[0].dendrites[1].probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                                                        nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                                                                        nx.ProbeParameter.SPIKE])

            (uB, vB, sB) = neurongroup.dendrites[0].dendrites[0].probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                                                        nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                                                                        nx.ProbeParameter.SPIKE])

            (uC, vC, sC) = neurongroup.dendrites[0].probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                                           nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                                                           nx.ProbeParameter.SPIKE])

            (uD, vD, sD) = neurongroup.soma.probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                                   nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                                                   nx.ProbeParameter.SPIKE])

            probes = {
                'A_current': uA,
                'A_voltage': vA,
                'A_spikes': sA,
                'B_current': uB,
                'B_voltage': vB,
                'B_spikes': sB,
                'C_current': uC,
                'C_voltage': vC,
                'C_spikes': sC,
                'D_current': uD,
                'D_voltage': vD,
                'D_spikes': sD,
            }
        elif params['do_probes'] == 'spikes':
            sA = neurongroup.dendrites[0].dendrites[1].probe([nx.ProbeParameter.SPIKE])
            sB = neurongroup.dendrites[0].dendrites[0].probe([nx.ProbeParameter.SPIKE])
            sC = neurongroup.dendrites[0].probe([nx.ProbeParameter.SPIKE])
            sD = neurongroup.soma.probe([nx.ProbeParameter.SPIKE])

            probes = {
                'A_spikes': sA,
                'B_spikes': sB,
                'C_spikes': sC,
                'D_spikes': sD,
            }
        else:
            probes = None

        self.neurongroup = neurongroup
        self.probes = probes
        self.spikegens = spikegens
        self.input0 = neurongroup.dendrites[0].dendrites[0]
        self.input1 = neurongroup.dendrites[0].dendrites[1]
        

    def add_spikes(self, spiketimes_a, indices_a, spiketimes_b, indices_b):
        for sg_neuron in range(self.num_neurons):
            self.spikegens[0].addSpikes(spikeInputPortNodeIds=sg_neuron,
                                   spikeTimes=np.asarray(spiketimes_a)[
                                       np.where(np.asarray(indices_a) == sg_neuron)[0]].tolist())
            self.spikegens[1].addSpikes(spikeInputPortNodeIds=sg_neuron,
                                   spikeTimes=np.asarray(spiketimes_b)[
                                       np.where(np.asarray(indices_b) == sg_neuron)[0]].tolist())

    def plot(self, num_steps=None):
        if num_steps is None:
            num_steps = len(self.probes['A_spikes'].data[0])
        
        num_probes = len(self.probes)
        
        # Plot the results
        fig = plt.figure(1, figsize=(18, 10))

        for i, key in enumerate(np.sort(list(self.probes.keys()))):
            plt.subplot(int(num_probes/3),3, i + 1)
            self.probes[key].plot()
            # plt.plot(probes[key].data.T)
            plt.title(key)
            plt.xlim([0, num_steps])
        
        plt.tight_layout()
