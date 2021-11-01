'''
This is a Loihi implementation of the Time Difference Encoder (TDE) / spiking Elementary Motion Detector (sEMD).
It converts a temporal difference between two spikes from different sources into a firing rate (number of spikes).

The TDE has been introduced and used by:
- Milde, M. B., Bertrand, O. J., Ramachandran, H., Egelhaaf, M., & Chicca, E. (2018). Spiking elementary motion detector in neuromorphic systems. Neural computation, 30(9), 2384-2417.
- D'Angelo, G., Janotte, E., Schoepe, T., O'Keeffe, J., Milde, M. B., Chicca, E., & Bartolozzi, C. (2020). Event-based eccentric motion detection exploiting time difference encoding. Frontiers in Neuroscience, 14, 451.

This file was started at the Telluride Neuromorphic Workshop 2019
Contributors:
Alpha Renner (alpren@ini.uzh.ch)
Lyes Khacef (l.khacef@rug.nl)
Elisabetta Chicca
Garrick Orchard
Andreas Wild
Mike Davies

Version 1.4
Updated for nxsdk version 1.0.0
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


class TDE_group(object):
    def __init__(self, params, net=None, name=None):
        """
        The TDE_group contains the TDE neurons.
        One neuron consists of 4 compartments that are connected as follows:
        
                    D (main/soma)
                    |
                    C (current)
                   / \\
        (trigger) A   B (facilitator)
                 
        A is the gate and lets B's voltage pass whenever it spikes;
        B integrates (leaky integrator) incoming spikes with all-to-all (interagtive) or nearest (capped) spike interaction;
        C receives B's voltage when A spikes (leaky integrator);
        D receives C's voltage and fires when it reaches its threshold (leaky integrate and fire).
        
        The two inputs are called trigger and facilitator following Milde (2018).

        The facilitator B compartment can behvave in either
        'integrative' way, where every spike produces a jump BY a fixed value (i.e. approximates the spike rate over a short time window) or 
        'capped' way, where evey spike produces a jump TO a fixed value (i.e. approximates the time since the last spike)
        where the jump value is modulated bu the facilitator synapse weight, and the 'capped' TDE has a delay of one time step in this implementation.
        
        The parameters are
        'fac_type': can be 'integrative' or 'capped'
        'tau_fac': current tau of facilitator input
        'tau_trig': current tau of trigger input
        'tau_soma': voltage tau of TDE Neuron
        'wgt_fac': amplitude of the facilitator spike
        'do_probes' : can be 'all', 'spikes' or None
        'num_neurons' : number of TDE neurons that are created
        """

        if net is None:
            net = nx.NxNet()
            
        self.net = net

        self.capped = True if params['fac_type'] == 'capped' else False
        self.num_neurons = params['num_neurons']
        self.neurongroups = {}
        self.probes = {}
        self.spikegens = {}

        wgt_fac, exponent_fac = calculate_mant_exp(params['wgt_fac']/params['tau_fac'],7)

        # create auxiliary compartments
        # create compartment A that receives trigger spikes (without integration) and will act as a gate
        cpA = nx.CompartmentPrototype(
            vThMant=1,
            compartmentCurrentDecay=4096,
            compartmentVoltageDecay=4096,
        )

        # create compartment B that receives facilitator spikes and integrates them in its voltage 
        # where the threshold is wgt_fac+1 if capped TDE

        cpB = nx.CompartmentPrototype(
            vThMant=(wgt_fac+1)*2**exponent_fac if self.capped else 100,
            compartmentCurrentDecay=4096,
            compartmentVoltageDecay=int(1 / params['tau_fac'] * 2 ** 12)
        )

        # create compartment C that receives B's voltage
        cpC = nx.CompartmentPrototype(
            vThMant=1000,
            compartmentCurrentDecay=4096,
            compartmentVoltageDecay=int(1 / params['tau_trig'] * 2 ** 12),
            thresholdBehavior=nx.COMPARTMENT_THRESHOLD_MODE.NO_SPIKE_AND_PASS_V_LG_VTH_TO_PARENT
        )

        # create soma compartment D that receives C's voltage and fires when it exceeds its threshold
        cpD = nx.CompartmentPrototype(
            vThMant=100,
            compartmentCurrentDecay=4096,
            compartmentVoltageDecay=int(1 / params['tau_soma'] * 2 ** 12),
        )

        # build compartment tree for the TDE neuron
        cpC.addDendrites(prototypeA=cpA, prototypeB=cpB, joinOp=nx.COMPARTMENT_JOIN_OPERATION.PASS)
        cpD.addDendrite(prototype=[cpC], joinOp=nx.COMPARTMENT_JOIN_OPERATION.ADD)

        # create the TDE neuron prototype
        neuronPrototype = nx.NeuronPrototype(cpD)

        # create the TDE neurons group
        num_neurons = params['num_neurons']
        neurongroup = net.createNeuronGroup(prototype=neuronPrototype, size=num_neurons)

        # create the spike generator processes
        sgpA = net.createSpikeGenProcess(numPorts=num_neurons)
        sgpB = net.createSpikeGenProcess(numPorts=num_neurons)

        # create the connection prototypes
        # synapse with synaptic weight wgt_fac
        connProto = nx.ConnectionPrototype(weight=wgt_fac, weightExponent=exponent_fac)
        if self.capped:
            # synapse with synaptic weight wgt_fac+2 to reset B's voltage
            connProto_reset = nx.ConnectionPrototype(weight=wgt_fac+2, weightExponent=exponent_fac)
            # synapse with synaptic weight wgt_fac and delay 1 to integrate after the reset
            # TODO: quantify the impact of the one time step delay in the compartment B
            connProto_delay = nx.ConnectionPrototype(weight=wgt_fac, weightExponent=exponent_fac, delay=1, disableDelay=False, numDelayBits=3)
        
        # connect the spike generators to the TDE facilitator and trigger inputs
        sgpA.connect(neurongroup.dendrites[0].dendrites[1], prototype=connProto, connectionMask=sp.sparse.identity(num_neurons))
        if self.capped:
            sgpB.connect(neurongroup.dendrites[0].dendrites[0], prototype=connProto_reset, connectionMask=sp.sparse.identity(num_neurons))
            sgpB.connect(neurongroup.dendrites[0].dendrites[0], prototype=connProto_delay, connectionMask=sp.sparse.identity(num_neurons))
        else:
            sgpB.connect(neurongroup.dendrites[0].dendrites[0], prototype=connProto, connectionMask=sp.sparse.identity(num_neurons))
        
        spikegens = [sgpA, sgpB]

        # configure the probes
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
        
        # plot the results
        fig = plt.figure(1, figsize=(18, 10))

        for i, key in enumerate(np.sort(list(self.probes.keys()))):
            plt.subplot(int(num_probes/3),3, i + 1)
            self.probes[key].plot()
            # plt.plot(probes[key].data.T)
            plt.title(key)
            plt.xlim([0, num_steps])
        
        plt.tight_layout()