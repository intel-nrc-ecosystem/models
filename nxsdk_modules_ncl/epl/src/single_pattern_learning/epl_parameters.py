# Copyright(c) 2019-2020 Intel Corporation All rights reserved.
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
# pylint: disable-all

from collections import namedtuple


def positiveInt(val):
    """validates if the value passed is a positive integer or not"""
    if val < 1 or type(val) != int:
        raise ValueError("value passed must be a positive integer")
    else:
        return val


def prob(val):
    """ validates the range of (connection) probability"""
    if not (type(val) == float and 0.0 <= val <= 1.0):
        raise ValueError("0.0 <= val <= 1.0 is the correct range")
    else:
        return val


def randSeed(val):
    """ validates the range of random seed"""
    if type(val) == int and 0 <= val < 2**32:
        return val
    else:
        raise ValueError("0 <= random seed < 2^32 is the correct range")


def binary(val):
    """ validates if the value passed is binary (true/false)"""
    if type(val) == bool:
        return val
    else:
        raise ValueError("random seed is a boolean flag")


# Each parameter is a named tuple of (parameter name, default value and its
# associated validator
Parameter = namedtuple('Parameter', ['name', 'default', 'validator'])
# list of parameters
parameters = [
    # for network setup
    Parameter(name='numPatterns', default=1, validator=positiveInt),
    Parameter(name='numColumns', default=72, validator=positiveInt),
    Parameter(name='numMCsPerColumn', default=1, validator=positiveInt),
    Parameter(name='numGCsPerColumn', default=3, validator=positiveInt),
    Parameter(name='numDelaysMCToGC', default=1, validator=positiveInt),
    Parameter(name='minDelaysMCToGC', default=16, validator=positiveInt),
    Parameter(name='gammaCycleDuration', default=40, validator=positiveInt),
    Parameter(name='connProbMCToGC', default=0.2, validator=prob),
    Parameter(name='useRandomSeed', default=False, validator=binary),
    Parameter(name='randomGenSeed', default=0, validator=randSeed),
    # network operation
    Parameter(name='numGammaCyclesTrain', default=45, validator=positiveInt),
    Parameter(name='numGammaCyclesTest', default=5, validator=positiveInt),
    Parameter(name='numGammaCyclesIdle', default=5, validator=positiveInt),
    Parameter(name='useLMTSpikeCounters', default=False, validator=binary),
    # logistics
    Parameter(name='logSNIPs', default=False, validator=binary),
    Parameter(name='executionTimeProbe', default=False, validator=binary)
]

validators = {par.name: par.validator for par in parameters}
defaults = {par.name: par.default for par in parameters}


class ParamsEPLSlots:
    """ Defines the various parameters of the EPL network. Defined as slots
    for ease of use"""
    __slots__ = ['numPatterns', 'numColumns', 'numMCsPerColumn',
                 'numGCsPerColumn', 'numDelaysMCToGC', 'minDelaysMCToGC',
                 'gammaCycleDuration', 'connProbMCToGC', 'useRandomSeed',
                 'randomGenSeed', 'numGammaCyclesTrain',
                 'numGammaCyclesTest', 'numGammaCyclesIdle',
                 'useLMTSpikeCounters', 'logSNIPs', 'executionTimeProbe']

    def __init__(self):
        pass


class ParamemtersForEPL(ParamsEPLSlots):
    """ Sets the default values for the parameters to EPL network. The values
    of the parameter are validated every time they are set.

    :param int numPatterns: number of patterns to be learned by the EPL \
    network
    :param int numColumns: number of columns in the network
    :param int numMCsPerColumn: number of MCs per column
    :param int numGCsPerColumn: number of GCs per column
    :param int numDelaysMCToGC: number of different MC->GC connnection delays
    :param int minDelaysMCToGC: value of the smallest MC->GC delay
    :param int gammaCycleDuration: number of algorithmic timesteps in a \
    gamma cycle
    :param float connProbMCToGC: MC->GC connection probability
    :param bool useRandomSeed: used to set if we want to use a random seed
    :param int randomGenSeed: the value of random seed
    :param int numGammaCyclesTrain: number of gamma cycles to train the network
    :param int numGammaCyclesTest: number of gamma cycles to test the network
    :param int numGammaCyclesIdle: number of gamma cycles when no inputs are \
    presented
    :param bool useLMTSpikeCounters: set to true if we want to use SNIP based \
    spike counter instead of spike probes
    :param bool logSNIPs: used to turn SNIP logging on and off
    :param bool executionTimeProbe: set to enable/disable execution time probe

    """

    def __init__(self):
        super().__init__()
        for par in parameters:
            setattr(self, par.name, par.default)

    def __setattr__(self, key, value):
        """ Ensures that only the parameters defined in slots are set"""
        if key not in self.__slots__:
            raise \
                AttributeError(
                    """Can't create new attribute '{}'""".format(key))
        validator = validators[key]
        value = validator(value)
        ParamsEPLSlots.__dict__[key].__set__(self, value)

    def showDefaults(self):
        """ lists the default values for the various parameters"""
        for attr in self.__slots__:
            print("{}={}".format(attr, getattr(self, attr)))


if __name__ == '__main__':
    print([par.name for par in parameters])
    par = ParamemtersForEPL()
    par.showDefaults()
