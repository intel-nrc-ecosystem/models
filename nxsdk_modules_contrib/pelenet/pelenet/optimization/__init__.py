import numpy as np
import matplotlib.pyplot as plt

from innnet import experiment as exp
from innnet.parameters import Parameters
import utils

class Optimization():

    """
    @desc: Initiates the optimization object
    """
    def __init__(self, seedFactor=0):
        self.p = Parameters(includeDerived = False)  # Parameters
        self.parameterConfigs = []  # Parameter configurations
        self.results = []  # Results (array of dictionaries)
        self.seedFactor = seedFactor  # Seed feactor coming from ensemble

        # Get parameter to optimize and related parameter values
        toOptimize = self.p.optimizeParameters
        self.parameterValues = getattr(self.p, toOptimize)

        # Generate datalog folder name
        dataLogDir = '2019---_optimize'

        # Create several parameters configurations, based on the global parameter settings
        for par in self.parameterValues:
            # Get another object instans of parameters
            parTmp = Parameters(includeDerived = False)
            
            # Reset value according to current parameter value
            setattr(parTmp, toOptimize, par)

            # Redefine datalog path
            newDataLogPath = os.path.join(getattr(parTmp, dataLogPath), dataLogDir)
            setattr(parTmp, dataLogPath, newDataLogPath)

            # Compute derived parameters
            parTmp.computeDerivedParameters()

            # Add parameter configuration to list
            self.parameterConfigs.append(parTmp)
    
    def run(self):
        # Run the experiment for every parameter configuration
        for i, par in enumerate(self.parameterConfigs): self.step(par, i)
        
    def step(self, parameters, idx):
        print('# Optimization {} (parameter value: {})'.format(idx, self.parameterValues[idx]))
        seedFactor = idx*10000 + self.seedFactor

        # Instantiate experiment
        inExp = exp.Experiment(parameters = parameters)

        # Start training
        inExp.trainOutput(log = False, seedFactor=seedFactor)

        # Start testing
        inExp.testNetwork(log = False, seedFactor=seedFactor)

        # Store result
        mse = np.mean(np.square(inExp.targetFunction - inExp.estimatedOutput))
        result = { 'mse': mse, 'parameters': parameters }
        self.results.append(result)

        # Delete object
        del inExp

    def plotResult(self):
        x = self.parameterValues
        y = np.array([ r['mse'] for r in self.results ])

        # Inverse, in order to get performance instead of error
        y = 1 - y/np.max(y)

        plt.figure(figsize=(16, 4))
        plt.xlabel('Network size')
        plt.ylabel('Performance')
        p = plt.plot(x, y)
