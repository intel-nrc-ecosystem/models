# PeleNet - Reservoir computing framework for Loihi

Pele is the goddess of volcanoes and fire in the Hawaiian religion. She has therefore also the control over Loihi.

The PeleNet is a framework for reservoir computing. It has the following features:

* Defining experiments, which can contain one or multiple network simulations
* Selecting and preprocessing of probes (including readout)
* Extensive management of parameters (default and specific for an experiment)
* Connectvity matrix abstraction (connectivity matrix is automatically splitted and ditributed over the Loihi cores)
* Logging of experiments
* Object-oriented code design

## Warning

The framework is still under development. Some functionality may not work as described. For questions please contact me (see below).

## Using the framework

### Prerequisites

The framework is based on the NxSDK from Intel in version 0.9.x.

This in turn requires currently exactly Python 3.5.2.

Further packages like numpy, scipy, etc. are also required. Some of them are already installed as a dependency of the NxSDK. Some may need to be installed manually when facing errors of missing libraries.

### Necessary folder

After cloning the code, it is necessary to create a "log" folder in the main directory. In this folder a new folder with a time stamp is created each time an experiment runs. The new folder contains information about the chosen parameters and stores some basic plots and/or data (if set so). This helps to reproduce older results. The log folder is excluded from commiting.

### Optional folder

Optionally a "data" and a "plots" folder can be created to store plots or data for later use, like data evaluation or publication. These folders are excluded from commiting.

### How to use it

The main entry point for the PeleNet framework are the jupyter notebooks in the main folder. Here some experiments and evaulation scripts were created for my research. Just use them as prototypes.

## Code structure

Note: the code is stricly object-oriented and modulized. If you add a function, you mostly also have to register it in the ``__ini__.py`` file of the module.

### Libs

The ``lib`` folder contains external libraries and some general helper functions.

The code in ``lib/anisotropic`` is taken from https://github.com/babsey/spatio-temporal-activity-sequence/tree/6d4ab597c98c01a2a9aa037834a0115faee62587

### Main folder

Contains the jupyter notebooks. Here starts your work :)

### Pelenet

Contains the framework code.

#### Experiments

Defines experiments. If a new experiment is intended, a new experiment class has to be created here.

#### Network

Provides many functions for defining a network, like

* Define input
* Connect neurons
* Add noise
* Define output
* Set probes
* Add snips (see below)
* Define/Draw weight matrices

#### Optimization

Not yet available. Intension: Run several experiments to optimize parameters automatically.

#### Parameters

Default parameters are defined here.

Note that parameters can also be defined or can be overwritten in the experiment files. So changing things here is not always necessary or even without effect.

#### Plots

Adds a plot object to an experiment.

If you want to add a plot function, which you regularly want to use, you can do it here.

#### Snips

Contains C scripts which run on the x86 cores of the Loihi chip.

#### System

Constains some basic stuff, mainly about the logging system.

#### Utils

The utils module is a singleton. It is independent from the experiments and can be called anywhere, also e.g. in evaluation scripts where only stored data is processed.

## Questions

If you have any questions, please just contact me via carlo.michaelis @ gmail.com
