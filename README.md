# DISCONTINUATION OF PROJECT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
INRC Ecosystem
==============

This repository will host models, modules, algorithms and applications developed by the INRC Community using nxsdk to run on the Intel Loihi Platform.

The repository is structured into directories. nxsdk_modules_ncl directory will be used to publish code/models which have been developed by maintainers of this repo. Researchers can publish and collaborate by publishing code/models within the nxsdk_modules_contrib directory. All code submissions will be reviewed. Please use the Github PR process. All contributed code will be distributed under the LICENSE provided with this repository.

To run the models, access to the Loihi software development kit (NxSDK) and the Loihi hardware is needed. 

For more information, please visit: https://www.intel.com/content/www/us/en/research/neuromorphic-community.html
For community support, Q&A and other information, please visit: http://neuromorphic.intel.com 

## Acknowledgments

* Pelenet : Carlo Michaelis, University of Goettingen (https://github.com/sagacitysite/pelenet)
  * Connection matrix and Connection asymmetry landscape Python packages in ``lib/anisotropic`` were contributed by Sebastian Spreizer (https://github.com/babsey/spatio-temporal-activity-sequence/tree/6d4ab597c98c01a2a9aa037834a0115faee62587)

* PCritical : Ismael Balafrej, prof. Jean Rouat. University of Sherbrooke. [NEuro COmputational & Intelligent Signal Processing Research Group (NECOTIS)](http://www.gel.usherbrooke.ca/necotis/)

* Time Difference Encoder : 
  * This is a Loihi implementation of the Time Difference Encoder (TDE) / spiking Elementary Motion Detector (sEMD). It converts a temporal difference between two spikes from different sources into a firing rate (number of spikes).

  * The TDE has been introduced and used by:
    - Milde, M. B., Bertrand, O. J., Ramachandran, H., Egelhaaf, M., & Chicca, E. (2018). Spiking elementary motion detector in neuromorphic systems. Neural computation, 30(9), 2384-2417.
    - D'Angelo, G., Janotte, E., Schoepe, T., O'Keeffe, J., Milde, M. B., Chicca, E., & Bartolozzi, C. (2020). Event-based eccentric motion detection exploiting time difference encoding. Frontiers in Neuroscience, 14, 451.

  * This file was started at the Telluride Neuromorphic Workshop 2019
  * Contributors:
    - Alpha Renner (alpren@ini.uzh.ch)
    - Lyes Khacef (l.khacef@rug.nl)
    - Elisabetta Chicca
    - Garrick Orchard
    - Andreas Wild
    - Mike Davies

  * Version 1.4
  * Updated for nxsdk version 1.0.0

* SpyTorch2Loihi :
  * SpyTorch2Loihi code is an extension of Slayer to import and deploy any trained (feedforward and recurrent) spiking neural network from SpyTorch into Loihi. The training and export code is available in "https://github.com/event-driven-robotics/tactile_braille_reading" as part of the work on "Braille Letter Reading: A Benchmark for Spatio-Temporal Pattern Recognition on Neuromorphic Hardware" (https://arxiv.org/abs/2205.15864).
  
  * Contributors:
    - Lyes Khacef (l.khacef@rug.nl)
    - Sumit Bam Shrestha