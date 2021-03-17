# NxP-CRITICAL
P-CRITICAL is a plasticity rule for reservoir computing running on Loihi.

This module is part of the P-CRITICAL library. This repository contains Loihi specific scripts. See P-CRITICAL repository at <https://github.com/NECOTIS/PCRITICAL> for PyTorch examples and analysis scripts.

A preprint for P-CRITICAL is also available at <https://arxiv.org/pdf/2009.05593.pdf>.

# Usage

This module was tested with the NxSDK v0.9.5.

```python
import numpy as np
from modules.pcritical import NxPCritical
from networkx import DiGraph
from quantities import ms

# Reservoir can be constructed using networkx's Directed Graphs
topology = DiGraph()
topology.add_nodes_from(range(32))

# Weights are floating points from [0, 1[, they will be automatically scaled to [0, 256[
for i in range(31):
    topology.add_edge(i, i+1, weight=np.random.random())

configs = { # See pcritical.py for complete parameters list
    "topology": topology,
    "input_dim": 2,
    "tau_v": 30 * ms,
    "dt": 1 * ms,
    "bin_size": 60 * ms,
}

with NxPCritical(**configs) as model:
    batch_size = 2  # P-CRITICAL can take batches and process them in a continuous Loihi simulation
    duration = 120
    input_spike_train = np.random.poisson(
        lam=0.5, size=(batch_size, configs["input_dim"], duration)
    ).clip(0, 1)
    output_bins = model(input_spike_train, 2)  # Output spikes are automatically time-binned
    print(output_bins.shape)  # (2, 32, 2)
```

Various experiments using NxPCritical are available under the `script` directory.

# Acknowledgements

© Copyright (September 2020) Ismael Balafrej, prof. Jean Rouat. University of Sherbrooke. [NEuro COmputational & Intelligent Signal Processing Research Group (NECOTIS)](http://www.gel.usherbrooke.ca/necotis/)
