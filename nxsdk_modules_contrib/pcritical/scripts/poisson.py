"""
This script runs random poisson input through a P-CRITICAL enabled reservoir.
"""

import numpy as np
from quantities import ms
from modules.topologies import SmallWorldTopology
from modules.pcritical import NxPCritical


def main(seed=0x1B):
    np.random.seed(seed)

    topology = SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=(4, 4, 4),
            macrocolumn_shape=(2, 2, 2),
            inhibitory_prob=0.2,
            minicolumn_spacing=1460,
            p_max=0.11,
            spectral_radius_norm=False,
            intracolumnar_sparseness=635,
            neuron_spacing=40,
            inhibitory_init_weight_range=(0.1, 0.3),
            excitatory_init_weight_range=(0.2, 0.5),
        )
    )

    freqs = [10, 15, 30, 50]  # in Hz
    duration = 4080 * 2
    np.save("freqs.npy", freqs)
    input_dim = topology.number_of_nodes() // 3

    with NxPCritical(
        topology,
        input_dim=input_dim,
        tau_v=60 * ms,
        tau_i=2 * ms,
        tau_v_pair=1 * ms,
        tau_i_pair=1 * ms,
        debug=False,
    ) as model:
        in_spikes = [
            np.random.poisson(lam=freq / 1000.0, size=(input_dim, duration)).clip(0, 1)
            for i, freq in enumerate(freqs)
        ]
        in_spikes = np.asarray(in_spikes)

        bins = model(spike_train=in_spikes)

        weights = model.read_weights()

    np.save("weights.npy", weights)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
