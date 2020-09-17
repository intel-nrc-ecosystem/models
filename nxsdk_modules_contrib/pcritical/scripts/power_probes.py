"""
This script computes power/time efficiency metrics with P-CRITICAL on Loihi.
"""

import os
import pickle
import numpy as np
import logging
import random
import torch
from quantities import ms, second, Hz
from modules.pcritical import NxPCritical
from ebdataset.audio import NTidigits


DATASET_PATH = os.environ["NTIDIGITS_DATASET_PATH"]


def get_topology():
    from modules.topologies import SmallWorldTopology

    return SmallWorldTopology(
        SmallWorldTopology.Configuration(
            minicolumn_shape=(4, 4, 4),
            macrocolumn_shape=(2, 2, 2),
            minicolumn_spacing=1460,
            p_max=0.11,
            spectral_radius_norm=False,
            intracolumnar_sparseness=635,
            neuron_spacing=40,
            inhibitory_init_weight_range=(0.1, 0.3),
            excitatory_init_weight_range=(0.2, 0.5),
        )
    )


def run_experiment():
    os.environ["PARTITION"] = "nahuku32"
    os.environ["BOARD"] = "ncl-ext-ghrd-01"
    logger = logging.getLogger(__name__)
    n_features = 64
    dt = 1 * ms

    def rec_array_to_spike_train(sparse_spike_train):
        """Transform a sparse numpy record array to a numpy dense array for processing"""
        ts = sparse_spike_train.ts * second
        ts = (ts.rescale(dt.units) / dt).magnitude
        duration = np.ceil(np.max(ts)) + 1
        spike_train = np.zeros((n_features, duration.astype(int)))
        spike_train[sparse_spike_train.addr, ts.astype(int)] = 1
        return spike_train

    train_set = NTidigits(
        DATASET_PATH,
        is_train=True,
        transforms=rec_array_to_spike_train,
        only_single_digits=True,
    )

    spiking_data = np.asarray([train_set[i] for i in range(len(train_set) // 4)])[
        :, 0
    ]  # Load a fourth of the training set in memory
    spike_freq = np.asarray([st.mean(axis=-1) for st in spiking_data]).mean() * (
        1 / dt
    )  # Average the spike count per sample, and over all samples/neurons
    logger.info("Using input frequency of %.3f Hz", spike_freq.rescale(Hz))

    nx_pcritical_configs = {
        "topology": get_topology(),
        "input_dim": n_features,
        "nb_of_conn_per_input": 2,
        "alpha": 2,
        "beta": 0.25,
        "tau_v": 30 * ms,
        "tau_i": 5 * ms,
        "v_th": 1.0,
        "refractory_period": 2 * ms,
        "dt": dt,
        "tau_v_pair": 5 * ms,
        "tau_i_pair": 1 * ms,
        "bin_size": 60 * ms,
        "pair_weight_mode": NxPCritical.PairWeightMode.HALF_VTH,
        "debug": False,
        "get_power_eff": True,
        "power_eff_input_freq": spike_freq,
    }

    model = NxPCritical(**nx_pcritical_configs)
    power_efficiency = model.power_efficiency_run(100000)

    print(power_efficiency)

    pickle.dump(power_efficiency, open("pcritical_power_eff.pkl", "wb"))


def main():
    # Set-up reproducibility
    seed = 0x1B
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="nx_ntidigits_power_probing.log",
    )

    run_experiment()


if __name__ == "__main__":
    main()
