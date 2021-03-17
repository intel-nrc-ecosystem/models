"""
This script runs the NTIDIGITS* dataset through a P-CRITICAL enabled reservoir on Loihi.
Dataset path is assumed to be available in env. variable NTIDIGITS_DATASET_PATH.
See `python -m scripts.ntidigits --help` for all options.

* Anumula, Jithendar, et al. “Feature Representations for Neuromorphic Audio Spike Streams.”
  Frontiers in Neuroscience, vol. 12, Feb. 2018, p. 23. DOI.org (Crossref), doi:10.3389/fnins.2018.00023.
  Available for download at https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M
"""
import os
import random
import multiprocessing
import numpy as np
import logging
import networkx as netx

DATASET_PATH = os.environ["NTIDIGITS_DATASET_PATH"]
TOPOLOGY_CACHE = "pcritical_ntidigits_topology_cache.npy"


def get_topology():
    if os.path.exists(TOPOLOGY_CACHE):
        adj_matrix = np.load(TOPOLOGY_CACHE)
        return netx.from_numpy_matrix(adj_matrix, create_using=netx.DiGraph())

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


def set_topology(adj_matrix):
    np.save(TOPOLOGY_CACHE, adj_matrix)


def run_experiment(debug, seed, nb_epochs, partition, board, out_file):
    """Self-contained script to process a single n-tidigits batch on loihi"""
    _logger = logging.getLogger(__name__)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    os.environ["PARTITION"] = partition
    os.environ["BOARD"] = board

    import gc
    from modules.pcritical import NxPCritical
    from ebdataset.audio import NTidigits
    from quantities import ms, second
    from h5py import File
    from tqdm import trange, tqdm

    n_features = 64
    dt = 1 * ms
    max_duration = 2464

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
    test_set = NTidigits(
        DATASET_PATH,
        is_train=False,
        transforms=rec_array_to_spike_train,
        only_single_digits=True,
    )
    topology = get_topology()
    nb_of_neurons = topology.number_of_nodes()

    nx_pcritical_configs = {
        "topology": topology,
        "input_dim": n_features,
        "nb_of_conn_per_input": 2,
        "alpha": 2,
        "beta": 0.25,
        "tau_v": 30 * ms,
        "tau_i": 5 * ms,
        "v_th": 1.0,
        "refractory_period": 1 * ms,
        "dt": dt,
        "tau_v_pair": 5 * ms,
        "tau_i_pair": 1 * ms,
        "bin_size": 60 * ms,
        "pair_weight_mode": NxPCritical.PairWeightMode.BIN_SIZE_SYNC,
        "debug": debug,
        "get_power_eff": False,
    }

    batch_size = 50  # Process up to batch_size samples in a single run
    if debug:
        batch_size = 10
    nb_of_bins = max_duration // 60 + 1

    def process_set(name, data_set, shuffle):
        with File(out_file, "a") as f_hndl:
            # Set-up h5 file output
            length = len(data_set)
            f_hndl.create_dataset(
                name, shape=(length, nb_of_neurons, nb_of_bins), dtype=np.int16
            )
            f_hndl.create_dataset(name + "_labels", shape=(length,), dtype=np.uint8)

        indices = np.random.permutation(length) if shuffle else np.arange(length)
        batches = np.array_split(
            indices, np.ceil(len(indices) / batch_size).astype(int)
        )
        for batch_indices in tqdm(batches, desc=name):
            gc.collect()
            samples = [data_set[i] for i in batch_indices]  # Load samples in memory
            spike_trains = [s[0] for s in samples]
            labels = np.asarray(
                [int(s[1].replace("z", "0").replace("o", "10")) for s in samples]
            )

            def process_batch(nx_pcritical_configs, spike_trains, nb_of_bins):
                nx_pcritical_configs["topology"] = get_topology()
                with File(out_file, "a") as f_hndl, NxPCritical(
                    **nx_pcritical_configs
                ) as model:
                    out_hndl = f_hndl[name]
                    labels_out_hndl = f_hndl[name + "_labels"]
                    bins = model(spike_trains, nb_of_bins)
                    set_topology(model.adj_matrix())

                    if shuffle:  # Sort indices for storage with h5py
                        sorted_aindices = batch_indices.argsort()
                        out_hndl[
                            batch_indices[sorted_aindices], :, : bins.shape[-1]
                        ] = bins[sorted_aindices]
                        labels_out_hndl[batch_indices[sorted_aindices]] = labels[
                            sorted_aindices
                        ]
                    else:
                        out_hndl[batch_indices, :, : bins.shape[-1]] = bins
                        labels_out_hndl[batch_indices] = labels

            p = multiprocessing.Process(
                target=process_batch,
                args=(nx_pcritical_configs, spike_trains, nb_of_bins),
            )
            p.start()
            p.join()
            if p.exitcode != 0:
                _logger.fatal("Error processing batch")
                exit(1)

    for epoch in trange(nb_epochs, desc="epoch"):
        process_set("train_%i" % epoch, train_set, True)

    process_set("test", test_set, False)


def main(
    debug=False,
    seed=0x1B,
    nb_of_epochs=10,
    partition="loihi_2h",
    board="ncl-ext-ghrd-04",
    out_file="ntidigits-nxpcritical-post-reservoir.h5",
):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="nx_ntidigits.log",
    )

    _logger = logging.getLogger(__name__)

    if os.path.exists(out_file):
        if input("File %s already exists, delete ? [Y/n]" % out_file).lower() != "n":
            os.remove(out_file)

    run_experiment(debug, seed, nb_of_epochs, partition, board, out_file)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
