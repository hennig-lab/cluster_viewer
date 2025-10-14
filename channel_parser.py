import os.path
import json
import glob
import numpy as np
import scipy.io as sio

def load_spike_data(mat_path, nbins=50):
    """
    Loads a MATLAB struct containing spike waveforms and cluster_class info.
    Returns a list of dicts, one per neuron (excluding cluster 0 / noise).
    Each dict contains:
        - 'ISI_bins': np.array of log-scaled bin centers (ms)
        - 'ISI_freqs': np.array of normalized histogram frequencies
        - 'waveform_quintiles': np.array shape (10, 64) summarizing waveform distribution
    """
    # Load .mat file
    mat = sio.loadmat(mat_path, squeeze_me=True)
    waveforms = mat['spikes']      # shape (N, 64)
    cluster_class = mat['cluster_class']  # shape (N, 2)
    
    cluster_ids = cluster_class[:, 0].astype(int)
    spike_times = cluster_class[:, 1].astype(float)  # ms

    # Ignore noise cluster (0)
    unique_clusters = np.unique(cluster_ids)
    unique_clusters = unique_clusters[unique_clusters != 0]

    # Log-spaced bins from 1 ms to 10 s (10,000 ms)
    ISI_bins = np.logspace(0, 4, nbins)  # log-spaced bins

    neurons = []
    for cid in unique_clusters:
        neuron_mask = cluster_ids == cid
        neuron_times = np.sort(spike_times[neuron_mask])
        neuron_waveforms = waveforms[neuron_mask, :]

        # Compute ISIs
        ISIs = np.diff(neuron_times)
        if len(ISIs) == 0:
            ISI_freqs = np.zeros(len(ISI_bins) - 1)
        else:
            ISI_freqs, _ = np.histogram(ISIs, bins=ISI_bins, density=True)

        # Compute waveform quintiles (0%, 10%, ..., 100%)
        quintiles = np.percentile(neuron_waveforms, np.arange(0, 110, 10), axis=0)

        neurons.append({
            'cluster_id': int(cid),
            'ISI_bins': ISI_bins[:-1],  # bin edges (left)
            'ISI_freqs': ISI_freqs,
            'waveform_quintiles': quintiles
        })

    return neurons

def collect_neuron_data(directory, outfile, pattern="times_*.mat", nbins=50):
    """
    Finds all .mat files matching the pattern in the given directory,
    extracts neuron data using load_spike_data(), adds filename to each dict,
    and saves combined results to a JSON file.
    """
    search_path = os.path.join(directory, pattern)
    files = sorted(glob.glob(search_path))
    print(f"Found {len(files)} files matching pattern '{pattern}'.")
    all_neurons = []

    for fpath in files:
        print(f"Processing {os.path.basename(fpath)} ...")
        neurons = load_spike_data(fpath, nbins=nbins)
        for n in neurons:
            n['filename'] = os.path.basename(fpath)
            # Convert numpy arrays to lists for JSON compatibility
            n['ISI_bins'] = n['ISI_bins'].tolist()
            n['ISI_freqs'] = n['ISI_freqs'].tolist()
            n['waveform_quintiles'] = np.asarray(n['waveform_quintiles']).tolist()
        all_neurons.extend(neurons)

    # Save all results as a single JSON file
    with open(outfile, 'w') as f:
        json.dump(all_neurons, f, indent=2)

    print(f"Saved {len(all_neurons)} neuron entries to {outfile}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Collect neuron data from times_*.mat files into a JSON summary."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing times_*.mat files."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="times_*.mat",
        help="Filename pattern to match (default: 'times_*.mat')."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output JSON file path."
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=50,
        help="Number of log-spaced ISI bins between 1 ms and 10 s (default: 50)."
    )
    args = parser.parse_args()    
    outfile = args.outfile
    if outfile is None:
        outfile = os.path.join(args.directory, "neuron_data_summary.json")
    print(f"Searching '{args.directory}' for '{args.pattern}' ...")
    collect_neuron_data(args.directory, outfile, pattern=args.pattern, nbins=args.nbins)
    print(f"Done. Output saved to {outfile}")
