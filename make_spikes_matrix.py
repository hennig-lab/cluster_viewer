#%%

import os
import re
import warnings
from datetime import datetime

import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import lil_matrix
from scipy.stats import scoreatpercentile

def make_spikes_matrix(directory, outfile=None, ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=True, exclusionfile=None):
    """
    Convert *times.mat or *spikes.mat files to a sparse spike matrix (0s and 1s).

    Parameters
    ----------
    directory : str
        Folder containing *times*.mat or *spikes*.mat files.
    outfile : str, optional
        File path to save output .mat file.
    ignoreClusters : bool, optional
        If True, treat all spikes on a channel as one unit.
    includeClusterZero : bool, optional
        If True, include spikes even if cluster == 0.
    ignoreForced : bool, optional
        If True, ignore spikes marked as "forced".
    ignoreDuplicates : bool, optional
        If True, ignore duplicate spikes as detected by DER
    exclusionfile : str, optional
        If provided, path to a CSV file with (filename, cluster_id) pairs to exclude

    Returns
    -------
    result : dict
        Dictionary with keys: chan, spikes, waveform, qual, cluster_ids, params
    """

    if ignoreForced and ignoreClusters and includeClusterZero:
        raise ValueError("Cannot set ignoreForced=True with includeClusterZero && ignoreClusters.")

    # get all .mat files in directory
    if ignoreClusters and includeClusterZero:
        allFiles = []
        if not exclusionfile: # if no exclusionfile, look for spikes files first
            allFiles = [f for f in os.listdir(directory) if f.endswith('.mat') and 'spikes' in f]
        if not allFiles: # if we still have no files, look for times files
            allFiles = [f for f in os.listdir(directory) if f.endswith('.mat') and 'times' in f]
    else:
        allFiles = [f for f in os.listdir(directory) if f.endswith('.mat') and 'times' in f]

    if not allFiles:
        warnings.warn(f"No *times*.mat files found in {directory}")
        return None

    if exclusionfile:
        if not os.path.exists(exclusionfile):
            raise FileNotFoundError(f"Excluded file {exclusionfile} not found.")
        excluded = set()
        # read excluded (filename, cluster_id) pairs from file
        with open(exclusionfile, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    # e.g., "times_mRF3C02_2342.mat,1"
                    row = (parts[0].strip(), int(parts[1].strip()))
                    if row[0] not in allFiles:
                        raise Exception(f"Excluded file {row[0]} is mentioned in exclusionfile but not found in directory.")
                    excluded.add(row)
    else:
        excluded = set()

    # Extract channel numbers from filenames
    chan_inds = []
    for fname in allFiles:
        parts = fname.replace('.mat', '').split('_')
        if parts[0] == 'times':
            # e.g. 'times_mRF3C02_2342.mat'
            chan_inds.append(int(parts[2]))
        elif len(parts) >= 3 and parts[2] == 'spikes':
            # e.g. 'mRF3C02_2342_spikes.mat'
            chan_inds.append(int(parts[1]))
        else:
            raise ValueError(f"File name format not recognized: {fname}")
    chan_inds = np.array(chan_inds, dtype=int)
    print(f"Found {len(chan_inds)} channels. Creating spike matrix with {ignoreClusters=}, {includeClusterZero=}, {ignoreForced=}, and {len(excluded)} excluded clusters.")

    if (chan_inds.min() != 1) or (len(np.unique(chan_inds)) != chan_inds.max()):
        print(f"Will reindex channels by subtracting the smallest ({chan_inds.min()})")
    chan_inds = chan_inds - chan_inds.min() + 1

    # Initialize containers
    u = 0  # unit counter
    spikes = None
    waveform_list = []
    chan_list = []
    cluster_ids_list = []
    channel_file_names = []
    n_spikes_excluded = []

    # Loop through files
    for c, fname in enumerate(allFiles):
        full_path = os.path.join(directory, fname)
        data = loadmat(full_path, squeeze_me=True)

        if ignoreForced and 'forced' not in data:
            warnings.warn(f"ignoreForced=True but {fname} does not include 'forced' field.")
        if ignoreDuplicates and 'detectionLabel' not in data:
            warnings.warn(f"ignoreDuplicates=True but {fname} does not include 'detectionLabel' field.")

        # Extract spike data
        cell_waveforms = data.get('spikes', np.array([]))
        detection_label = data.get('detectionLabel', np.ones(cell_waveforms.shape[0], dtype=bool)) if ignoreDuplicates else None
        if 'cluster_class' in data:
            cluster_class = np.array(data['cluster_class'][:, 0], dtype=float)
            spike_times = np.array(data['cluster_class'][:, 1], dtype=float)
        else:
            spike_times = np.array(data.get('index', []), dtype=float)
            cluster_class = np.ones_like(spike_times)

        # Exclude specified (filename, cluster_id) pairs
        if excluded:
            assert 'cluster_class' in data, f"Cannot exclude clusters for file {fname} without 'cluster_class' field."
            clusters_to_ignore = [ec for ef, ec in excluded if ef == fname]
            # if any excluded clusters are not actually present, we have a problem
            cluster_classes = np.unique(cluster_class)
            if any(ec not in cluster_classes for ec in clusters_to_ignore):
                raise Exception(f"Excluded clusters {clusters_to_ignore} not all found in file {fname}.")
            for ec in clusters_to_ignore:
                # mark excluded clusters as -1
                cluster_class[cluster_class == ec] = -1
        
        # Adjust cluster labels
        if ignoreClusters:
            if includeClusterZero:
                cluster_class[cluster_class >= 0] = 1
            else:
                cluster_class[cluster_class > 0] = 1

        # Initialize spike matrix if first file
        if spikes is None:
            mxTime = int(np.ceil(spike_times.max())) if spike_times.size else 1
            spikes = lil_matrix((len(allFiles) * 10, mxTime*3), dtype=bool)  # overallocate a bit

        # Identify unique clusters
        cluster_classes = np.unique(cluster_class)
        if includeClusterZero:
            cluster_classes = cluster_classes[cluster_classes >= 0]
        else:
            cluster_classes = cluster_classes[cluster_classes > 0]

        if cluster_classes.size > 0:
            for cluster_id in cluster_classes:
                ixc = cluster_class == cluster_id
                if ignoreForced and 'forced' in data:
                    forced = np.array(data['forced']).flatten()
                    if forced.size == len(cluster_class):
                        ixc = ixc & (forced == 0)
                if ignoreDuplicates:
                    cur_spikes_excluded = np.sum((detection_label[ixc] != 1))
                    ixc = ixc & (detection_label == 1) # Apply detection label mask
                else:
                    cur_spikes_excluded = 0

                spike_inds = np.ceil(spike_times[ixc]).astype(int)
                spike_inds = spike_inds[(spike_inds > 0) & (spike_inds <= spikes.shape[1])]
                spikes[u, spike_inds-1] = True

                if cell_waveforms.size > 0 and np.any(ixc):
                    waveform = np.mean(cell_waveforms[ixc, :], axis=0)
                else:
                    waveform = np.full((cell_waveforms.shape[1] if cell_waveforms.ndim > 1 else 1,), np.nan)

                waveform_list.append(waveform)
                chan_list.append(chan_inds[c])
                cluster_ids_list.append(cluster_id)
                channel_file_names.append(fname)
                n_spikes_excluded.append(cur_spikes_excluded)
                u += 1
        else:
            print(f"WARNING: No spikes found on channel {chan_inds[c]}, so will skip this channel.")
            # chan_list.append(chan_inds[c])
            # cluster_ids_list.append(np.nan)
            # channel_file_names.append(fname)
            # waveform_list.append(np.full((cell_waveforms.shape[1] if cell_waveforms.ndim > 1 else 1,), np.nan))
            # n_spikes_excluded.append(0)
            # u += 1

    # Trim spike matrix
    spikes = spikes[:u, :]
    spikes = spikes[:, :int(spikes.nonzero()[1].max()) + 1]

    # Compute waveform peak differences
    waveform = np.vstack(waveform_list)

    # resort chan, spikes, waveforms by channel number
    chan_list = np.array(chan_list)
    sort_order = np.argsort(chan_list)
    chan_list = chan_list[sort_order]
    cluster_ids_list = np.array(cluster_ids_list)[sort_order]
    waveform = waveform[sort_order, :]
    spikes = spikes[sort_order, :]
    channel_file_names = [channel_file_names[i] for i in sort_order]
    n_spikes_excluded = np.array(n_spikes_excluded)[sort_order]

    # Construct params struct
    params = {
        'directory': directory,
        'ignoreClusters': ignoreClusters,
        'includeClusterZero': includeClusterZero,
        'ignoreForced': ignoreForced,
        'ignoreDuplicates': ignoreDuplicates,
        'timeNow': datetime.now().isoformat()
    }

    # Construct result
    result = dict(
        chan=chan_list,
        cluster_ids=np.array(cluster_ids_list),
        params=params,
        spikes=spikes.astype(bool).tocsc(),
        waveform=waveform,
        channel_file_names=channel_file_names,
        n_spikes_excluded=n_spikes_excluded,
        clusters_excluded=list([','.join([str(x) for x in parts]) for parts in excluded])
    )

    # Save output if requested
    if outfile:
        print(str(outfile))
        savemat(str(outfile), result, do_compression=True)
        print(f"Saved spike matrix to {outfile}")

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create sparse spike matrix from times.mat or spikes.mat files.")
    parser.add_argument("--directory", required=True, help="Path to directory containing times.mat or spikes.mat files")
    parser.add_argument("--outfile", default=None, help="Path to output .mat file")
    parser.add_argument("--ignore_clusters", action="store_true", help="If set, treat all spikes on a channel as one unit")
    parser.add_argument("--include_cluster_zero", action="store_true", help="If set, include spikes even if cluster == 0")
    parser.add_argument("--ignore_forced", action="store_true", help="If set, ignore spikes marked as 'forced'")
    parser.add_argument("--ignore_duplicates", action="store_true", help="If set, ignore duplicate spikes")
    parser.add_argument("--exclusionfile", default=None, help="Path to CSV file with (filename, cluster_id) pairs to exclude")
    args = parser.parse_args()

    make_spikes_matrix(
        args.directory,
        outfile=args.outfile,
        ignoreClusters=args.ignore_clusters,
        includeClusterZero=args.include_cluster_zero,
        ignoreForced=args.ignore_forced,
        ignoreDuplicates=args.ignore_duplicates,
        exclusionfile=args.exclusionfile
    )
