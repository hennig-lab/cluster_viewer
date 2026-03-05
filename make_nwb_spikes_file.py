#!/usr/bin/env python3
"""
Create NWB file from spike data stored in *times.mat files.
"""

import os
import warnings
from datetime import datetime
from dateutil.tz import tzlocal

import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from pynwb.file import Subject

from make_spikes_matrix import make_spikes_matrix


def make_nwb_spikes_file(
    directory,
    outfile=None,
    ignoreClusters=False,
    includeClusterZero=False,
    ignoreForced=False,
    ignoreDuplicates=True,
    skipEmptyChannels=False,
    exclusionfile=None,
    sampling_rate=30000.0,
    session_description="Spike sorting session",
    session_id=None,
):
    """
    Convert *times.mat or *spikes.mat files to an NWB file with spike times.

    Parameters
    ----------
    directory : str
        Folder containing *times*.mat or *spikes*.mat files.
    outfile : str, optional
        File path to save output .nwb file.
    ignoreClusters : bool, optional
        If True, treat all spikes on a channel as one unit.
    includeClusterZero : bool, optional
        If True, include spikes even if cluster == 0.
    ignoreForced : bool, optional
        If True, ignore spikes marked as "forced".
    ignoreDuplicates : bool, optional
        If True, ignore duplicate spikes as detected by DER
    skipEmptyChannels : bool, optional
        If True, skip channels without spikes
    exclusionfile : str, optional
        If provided, path to a CSV file with (filename, cluster_id) pairs to exclude
    sampling_rate : float, optional
        Sampling rate in Hz (default: 30000.0)
    session_description : str, optional
        Description of the recording session
    session_id : str, optional
        Session identifier (defaults to directory name)

    Returns
    -------
    result : dict
        Dictionary with keys: chan, spikes, waveform, qual, cluster_ids, params
    """

    # First, get the spike data using the existing function
    print("Extracting spike data from .mat files...")
    result = make_spikes_matrix(
        directory=directory,
        outfile=None,  # Don't save to mat file
        ignoreClusters=ignoreClusters,
        includeClusterZero=includeClusterZero,
        ignoreForced=ignoreForced,
        ignoreDuplicates=ignoreDuplicates,
        skipEmptyChannels=skipEmptyChannels,
        exclusionfile=exclusionfile,
    )

    if result is None:
        warnings.warn("No spike data extracted. NWB file not created.")
        return None

    # Extract data from result
    chan_list = result['chan']
    cluster_ids = result['cluster_ids']
    spikes_matrix = result['spikes']
    waveforms = result['waveform']
    channel_file_names = result['channel_file_names']
    n_spikes_excluded = result['n_spikes_excluded']
    clusters_excluded = result['clusters_excluded']
    params = result['params']

    # Create NWB file
    if session_id is None:
        session_id = os.path.basename(os.path.normpath(directory))

    print(f"Creating NWB file for session: {session_id}")
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=session_id,
        session_start_time=datetime.now(tzlocal()),
        session_id=session_id,
    )

    # Add processing parameters as scratch data
    nwbfile.add_scratch(
        name="processing_parameters",
        data=str(params),
        description="Parameters used for spike extraction and processing"
    )

    # Create electrode table
    unique_channels = np.unique(chan_list)
    device = nwbfile.create_device(
        name="recording_device",
        description="Electrode array used for recording",
        manufacturer="Unknown"
    )

    electrode_group = nwbfile.create_electrode_group(
        name="electrodes",
        description="Electrode group",
        location="unknown",
        device=device
    )

    # Add electrodes and create mapping from channel ID to electrode table index
    # (channels may not be consecutive, so we need this mapping)
    chan_to_electrode_idx = {}
    for electrode_idx, chan_id in enumerate(unique_channels):
        nwbfile.add_electrode(
            id=int(chan_id),
            x=np.nan,
            y=np.nan,
            z=np.nan,
            imp=np.nan,
            location="unknown",
            filtering="unknown",
            group=electrode_group
        )
        chan_to_electrode_idx[int(chan_id)] = electrode_idx

    # Add units (spike times)
    print(f"Adding {len(chan_list)} units to NWB file...")
    nwbfile.add_unit_column(name="channel", description="Channel number")
    nwbfile.add_unit_column(name="cluster_id", description="Cluster ID")
    nwbfile.add_unit_column(name="waveform_mean", description="Mean waveform")
    nwbfile.add_unit_column(name="channel_file_name", description="Source file name")
    nwbfile.add_unit_column(name="n_spikes_excluded", description="Number of excluded spikes")

    for unit_idx in range(len(chan_list)):
        # Extract spike times for this unit (convert from sample indices to seconds)
        spike_indices = spikes_matrix[unit_idx, :].nonzero()[1]
        spike_times = (spike_indices + 1) / sampling_rate  # +1 because matrix is 0-indexed but times are 1-indexed

        # Get waveform
        waveform_mean = waveforms[unit_idx, :]

        # Add unit
        nwbfile.add_unit(
            spike_times=spike_times,
            channel=int(chan_list[unit_idx]),
            cluster_id=float(cluster_ids[unit_idx]),
            waveform_mean=waveform_mean,
            channel_file_name=channel_file_names[unit_idx],
            n_spikes_excluded=int(n_spikes_excluded[unit_idx]),
            electrodes=[chan_to_electrode_idx[int(chan_list[unit_idx])]]
        )

    # Add excluded clusters as scratch data
    if clusters_excluded:
        nwbfile.add_scratch(
            name="clusters_excluded",
            data=np.array(clusters_excluded, dtype=str),
            description="List of excluded clusters (filename, cluster_id pairs)"
        )

    # Save NWB file
    if outfile:
        print(f"Saving NWB file to {outfile}")
        with NWBHDF5IO(str(outfile), 'w') as io:
            io.write(nwbfile)
        print(f"Successfully saved NWB file to {outfile}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create NWB file from times.mat or spikes.mat files."
    )
    parser.add_argument(
        "--directory",
        required=True,
        help="Path to directory containing times.mat or spikes.mat files"
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Path to output .nwb file"
    )
    parser.add_argument(
        "--ignore_clusters",
        action="store_true",
        help="If set, treat all spikes on a channel as one unit"
    )
    parser.add_argument(
        "--include_cluster_zero",
        action="store_true",
        help="If set, include spikes even if cluster == 0"
    )
    parser.add_argument(
        "--ignore_forced",
        action="store_true",
        help="If set, ignore spikes marked as 'forced'"
    )
    parser.add_argument(
        "--ignore_duplicates",
        action="store_true",
        help="If set, ignore duplicate spikes"
    )
    parser.add_argument(
        "--skip_empty_channels",
        action="store_true",
        help="If set, skips channels without spikes (note this will affect channel indexing)"
    )
    parser.add_argument(
        "--exclusionfile",
        default=None,
        help="Path to CSV file with (filename, cluster_id) pairs to exclude"
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=30000.0,
        help="Sampling rate in Hz (default: 30000)"
    )
    parser.add_argument(
        "--session_description",
        default="Spike sorting session",
        help="Description of the recording session"
    )
    parser.add_argument(
        "--session_id",
        default=None,
        help="Session identifier (defaults to directory name)"
    )
    args = parser.parse_args()

    make_nwb_spikes_file(
        directory=args.directory,
        outfile=args.outfile,
        ignoreClusters=args.ignore_clusters,
        includeClusterZero=args.include_cluster_zero,
        ignoreForced=args.ignore_forced,
        ignoreDuplicates=args.ignore_duplicates,
        skipEmptyChannels=args.skip_empty_channels,
        exclusionfile=args.exclusionfile,
        sampling_rate=args.sampling_rate,
        session_description=args.session_description,
        session_id=args.session_id,
    )
