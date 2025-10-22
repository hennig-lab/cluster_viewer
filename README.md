# Cluster Viewer

A browser-based GUI for quickly choosing which units to include/exclude based on spike waveforms and interspike interval (ISI) distributions.
- Before using this tool, you must have a directory containing "times_*.mat" files, each corresponding to a single channel, as output by [wave_clus](https://github.com/csn-le/wave_clus)
- Start the browser with `python cluster_viewer.py --directory PATH_TO_YOUR_DIRECTORY`
- Click a unit to mark it for exclusion, or click it again if you change your mind.
- Excluded units are saved automatically to `PATH_TO_YOUR_DIRECTORY/clusters_excluded.csv`
- When you stop the server (Ctrl+C), a sparse matrix of all spike times from non-excluded units is written automatically to `PATH_TO_YOUR_DIRECTORY/_spikes.mat`

# Installation

Requirements: `numpy`, `scipy`, `flask`

Tested with: Python 3.9.13, `numpy==1.24.4`, `scipy==1.9.1`, `flask==1.1.2`, but other versions will likely work.

# Contributing

If you modify any part of this codebase, please install the python package `pytest` and then run `pytest -s` in a terminal to ensure all tests pass.
