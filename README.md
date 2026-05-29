# Cluster Viewer

A browser-based GUI for quickly choosing which units to include/exclude based on spike waveforms and interspike interval (ISI) distributions.
- Before using this tool, you must have a directory containing "times_*.mat" files, each corresponding to a single channel, as output by [wave_clus](https://github.com/csn-le/wave_clus)
- Start the browser with `uv run python cluster_viewer.py --directory PATH_TO_YOUR_DIRECTORY`
- Click a unit to mark it for exclusion, or click it again if you change your mind.
- Excluded units are saved automatically to `PATH_TO_YOUR_DIRECTORY/cluster_viewer_results/clusters_excluded.csv`
- When you hit the `Export` button, a sparse matrix of all spike times from non-excluded units is written to `PATH_TO_YOUR_DIRECTORY/cluster_viewer_results/spikes.mat`

## Automatic cluster selection

To get a spike matrix of spike times without any manual intervention, run: `uv run python cluster_viewer.py --directory PATH_TO_YOUR_DIRECTORY --skip_manual`

This will export a spike matrix to `PATH_TO_YOUR_DIRECTORY/cluster_viewer_results/spikes_auto.mat`

# Installation

Once you have uv, just run `uv sync`

# Contributing

If you modify any part of this codebase, please run `uv run pytest -s` in a terminal and ensure ALL tests pass before pushing any changes.
