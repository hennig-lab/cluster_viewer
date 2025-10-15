# Unit Excluder

A browser-based GUI for quickly choosing which units to exclude based on spike waveforms and interspike interval (ISI) distributions.
- Before using this tool, you must have a directory containing "times_*.mat" files, each corresponding to a single channel, as output by [wave_clus](https://github.com/csn-le/wave_clus)
- Start the browser with `python unit_excluder.py --directory PATH_TO_YOUR_DIRECTORY`
- Click a unit to mark it for exclusion, or click it again if you change your mind.
- Excluded units are saved automatically to `PATH_TO_YOUR_DIRECTORY/clusters_excluded.csv`
- When you stop the server (Ctrl+C), a sparse matrix of all spike times from non-excluded units is written automatically to `PATH_TO_YOUR_DIRECTORY/_spikes.mat`

# Installation

Requirements: `numpy`, `scipy`, `flask`

Tested with: Python 3.9.13, `numpy==1.24.4`, `scipy==1.9.1`, `flask==1.1.2`, but other versions will likely work.
