# Channel Browser

A browser-based GUI for quickly choosing which units to keep or remove based on spike waveforms and interspike interval (ISI) distributions.
- Before using this tool, you must have a directory containing "times_*.mat" files, each corresponding to a single channel, as output by [wave_clus](https://github.com/csn-le/wave_clus)
- Start the browser with `python channel_browser.py --directory PATH_TO_YOUR_DIRECTORY`
- Click a channel to mark it for removal, or click it again if you change your mind.
- Results are saved automatically as a csv file at `PATH_TO_YOUR_DIRECTORY/clusters_excluded.csv` which lists all units (filename, cluster_id) you have marked to exclude

# Installation

Requirements: `numpy`, `scipy`, `flask`

Tested with: Python 3.9.13, `numpy==1.24.4`, `scipy==1.9.1`, `flask==1.1.2`, but other versions will likely work.
