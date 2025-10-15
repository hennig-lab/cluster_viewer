#!/usr/bin/env python3
import os
import csv
import json
import webbrowser
from flask import Flask, jsonify, request, send_from_directory
from channel_parser import collect_neuron_data
from make_spikes_matrix import make_spikes_matrix

app = Flask(__name__, static_folder="static")

DATA_FILE = "neuron_data.json"
EXCLUDE_FILE = "clusters_excluded.csv"

# ---------------------------
# Helpers
# ---------------------------

def load_neurons():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def load_exclusions():
    excluded = set()
    if not os.path.exists(EXCLUDE_FILE):
        save_exclusions(excluded)
        return excluded
    with open(EXCLUDE_FILE, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                excluded.add((row[0], int(row[1])))
    return excluded

def save_exclusions(excluded):
    with open(EXCLUDE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        for fn, cid in sorted(excluded):
            writer.writerow([fn, cid])

# ---------------------------
# Routes
# ---------------------------

@app.route("/api/neurons")
def api_neurons():
    neurons = load_neurons()
    print(f"Loaded {len(neurons)} neurons from {DATA_FILE}")
    excluded = load_exclusions()
    for n in neurons:
        n["excluded"] = (n["filename"], n["cluster_id"]) in excluded
    return jsonify(neurons)

@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    data = request.json
    filename = data.get("filename")
    cluster_id = int(data.get("cluster_id"))
    excluded = load_exclusions()
    key = (filename, cluster_id)
    if key in excluded:
        excluded.remove(key)
    else:
        excluded.add(key)
    save_exclusions(excluded)
    return jsonify({"status": "ok", "excluded": list(excluded)})

@app.route("/")
def root():
    return send_from_directory("static", "index.html")

# ---------------------------
# Launch server
# ---------------------------

if __name__ == "__main__":
    import argparse
    from threading import Timer

    parser = argparse.ArgumentParser(description="Local viewer for neuron data.")
    parser.add_argument("--directory", default=None, help="Path to directory containing times.mat files")
    parser.add_argument("--jsonfile", default=None, help="Path to neuron JSON file")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default 5000)")
    parser.add_argument("--nbins", type=int, default=50, help="Number of bins (default 50)")
    parser.add_argument("--pattern", default="times_*.mat", help="Filename pattern to match (default: 'times_*.mat')")
    parser.add_argument("--csvfile", default=None, help="Path to CSV exclusion file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.directory:
        if not os.path.isdir(args.directory):
            raise NotADirectoryError(f"{args.directory} is not a valid directory")
        DATA_FILE = args.jsonfile if args.jsonfile else os.path.join(args.directory, "neuron_data.json")
        collect_neuron_data(args.directory, DATA_FILE, pattern=args.pattern, nbins=args.nbins, verbose=args.verbose)
    else:
        if not args.jsonfile:
            raise ValueError("Must provide --directory or --jsonfile")
        DATA_FILE = args.jsonfile
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find {DATA_FILE}")
    EXCLUDE_FILE = args.csvfile
    if EXCLUDE_FILE is None:
        EXCLUDE_FILE = os.path.join(os.path.dirname(DATA_FILE), "clusters_excluded.csv")

    url = f"http://127.0.0.1:{args.port}"
    print(f"Starting server at {url}")
    Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=False, port=args.port)

    print('Server stopped. Creating spike matrix files...')
    make_spikes_matrix(args.directory, outfile=os.path.join(args.directory, "_spikes.mat"), ignoreClusters=False, includeClusterZero=False, ignoreForced=False, exclusionfile=EXCLUDE_FILE)
    make_spikes_matrix(args.directory, outfile=os.path.join(args.directory, "_spikes_perChannel.mat"), ignoreClusters=True, includeClusterZero=False, ignoreForced=False, exclusionfile=EXCLUDE_FILE)
