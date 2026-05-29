#!/usr/bin/env python3
import os
import csv
import json
import webbrowser
from flask import Flask, jsonify, request, send_from_directory
from channel_parser import collect_neuron_data
from make_spikes_matrix import make_spikes_matrix
from training.data_loader import get_neuron_feature
from training.train import load_model_predictor

app = Flask(__name__, static_folder="static")

app.config["DATA_FILE"] = "neuron_data.json"
app.config["EXCLUDE_FILE"] = "clusters_excluded.csv"
app.config["EXPORT_ARGS"] = None
app.config["MODEL_FILE"] = None

# ---------------------------
# Helpers
# ---------------------------

def load_neurons():
    with open(app.config["DATA_FILE"], "r") as f:
        return json.load(f)

def load_exclusions():
    excluded = set()
    if not os.path.exists(app.config["EXCLUDE_FILE"]):
        save_exclusions(excluded)
        return excluded
    with open(app.config["EXCLUDE_FILE"], newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                excluded.add((row[0], int(row[1])))
    return excluded

def save_exclusions(excluded):
    with open(app.config["EXCLUDE_FILE"], "w", newline="") as f:
        writer = csv.writer(f)
        for fn, cid in sorted(excluded):
            writer.writerow([fn, cid])

def _load_model():
    if app.config["MODEL_FILE"] is None:
        return None
    path = app.config["MODEL_FILE"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found")
    return load_model_predictor(path)

# ---------------------------
# Routes
# ---------------------------

@app.route("/api/neurons")
def api_neurons():
    neurons = load_neurons()
    print(f"Loaded {len(neurons)} neurons from {app.config['DATA_FILE']}")
    excluded = load_exclusions()

    if app.config["MODEL_FILE"] is not None:
        model = _load_model()
        if model is not None:
            print(f"Loaded model from {app.config['MODEL_FILE']}")
    for n in neurons:
        n["excluded"] = (n["filename"], n["cluster_id"]) in excluded
        if app.config["MODEL_FILE"] is not None:
            n["model_feature"] = get_neuron_feature(n)
            n["model_logit"] = model(n["model_feature"])
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

def export_spike_matrices():
    a = app.config["EXPORT_ARGS"]
    if a is None:
        return
    outdir = os.path.join(a.directory, "cluster_viewer_results")
    exclude_file = app.config["EXCLUDE_FILE"]
    print('Creating spike matrix files...')
    make_spikes_matrix(a.directory, outfile=os.path.join(outdir, "spikes.mat"), ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=not a.keep_duplicates, skipEmptyChannels=a.skip_empty_channels, exclusionfile=exclude_file)
    make_spikes_matrix(a.directory, outfile=os.path.join(outdir, "spikes_perChannel.mat"), ignoreClusters=True, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=not a.keep_duplicates, skipEmptyChannels=a.skip_empty_channels, exclusionfile=exclude_file)

@app.route("/api/export", methods=["POST"])
def api_export():
    if app.config["EXPORT_ARGS"] is None:
        return jsonify({"status": "error", "message": "Export not available (no directory set)"}), 400
    export_spike_matrices()
    return jsonify({"status": "ok"})

@app.route("/")
def root():
    return send_from_directory("static", "index.html")

# ---------------------------
# Launch server
# ---------------------------

if __name__ == "__main__":
    import argparse
    from threading import Timer
    basedir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Local viewer for neuron data.")
    parser.add_argument("--directory", default=None, help="Path to directory containing times.mat files")
    parser.add_argument("--jsonfile", default=None, help="Path to neuron JSON file")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default 5000)")
    parser.add_argument("--nbins", type=int, default=50, help="Number of bins (default 50)")
    parser.add_argument("--pattern", default="times_*.mat", help="Filename pattern to match (default: 'times_*.mat')")
    parser.add_argument("--csvfile", default=None, help="Path to CSV exclusion file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--keep_duplicates", action="store_true", help="Keep duplicates (if DER labels are included)")
    parser.add_argument("--skip_empty_channels", action="store_true", help="If set, skips channels without spikes (note this will affect channel indexing)")
    parser.add_argument("--model-file", default=os.path.join(basedir, "training/model.pt"), help="Path to trained model file (.pt) for predictions")

    args = parser.parse_args()

    if args.directory:
        if not os.path.isdir(args.directory):
            raise NotADirectoryError(f"{args.directory} is not a valid directory")
        savedir = os.path.join(args.directory, "cluster_viewer_results")
        os.makedirs(savedir, exist_ok=True)
        app.config["DATA_FILE"] = args.jsonfile if args.jsonfile else os.path.join(savedir, app.config["DATA_FILE"])
        collect_neuron_data(args.directory, app.config["DATA_FILE"], pattern=args.pattern, nbins=args.nbins, verbose=args.verbose, keep_duplicates=args.keep_duplicates)
        app.config["EXPORT_ARGS"] = args
    else:
        if not args.jsonfile:
            raise ValueError("Must provide --directory or --jsonfile")
        app.config["DATA_FILE"] = args.jsonfile
    if not os.path.exists(app.config["DATA_FILE"]):
        raise FileNotFoundError(f"Cannot find {app.config['DATA_FILE']}")
    if args.csvfile is not None:
        app.config["EXCLUDE_FILE"] = args.csvfile
    else:
        app.config["EXCLUDE_FILE"] = os.path.join(os.path.dirname(app.config["DATA_FILE"]), app.config["EXCLUDE_FILE"])

    url = f"http://127.0.0.1:{args.port}"
    print(f"Starting server at {url}")
    Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=False, port=args.port)
