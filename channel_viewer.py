#!/usr/bin/env python3
import os
import csv
import json
import webbrowser
from flask import Flask, jsonify, request, send_from_directory

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
    if not os.path.exists(EXCLUDE_FILE):
        return set()
    excluded = set()
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
    parser.add_argument("data", default="neuron_data.json", help="Path to neuron JSON file")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default 5000)")
    parser.add_argument("--csvfile", default=None, help="Path to CSV exclusion file")
    args = parser.parse_args()

    DATA_FILE = args.data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find {DATA_FILE}")
    EXCLUDE_FILE = args.csvfile
    if EXCLUDE_FILE is None:
        EXCLUDE_FILE = os.path.join(os.path.dirname(DATA_FILE), "clusters_excluded.csv")

    url = f"http://127.0.0.1:{args.port}"

    print(f"Starting server at {url}")
    Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=False, port=args.port)
