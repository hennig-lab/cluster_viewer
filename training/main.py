import os.path
import csv
import json
import numpy as np

def load_neuron_features(data_file):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")
    neurons = []
    with open(data_file, "r") as f:
        neurons = json.load(f)
    
    features = []
    for n in neurons:
        n['name'] = (n['filename'], n['cluster_id'])
        n['waveforms'] = n['waveform_quintiles']
        features.append(n)
    return features

def get_excluded_clusters(exclude_file):
    if not os.path.exists(exclude_file):
        raise FileNotFoundError(f"Exclude file {exclude_file} not found")
    excluded = set()
    with open(exclude_file, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                excluded.add((row[0], int(row[1])))
    return excluded

def get_training_data(neurons, excluded):
    X = []
    y = []
    for n in neurons:
        waveforms = n['waveforms']
        middle = len(waveforms) // 2
        Xc = waveforms[middle]
        yc = 0 if n['name'] in excluded else 1
        X.append(Xc)
        y.append(yc)
    return X, y

def session_loader(session_dir):
    data_file = os.path.join(session_dir, "neuron_data.json")
    exclude_file = os.path.join(session_dir, "clusters_excluded.csv")
    neurons = load_neuron_features(data_file)
    excluded = get_excluded_clusters(exclude_file)
    return get_training_data(neurons, excluded)

def load_sessions(base_dir):
    X = []
    y = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            session_name = entry.name
            session_dir = os.path.join(base_dir, session_name)
            Xc, yc = session_loader(session_dir)
            X.extend(Xc)
            y.extend(yc)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "training_data")
    X, y = load_sessions(data_dir)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
