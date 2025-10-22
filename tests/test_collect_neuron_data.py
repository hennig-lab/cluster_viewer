import json
import tempfile
from pathlib import Path
import pytest
from channel_parser import collect_neuron_data

def make_test_files():
    data_dir = Path(__file__).parent / "data" / "dataset1"
    expected_file = data_dir / "expected_summary.json"
    collect_neuron_data(data_dir, expected_file, pattern="times_*.mat", nbins=50, verbose=True)

def test_collect_neuron_data(tmp_path):
    data_dir = Path(__file__).parent / "data" / "dataset1"
    expected_file = data_dir / "expected_summary.json"
    output_file = tmp_path / "summary.json"

    collect_neuron_data(data_dir, output_file, pattern="times_*.mat", nbins=50, verbose=True)

    # Compare JSON content ignoring key order
    with open(expected_file, "r") as f1, open(output_file, "r") as f2:
        expected = json.load(f1)
        actual = json.load(f2)
    assert actual == expected, "Summary JSON mismatch"
