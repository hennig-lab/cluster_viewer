import tempfile
from pathlib import Path
import pytest
from scipy.io import loadmat
from make_spikes_matrix import make_spikes_matrix
from tests.compare_mat_files import compare_struct_fields

def make_test_files():
    data_dir = Path(__file__).parent / "data" / "dataset1"

    expected_file = str(data_dir / "expected_spike_matrix_base.mat")
    make_spikes_matrix(str(data_dir), expected_file, ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=False, skipEmptyChannels=False, exclusionfile=None)

    expected_file = str(data_dir / "expected_spike_matrix_ignore_clusters.mat")
    make_spikes_matrix(str(data_dir), expected_file, ignoreClusters=True, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=False, skipEmptyChannels=False, exclusionfile=None)

    expected_file = str(data_dir / "expected_spike_matrix_ignore_duplicates.mat")
    make_spikes_matrix(str(data_dir), expected_file, ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=True, skipEmptyChannels=False, exclusionfile=None)

    expected_file = str(data_dir / "expected_spike_matrix_with_exclusions.mat")
    exclusion_file = str(data_dir / "clusters_excluded.csv")
    make_spikes_matrix(str(data_dir), expected_file, ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=False, skipEmptyChannels=False, exclusionfile=exclusion_file)

    expected_file = str(data_dir / "expected_spike_matrix_default_options.mat")
    exclusion_file = str(data_dir / "clusters_excluded.csv")
    make_spikes_matrix(str(data_dir), expected_file, ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=True, skipEmptyChannels=False, exclusionfile=exclusion_file)

def test_make_spikes_matrix_base(tmp_path):
    data_dir = Path(__file__).parent / "data" / "dataset1"
    expected_file = data_dir / "expected_spike_matrix_base.mat"
    output_file = tmp_path / "spike_matrix.mat"

    make_spikes_matrix(str(data_dir), str(output_file), ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=False, skipEmptyChannels=False, exclusionfile=None)

    # Compare mat files
    expected = loadmat(expected_file)
    actual = loadmat(output_file)
    assert compare_struct_fields(actual, expected), "Spike matrix mismatch, base case"

def test_make_spikes_matrix_ignore_clusters(tmp_path):
    data_dir = Path(__file__).parent / "data" / "dataset1"
    expected_file = data_dir / "expected_spike_matrix_ignore_clusters.mat"
    output_file = tmp_path / "spike_matrix.mat"

    make_spikes_matrix(str(data_dir), str(output_file), ignoreClusters=True, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=False, skipEmptyChannels=False, exclusionfile=None)

    # Compare mat files
    expected = loadmat(expected_file)
    actual = loadmat(output_file)
    assert compare_struct_fields(actual, expected), "Spike matrix mismatch, ignoring clusters"

def test_make_spikes_matrix_ignore_duplicates(tmp_path):
    data_dir = Path(__file__).parent / "data" / "dataset1"
    expected_file = data_dir / "expected_spike_matrix_ignore_duplicates.mat"
    output_file = tmp_path / "spike_matrix.mat"

    make_spikes_matrix(str(data_dir), str(output_file), ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=True, skipEmptyChannels=False, exclusionfile=None)

    # Compare mat files
    expected = loadmat(str(expected_file))
    actual = loadmat(str(output_file))
    assert compare_struct_fields(actual, expected), "Spike matrix mismatch, ignoring duplicates"

def test_make_spikes_matrix_with_exclusions(tmp_path):
    data_dir = Path(__file__).parent / "data" / "dataset1"
    expected_file = data_dir / "expected_spike_matrix_with_exclusions.mat"
    exclusion_file = data_dir / "clusters_excluded.csv"
    output_file = tmp_path / "spike_matrix.mat"

    make_spikes_matrix(str(data_dir), str(output_file), ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=False, skipEmptyChannels=False, exclusionfile=exclusion_file)

    # Compare mat files
    expected = loadmat(str(expected_file))
    actual = loadmat(str(output_file))
    assert compare_struct_fields(actual, expected), "Spike matrix mismatch, with exclusions"

def test_make_spikes_matrix_default_options(tmp_path):
    data_dir = Path(__file__).parent / "data" / "dataset1"
    expected_file = data_dir / "expected_spike_matrix_default_options.mat"
    exclusion_file = data_dir / "clusters_excluded.csv"
    output_file = tmp_path / "spike_matrix.mat"

    make_spikes_matrix(str(data_dir), str(output_file), ignoreClusters=False, includeClusterZero=False, ignoreForced=False, ignoreDuplicates=True, skipEmptyChannels=False, exclusionfile=exclusion_file)

    # Compare mat files
    expected = loadmat(str(expected_file))
    actual = loadmat(str(output_file))
    assert compare_struct_fields(actual, expected), "Spike matrix mismatch, with default options"
