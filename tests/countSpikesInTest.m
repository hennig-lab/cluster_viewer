%% remove unnecessary fields

datadir = 'data/dataset1';
fnms = dir(fullfile(datadir, '*times*.mat'));
for ii = 1:numel(fnms)
    infile = fullfile(fnms(ii).folder, fnms(ii).name);
    d = load(infile);
    d = rmfield(d, 'par');
    d = rmfield(d, 'inspk');
    d = rmfield(d, 'ipermut');
    d = rmfield(d, 'Temp');
    d = rmfield(d, 'gui_status');
    save(infile, '-struct', 'd');
end

%% summarize spike counts

datadir = 'data/dataset1';
fnms = dir(fullfile(datadir, '*times*.mat'));
counts = [];
for ii = 1:numel(fnms)
    infile = fullfile(fnms(ii).folder, fnms(ii).name);
    d = load(infile);
    disp(fnms(ii).name);
    n0 = size(d.spikes,1);
    n1 = sum(d.detectionLabel == 1);
    grps = unique(d.cluster_class(:,1));
    n2 = sum(d.cluster_class(:,1) == 0);
    ns = [];
    ns_nondupe = [];
    for jj = 1:numel(grps)
        if grps(jj) == 0
            continue;
        end
        ix = d.cluster_class(:,1) == grps(jj);
        ns = [ns sum(ix)];
        ns_nondupe = [ns_nondupe sum(ix & d.detectionLabel == 1)];
    end
    cdata = struct('name', fnms(ii).name, 'ntotal', n0, ...
        'n_nondupe', n1,  'n_zeroClus', n2, 'ngrps', numel(grps), ...
        'grpCounts_noZero', ns, 'grpCounts_noZeroNoDupe', ns_nondupe);
    counts = [counts cdata];
end

%% compare counts to test counts
% after running tests.make_test_files()

datadir = 'data/dataset1';
d0 = load(fullfile(datadir, 'expected_spike_matrix_base.mat'));
d1 = load(fullfile(datadir, 'expected_spike_matrix_default_options.mat'));
d2 = load(fullfile(datadir, 'expected_spike_matrix_ignore_clusters.mat'));
d3 = load(fullfile(datadir, 'expected_spike_matrix_ignore_duplicates.mat'));
d4 = load(fullfile(datadir, 'expected_spike_matrix_with_exclusions.mat'));
exclusionMask = [false true false false true]; % should match clusters_excluded.csv

n0a = [counts.grpCounts_noZero];
n0b = full(sum(d0.spikes,2))';
assert(isequal(n0a, n0b));

n1a = [counts.grpCounts_noZeroNoDupe]; n1a = n1a(~exclusionMask);
n1b = full(sum(d1.spikes,2))';
assert(isequal(n1a, n1b));

n2a = [counts.ntotal]-[counts.n_zeroClus];
n2b = full(sum(d2.spikes,2))';
assert(isequal(n2a, n2b));

n3a = [counts.grpCounts_noZeroNoDupe];
n3b = full(sum(d3.spikes,2))';
assert(isequal(n3a, n3b));

n4a = n0a(~exclusionMask);
n4b = full(sum(d4.spikes,2))';
assert(isequal(n4a, n4b));

