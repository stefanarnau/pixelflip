clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_TF_DATA     = '/mnt/data_dump/pixelflip/3_tf_data/';
PATH_TF_RESULTS  = '/mnt/data_dump/pixelflip/5_tf_results/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP04', 'VP05', 'VP06', 'VP07', 'VP08', 'VP09', 'VP10',...
                'VP11', 'VP12', 'VP13', 'VP14', 'VP15', 'VP16', 'VP17', 'VP18', 'VP19', 'VP20',...
                'VP21', 'VP22', 'VP23', 'VP24', 'VP25', 'VP26', 'VP27', 'VP28', 'VP29', 'VP30',...
                'VP31', 'VP32', 'VP33', 'VP34', 'VP35', 'VP36', 'VP37', 'VP38', 'VP39', 'VP40'};

% Exclude from analysis
subject_list = setdiff(subject_list, {'VP07'}); % Age outlier

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Init fieldtrip
ft_path = '/home/plkn/fieldtrip-master/';
addpath(ft_path);
ft_defaults;

% Load metadata
load([PATH_TF_DATA, 'chanlocs.mat']);
load([PATH_TF_DATA, 'tf_freqs.mat']);
load([PATH_TF_DATA, 'tf_times.mat']);
load([PATH_TF_DATA, 'neighbours.mat']);
load([PATH_TF_DATA, 'obs_fwhmT.mat']);
load([PATH_TF_DATA, 'obs_fwhmF.mat']);

% Remember number of trials of conditions
n_trials = [];

% ERSP matrices
ersp_agen00 = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen10 = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen11 = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_easy = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_hard = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen00_easy = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen00_hard = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen10_easy = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen10_hard = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen11_easy = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));
ersp_agen11_hard = zeros(length(subject_list), 65, length(tf_freqs), length(tf_times));

% Loop subjects
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Load trialinfo
    load([PATH_TF_DATA, 'trialinfo_', subject, '.mat']);

    % Trialinfo columns:
    % 01: trial_nr
    % 02: block_nr
    % 03: reliability
    % 04: difficulty
    % 05: flipped
    % 06: key_pressed
    % 07: rt
    % 08: color_pressed
    % 09: feedback_accuracy
    % 10: feedback_color
    % 11: accuracy  
    % 12: Previous accuracy (-1 if previous trial in different block or previous trial not available or if previous trial response is missing)
    % 13: Previous flipped (-1 if previous trial in different block or previous trial not available)

    % Get trial-indices for main effect agency
    idx_agen00 = trialinfo(:, 3) == 1 & trialinfo(:, 12) == 1 & trialinfo(:, 13) == 0;
    idx_agen10 = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 13) == 0;
    idx_agen11 = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 13) == 1;

    % Get trial-indices for main effect difficulty
    idx_easy = trialinfo(:, 4) == 0 & trialinfo(:, 12) == 1;
    idx_hard = trialinfo(:, 4) == 1 & trialinfo(:, 12) == 1;

    % Get trial-indices for factor combinations
    idx_agen00_easy = trialinfo(:, 3) == 1 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 0 & trialinfo(:, 13) == 0;
    idx_agen00_hard = trialinfo(:, 3) == 1 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 1 & trialinfo(:, 13) == 0;
    idx_agen10_easy = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 0 & trialinfo(:, 13) == 0;
    idx_agen10_hard = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 1 & trialinfo(:, 13) == 0;
    idx_agen11_easy = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 0 & trialinfo(:, 13) == 1;
    idx_agen11_hard = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 1 & trialinfo(:, 13) == 1;

    % Loop channels
    for ch = 1 : 65

        % Talk
        fprintf('\nRead data | subject %i/%i | channel %i/%i\n', s, length(subject_list), ch, 65);

        % Load power data
        load([PATH_TF_DATA, 'powcube_', subject, '_chan_', num2str(ch), '.mat']);
        powcube = double(powcube);

        % Define baseline time window
        ersp_bl = [-500, 0];

        % Get condition general baseline values
        tmp = squeeze(mean(powcube, 3));
        [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        bl_condition_general = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ERSPs using condition-specific baselines 
        ersp_agen00(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen00), 3)), bl_condition_general)));
        ersp_agen10(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen10), 3)), bl_condition_general)));
        ersp_agen11(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen11), 3)), bl_condition_general)));
        ersp_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_easy), 3)), bl_condition_general)));
        ersp_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_hard), 3)), bl_condition_general)));
        ersp_agen00_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen00_easy), 3)), bl_condition_general)));
        ersp_agen00_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen00_hard), 3)), bl_condition_general)));
        ersp_agen10_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen10_easy), 3)), bl_condition_general)));
        ersp_agen10_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen10_hard), 3)), bl_condition_general)));
        ersp_agen11_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen11_easy), 3)), bl_condition_general)));
        ersp_agen11_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen11_hard), 3)), bl_condition_general)));

        % % Get condition specific baseline values
        % tmp = squeeze(mean(powcube(:, :, idx_agen00), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen00 = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen10), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen10 = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen11), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen11 = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_easy), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_easy = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_hard), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_hard = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen00_easy), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen00_easy = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen00_hard), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen00_hard = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen10_easy), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen10_easy = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen10_hard), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen10_hard = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen11_easy), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen11_easy = squeeze(mean(tmp(:, blidx1 : blidx2), 2));
        % tmp = squeeze(mean(powcube(:, :, idx_agen11_hard), 3));
        % [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        % [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        % bl_agen11_hard = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % % Calculate ERSPs using condition-specific baselines 
        % ersp_agen00(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen00), 3)), bl_agen00)));
        % ersp_agen10(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen10), 3)), bl_agen10)));
        % ersp_agen11(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen11), 3)), bl_agen11)));
        % ersp_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_easy), 3)), bl_easy)));
        % ersp_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_hard), 3)), bl_hard)));
        % ersp_agen00_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen00_easy), 3)), bl_agen00_easy)));
        % ersp_agen00_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen00_hard), 3)), bl_agen00_hard)));
        % ersp_agen10_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen10_easy), 3)), bl_agen10_easy)));
        % ersp_agen10_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen10_hard), 3)), bl_agen10_hard)));
        % ersp_agen11_easy(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen11_easy), 3)), bl_agen11_easy)));
        % ersp_agen11_hard(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_agen11_hard), 3)), bl_agen11_hard)));

    end % end channel loop

end % End subject loop

% Prune again to focus cluster permutation testing of cue-target interval
prune_times = [-500, 1200];
time_idx = dsearchn(tf_times', prune_times(1)) : dsearchn(tf_times', prune_times(2));
tf_times = tf_times(time_idx);
ersp_agen00 = ersp_agen00(:, :, :, time_idx);
ersp_agen10 = ersp_agen10(:, :, :, time_idx);
ersp_agen11 = ersp_agen11(:, :, :, time_idx);
ersp_easy = ersp_easy(:, :, :, time_idx);
ersp_hard = ersp_hard(:, :, :, time_idx);
ersp_agen00_easy = ersp_agen00_easy(:, :, :, time_idx);
ersp_agen00_hard = ersp_agen00_hard(:, :, :, time_idx);
ersp_agen10_easy = ersp_agen10_easy(:, :, :, time_idx);
ersp_agen10_hard = ersp_agen10_hard(:, :, :, time_idx);
ersp_agen11_easy = ersp_agen11_easy(:, :, :, time_idx);
ersp_agen11_hard = ersp_agen11_hard(:, :, :, time_idx);

% Get channel labels
chanlabs = {};
for c = 1 : numel(chanlocs)
    chanlabs{c} = chanlocs(c).labels;
end

% A template for GA structs
cfg=[];
cfg.keepindividual = 'yes';
ga_template = [];
ga_template.dimord = 'chan_freq_time';
ga_template.label = chanlabs;
ga_template.freq = tf_freqs;
ga_template.time = tf_times;

% GA struct agen00
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_agen00(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_agen00 = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct agen10
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_agen10(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_agen10 = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct agen11
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_agen11(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_agen11 = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct easy
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_easy(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_easy = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct hard
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_hard(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_hard = ft_freqgrandaverage(cfg, GA{1, :});

% GA diff agen00 (hard - easy)
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_agen00_hard(s, :, :, :)) - squeeze(ersp_agen00_easy(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_diff_agen00 = ft_freqgrandaverage(cfg, GA{1, :});

% GA diff agen10 (hard - easy)
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_agen10_hard(s, :, :, :)) - squeeze(ersp_agen10_easy(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_diff_agen10 = ft_freqgrandaverage(cfg, GA{1, :});

% GA diff agen11 (hard - easy)
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_agen11_hard(s, :, :, :)) - squeeze(ersp_agen11_easy(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_diff_agen11 = ft_freqgrandaverage(cfg, GA{1, :});

% Testparams
testalpha  = 0.05;
voxelalpha  = 0.01;
nperm = 1000;

% Set config
cfg = [];
cfg.tail             = 1;
cfg.statistic        = 'depsamplesFmultivariate';
cfg.alpha            = testalpha;
cfg.neighbours       = neighbours;
cfg.minnbchan        = 2;
cfg.method           = 'montecarlo';
cfg.correctm         = 'cluster';
cfg.clustertail      = 1;
cfg.clusteralpha     = voxelalpha;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = nperm;
cfg.computecritval   = 'yes'; 
cfg.ivar             = 1;
cfg.uvar             = 2;

% Set up design
n_subjects = length(subject_list);
design = zeros(2, n_subjects * 2);
design(1, :) = [ones(1, n_subjects), 2 * ones(1, n_subjects)];
design(2, :) = [1 : n_subjects, 1 : n_subjects];
cfg.design = design;

% The tests
[stat_agen_state]  = ft_freqstatistics(cfg, GA_agen00, GA_agen10);
[stat_agen_sequence]  = ft_freqstatistics(cfg, GA_agen10, GA_agen11);
[stat_difficulty]  = ft_freqstatistics(cfg, GA_easy, GA_hard);
[stat_interaction_state] = ft_freqstatistics(cfg, GA_diff_agen00, GA_diff_agen10);
[stat_interaction_sequence] = ft_freqstatistics(cfg, GA_diff_agen10, GA_diff_agen11);

% Save cluster structs
save([PATH_TF_RESULTS 'stat_agen_state.mat'], 'stat_agen_state');
save([PATH_TF_RESULTS 'stat_agen_sequence.mat'], 'stat_agen_sequence');
save([PATH_TF_RESULTS 'stat_difficulty.mat'], 'stat_difficulty');
save([PATH_TF_RESULTS 'stat_interaction_state.mat'], 'stat_interaction_state');
save([PATH_TF_RESULTS 'stat_interaction_sequence.mat'], 'stat_interaction_sequence');

% Save masks
dlmwrite([PATH_TF_RESULTS, 'contour_agen_state.csv'], stat_agen_state.mask);
dlmwrite([PATH_TF_RESULTS, 'contour_agen_sequence.csv'], stat_agen_sequence.mask);
dlmwrite([PATH_TF_RESULTS, 'contour_difficulty.csv'], stat_difficulty.mask);
dlmwrite([PATH_TF_RESULTS, 'contour_interaction_state.csv'], stat_interaction_state.mask);
dlmwrite([PATH_TF_RESULTS, 'contour_interaction_sequence.csv'], stat_interaction_sequence.mask);

% Calculate and save effect sizes
apes_agen_state = [];
apes_agen_sequence = [];
apes_difficulty = [];
apes_interaction_state = [];
apes_interaction_sequence = [];

for ch = 1 : 65

    petasq = (squeeze(stat_agen_state.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_agen_state.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_agen_state(ch, :, :) = adj_petasq;

    petasq = (squeeze(stat_agen_sequence.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_agen_sequence.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_agen_sequence(ch, :, :) = adj_petasq;

    petasq = (squeeze(stat_difficulty.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_difficulty.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_difficulty(ch, :, :) = adj_petasq;

    petasq = (squeeze(stat_interaction_state.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_interaction_state.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_interaction_state(ch, :, :) = adj_petasq;

    petasq = (squeeze(stat_interaction_sequence.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_interaction_sequence.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_interaction_sequence(ch, :, :) = adj_petasq;

end

% Save effect sizes
save([PATH_TF_RESULTS, 'apes_agen_state.mat'], 'apes_agen_state');
save([PATH_TF_RESULTS, 'apes_agen_sequence.mat'], 'apes_agen_sequence');
save([PATH_TF_RESULTS, 'apes_difficulty.mat'], 'apes_difficulty');
save([PATH_TF_RESULTS, 'apes_interaction_state.mat'], 'apes_interaction_state');
save([PATH_TF_RESULTS, 'apes_interaction_sequence.mat'], 'apes_interaction_sequence');

% Identify significant clusters
clust_thresh = 0.1;
clusts = struct();
cnt = 0;
stat_names = {'stat_agen_state', 'stat_agen_sequence', 'stat_difficulty', 'stat_interaction_state', 'stat_interaction_sequence'};
for s = 1 : numel(stat_names)
    stat = eval(stat_names{s});
    if ~isempty(stat.posclusters)
        pos_idx = find([stat.posclusters(1, :).prob] < clust_thresh);
        for c = 1 : numel(pos_idx)
            cnt = cnt + 1;
            clusts(cnt).testlabel = stat_names{s};
            clusts(cnt).clustnum = cnt;
            clusts(cnt).time = stat.time;
            clusts(cnt).freq = stat.freq;
            clusts(cnt).prob = stat.posclusters(1, pos_idx(c)).prob;
            clusts(cnt).idx = stat.posclusterslabelmat == pos_idx(c);
            clusts(cnt).stats = clusts(cnt).idx .* stat.stat;
            clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2, 3])));
        end
    end
end

% Plot identified cluster
clinecol = 'k';
cmap = 'jet';
for cnt = 1 : numel(clusts)

    figure('Visible', 'off'); clf;

    subplot(2, 2, 1)
    pd = squeeze(sum(clusts(cnt).stats, 1));
    contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
    hold on
    contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
    colormap(cmap)
    set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-max(abs(pd(:))), max(abs(pd(:)))], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
    colorbar;
    title(['sum t across chans '], 'FontSize', 10)

    subplot(2, 2, 2)
    pd = squeeze(mean(clusts(cnt).idx, 1));
    contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
    hold on
    contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
    colormap(cmap)
    set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-1, 1], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
    colorbar;
    title(['proportion chans significant'], 'FontSize', 10)

    subplot(2, 2, 3)
    pd = squeeze(sum(clusts(cnt).stats, [2, 3]));
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(cmap)
    set(gca, 'clim', [-max(abs(pd(:))), max(abs(pd(:)))])
    colorbar;
    title(['sum t per electrode'], 'FontSize', 10)

    subplot(2, 2, 4)
    pd = squeeze(mean(clusts(cnt).idx, [2, 3]));
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(cmap)
    set(gca, 'clim', [-1, 1])
    colorbar;
    title(['proportion tf-points significant'], 'FontSize', 10)

    saveas(gcf, [PATH_TF_RESULTS 'clustnum_' num2str(clusts(cnt).clustnum) '_' clusts(cnt).testlabel '.png']); 

end



% Plot some effect sizes
figure()
pd = squeeze(mean(apes_agen_state, 1));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
clim([0, 0.5])
colormap(hot)

th00 = squeeze(mean(GA_agen00.powspctrm(:, 65, tf_freqs >= 4 & tf_freqs <= 7, :), [1, 3]));
th10 = squeeze(mean(GA_agen10.powspctrm(:, 65, tf_freqs >= 4 & tf_freqs <= 7, :), [1, 3]));
th11 = squeeze(mean(GA_agen11.powspctrm(:, 65, tf_freqs >= 4 & tf_freqs <= 7, :), [1, 3]));
figure()
plot(tf_times, [th00,th10,th11])
legend({'00','10','11'})