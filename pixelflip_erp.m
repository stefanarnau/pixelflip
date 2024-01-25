clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_ERP_RESULTS  = '/mnt/data_dump/pixelflip/5_erp_results/';

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

% Load info
EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Get erp times
erp_times_idx = EEG.times >= -200 & EEG.times <= 1800;
erp_times = EEG.times(erp_times_idx);

% Get chanlocs
chanlocs = EEG.chanlocs;

% Matrices to collect data. Dimensionality: subjects x channels x times
erp_agen00 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen10 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen11 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_easy = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_hard = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen00_easy = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen00_hard = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen10_easy = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen10_hard = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen11_easy = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen11_hard = zeros(length(subject_list), EEG.nbchan, length(erp_times));

% Loop subjects
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Load subject data. EEG data has dimensionality channels x times x trials
    EEG = pop_loadset('filename',    [subject, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Create below:
    % 12: Previous accuracy (-1 if previous trial in different block or previous trial not available or if previous trial response is missing)
    % 13: Previous flipped (-1 if previous trial in different block or previous trial not available)

    % Loop epochs
    for e = 1 : size(EEG.trialinfo, 1)

        % Check in previous trial if available
        if sum(EEG.trialinfo(:, 1) == EEG.trialinfo(e, 1) - 1) > 0 

            % Get index of previous trial
            idx_prev = find(EEG.trialinfo(:, 1) == EEG.trialinfo(e, 1) - 1);

            % Check if different blocks
            if EEG.trialinfo(e, 2) ~= EEG.trialinfo(idx_prev, 2)
                EEG.trialinfo(e, 12) = -1;
                EEG.trialinfo(e, 13) = -1;
                continue;
            end

            % Set previous accuracy
            EEG.trialinfo(e, 12) = EEG.trialinfo(idx_prev, 11);

            % Set previous flipped
            EEG.trialinfo(e, 13) = EEG.trialinfo(idx_prev, 5);

        else
            EEG.trialinfo(e, 12) = -1;
            EEG.trialinfo(e, 13) = -1;
        end
    end

    trialinfo = EEG.trialinfo;

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

    % Calculate erps
    erp_agen00(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen00)), 3);
    erp_agen10(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen10)), 3);
    erp_agen11(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen11)), 3);
    erp_easy(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_easy)), 3);
    erp_hard(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_hard)), 3);
    erp_agen00_easy(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen00_easy)), 3);
    erp_agen00_hard(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen00_hard)), 3);
    erp_agen10_easy(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen10_easy)), 3);
    erp_agen10_hard(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen10_hard)), 3);
    erp_agen11_easy(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen11_easy)), 3);
    erp_agen11_hard(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen11_hard)), 3);

end

% The order of things
new_order_labels = {...
'Fp1',...
'Fp2',...
'AF7',...
'AF3',...
'AF4',...
'AF8',...
'F9',...
'F5',...
'F3',...
'F1',...
'Fz',...
'F2',...
'F4',...
'F6',...
'F10',...
'FT9',...
'FT7',...
'FC3',...
'FC1',...
'FCz',...
'FC2',...
'FC4',...
'FT8',...
'FT10',...
'T7',...
'C5',...
'C3',...
'C1',...
'Cz',...
'C2',...
'C4',...
'C6',...
'T8',...
'TP9',...
'TP7',...
'CP3',...
'CP1',...
'CPz',...
'CP2',...
'CP4',...
'TP8',...
'TP10',...
'P9',...
'P7',...
'P5',...
'P3',...
'P1',...
'Pz',...
'P2',...
'P4',...
'P6',...
'P8',...
'P10',...
'PO9',...
'PO7',...
'PO3',...
'POz',...
'PO4',...
'PO8',...
'PO10',...
'O9',...
'O1',...
'Oz',...
'O2',...
'O10',...
};

% Get original chanloc labels
chanlocs_labels = {};
for ch = 1 : length(chanlocs)
    chanlocs_labels{end + 1} = chanlocs(ch).labels;
end

% Get new order indices
new_order_idx = [];
for ch = 1 : length(chanlocs)
    new_order_idx(end + 1) = find(strcmpi(new_order_labels{ch}, chanlocs_labels));
end

% Install new order
erp_agen00 = erp_agen00(:, new_order_idx, :);
erp_agen10 = erp_agen10(:, new_order_idx, :);
erp_agen11 = erp_agen11(:, new_order_idx, :);
erp_easy = erp_easy(:, new_order_idx, :);
erp_hard = erp_hard(:, new_order_idx, :);
erp_agen00_easy = erp_agen00_easy(:, new_order_idx, :);
erp_agen00_hard = erp_agen00_hard(:, new_order_idx, :);
erp_agen10_easy = erp_agen10_easy(:, new_order_idx, :);
erp_agen10_hard = erp_agen10_hard(:, new_order_idx, :);
erp_agen11_easy = erp_agen11_easy(:, new_order_idx, :);
erp_agen11_hard = erp_agen11_hard(:, new_order_idx, :);
chanlocs = chanlocs(new_order_idx);

% Restructure coordinates
chanlabs = {};
coords = [];
for c = 1 : numel(chanlocs)
    chanlabs{c} = chanlocs(c).labels;
    coords(c, :) = [chanlocs(c).X, chanlocs(c).Y, chanlocs(c).Z];
end

% A sensor struct
sensors = struct();
sensors.label = chanlabs;
sensors.chanpos = coords;
sensors.elecpos = coords;

% Prepare neighbor struct
cfg                 = [];
cfg.elec            = sensors;
cfg.feedback        = 'no';
cfg.method          = 'triangulation'; 
neighbours          = ft_prepare_neighbours(cfg);

% A template for GA structs
cfg=[];
cfg.keepindividual = 'yes';
ga_template = [];
ga_template.dimord = 'chan_time';
ga_template.label = chanlabs;
ga_template.time = erp_times;

% GA struct agen00
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen00(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_agen00 = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct agen10
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen10(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_agen10 = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct agen11
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen11(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_agen11 = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct easy
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_easy(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_easy = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct hard
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_hard(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_hard = ft_timelockgrandaverage(cfg, GA{1, :});

% GA diff agen00 (hard - easy)
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen00_hard(s, :, :)) - squeeze(erp_agen00_easy(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_diff_agen00 = ft_timelockgrandaverage(cfg, GA{1, :});

% GA diff agen10 (hard - easy)
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen10_hard(s, :, :)) - squeeze(erp_agen10_easy(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_diff_agen10 = ft_timelockgrandaverage(cfg, GA{1, :});

% GA diff agen11 (hard - easy)
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen11_hard(s, :, :)) - squeeze(erp_agen11_easy(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_diff_agen11 = ft_timelockgrandaverage(cfg, GA{1, :});

% Testparams
testalpha  = 0.05;
voxelalpha  = 0.05;
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
[stat_agen_state]  = ft_timelockstatistics(cfg, GA_agen00, GA_agen10);
[stat_agen_sequence]  = ft_timelockstatistics(cfg, GA_agen10, GA_agen11);
[stat_difficulty]  = ft_timelockstatistics(cfg, GA_easy, GA_hard);
[stat_interaction_state] = ft_timelockstatistics(cfg, GA_diff_agen00, GA_diff_agen10);
[stat_interaction_sequence] = ft_timelockstatistics(cfg, GA_diff_agen10, GA_diff_agen11);

% Save cluster structs
save([PATH_ERP_RESULTS 'stat_agen_state.mat'], 'stat_agen_state');
save([PATH_ERP_RESULTS 'stat_agen_sequence.mat'], 'stat_agen_sequence');
save([PATH_ERP_RESULTS 'stat_difficulty.mat'], 'stat_difficulty');
save([PATH_ERP_RESULTS 'stat_interaction_state.mat'], 'stat_interaction_state');
save([PATH_ERP_RESULTS 'stat_interaction_sequence.mat'], 'stat_interaction_sequence');

% Save masks
dlmwrite([PATH_ERP_RESULTS, 'contour_agen_state.csv'], stat_agen_state.mask);
dlmwrite([PATH_ERP_RESULTS, 'contour_agen_sequence.csv'], stat_agen_sequence.mask);
dlmwrite([PATH_ERP_RESULTS, 'contour_difficulty.csv'], stat_difficulty.mask);
dlmwrite([PATH_ERP_RESULTS, 'contour_interaction_state.csv'], stat_interaction_state.mask);
dlmwrite([PATH_ERP_RESULTS, 'contour_interaction_sequence.csv'], stat_interaction_sequence.mask);

% Calculate and save effect sizes
apes_agen_state = [];
apes_agen_sequence = [];
apes_difficulty = [];
apes_interaction_state = [];
apes_interaction_sequence = [];

for ch = 1 : 65

    petasq = (squeeze(stat_agen_state.stat(ch, :)) .^ 2) ./ ((squeeze(stat_agen_state.stat(ch, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_agen_state(ch, :) = adj_petasq;

    petasq = (squeeze(stat_agen_sequence.stat(ch, :)) .^ 2) ./ ((squeeze(stat_agen_sequence.stat(ch, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_agen_sequence(ch, :) = adj_petasq;

    petasq = (squeeze(stat_difficulty.stat(ch, :)) .^ 2) ./ ((squeeze(stat_difficulty.stat(ch, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_difficulty(ch, :) = adj_petasq;

    petasq = (squeeze(stat_interaction_state.stat(ch, :)) .^ 2) ./ ((squeeze(stat_interaction_state.stat(ch, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_interaction_state(ch, :) = adj_petasq;

    petasq = (squeeze(stat_interaction_sequence.stat(ch, :)) .^ 2) ./ ((squeeze(stat_interaction_sequence.stat(ch, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    apes_interaction_sequence(ch, :) = adj_petasq;

end

% Save effect sizes
dlmwrite([PATH_ERP_RESULTS, 'apes_agen_state.csv'], apes_agen_state);
dlmwrite([PATH_ERP_RESULTS, 'apes_agen_sequence.csv'], apes_agen_sequence);
dlmwrite([PATH_ERP_RESULTS, 'apes_difficulty.csv'], apes_difficulty);
dlmwrite([PATH_ERP_RESULTS, 'apes_interaction_state.csv'], apes_interaction_state);
dlmwrite([PATH_ERP_RESULTS, 'apes_interaction_sequence.csv'], apes_interaction_sequence);

figure()
subplot(2, 3, 1)
pd = apes_agen_state;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_agen_state.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('agency state')

subplot(2, 3, 2)
pd = apes_agen_sequence;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_agen_sequence.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('agency sequence')

subplot(2, 3, 3)
pd = apes_difficulty;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_difficulty.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('difficulty')

subplot(2, 3, 4)
pd = apes_interaction_state;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_interaction_state.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('interaction state')

subplot(2, 3, 5)
pd = apes_interaction_sequence;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_interaction_sequence.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('interaction sequence')

% Save lineplots at Fz
dlmwrite([PATH_ERP_RESULTS, 'lineplots_fz.csv'],  [mean(squeeze(erp_agen00_easy(:, 11, :)), 1);...
                                             mean(squeeze(erp_agen00_hard(:, 11, :)), 1);...
                                             mean(squeeze(erp_agen10_easy(:, 11, :)), 1);...
                                             mean(squeeze(erp_agen10_hard(:, 11, :)), 1);...
                                             mean(squeeze(erp_agen11_easy(:, 11, :)), 1);...
                                             mean(squeeze(erp_agen11_hard(:, 11, :)), 1)]);

% Save lineplots at FCz
dlmwrite([PATH_ERP_RESULTS, 'lineplots_fcz.csv'],  [mean(squeeze(erp_agen00_easy(:, 20, :)), 1);...
                                              mean(squeeze(erp_agen00_hard(:, 20, :)), 1);...
                                              mean(squeeze(erp_agen10_easy(:, 20, :)), 1);...
                                              mean(squeeze(erp_agen10_hard(:, 20, :)), 1);...
                                              mean(squeeze(erp_agen11_easy(:, 20, :)), 1);...
                                              mean(squeeze(erp_agen11_hard(:, 20, :)), 1)]);


% Save lineplots at Cz
dlmwrite([PATH_ERP_RESULTS, 'lineplots_cz.csv'],  [mean(squeeze(erp_agen00_easy(:, 29, :)), 1);...
                                             mean(squeeze(erp_agen00_hard(:, 29, :)), 1);...
                                             mean(squeeze(erp_agen10_easy(:, 29, :)), 1);...
                                             mean(squeeze(erp_agen10_hard(:, 29, :)), 1);...
                                             mean(squeeze(erp_agen11_easy(:, 29, :)), 1);...
                                             mean(squeeze(erp_agen11_hard(:, 29, :)), 1)]);

% Save lineplots at CPz
dlmwrite([PATH_ERP_RESULTS, 'lineplots_cpz.csv'],  [mean(squeeze(erp_agen00_easy(:, 38, :)), 1);...
                                              mean(squeeze(erp_agen00_hard(:, 38, :)), 1);...
                                              mean(squeeze(erp_agen10_easy(:, 38, :)), 1);...
                                              mean(squeeze(erp_agen10_hard(:, 38, :)), 1);...
                                              mean(squeeze(erp_agen11_easy(:, 38, :)), 1);...
                                              mean(squeeze(erp_agen11_hard(:, 38, :)), 1)]);

% Save lineplots at Pz
dlmwrite([PATH_ERP_RESULTS, 'lineplots_pz.csv'],  [mean(squeeze(erp_agen00_easy(:, 48, :)), 1);...
                                             mean(squeeze(erp_agen00_hard(:, 48, :)), 1);...
                                             mean(squeeze(erp_agen10_easy(:, 48, :)), 1);...
                                             mean(squeeze(erp_agen10_hard(:, 48, :)), 1);...
                                             mean(squeeze(erp_agen11_easy(:, 48, :)), 1);...
                                             mean(squeeze(erp_agen11_hard(:, 48, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_ERP_RESULTS, 'lineplots_poz.csv'],  [mean(squeeze(erp_agen00_easy(:, 57, :)), 1);...
                                              mean(squeeze(erp_agen00_hard(:, 57, :)), 1);...
                                              mean(squeeze(erp_agen10_easy(:, 57, :)), 1);...
                                              mean(squeeze(erp_agen10_hard(:, 57, :)), 1);...
                                              mean(squeeze(erp_agen11_easy(:, 57, :)), 1);...
                                              mean(squeeze(erp_agen11_hard(:, 57, :)), 1)]);

% Save erp-times
dlmwrite([PATH_ERP_RESULTS, 'erp_times.csv'], erp_times);


% Plot effect size topos at selected time points for agency state
c_lim = [-0.8, 0.8];
tpoints = [700, 1100, 1450];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_agen_state(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(c_lim);
    saveas(gcf, [PATH_ERP_RESULTS, 'topo_agen_state_', num2str(tpoints(t)), 'ms', '.png']);
end

% Plot effect size topos at selected time points for agency sequence
c_lim = [-0.8, 0.8];
tpoints = [700, 1100, 1450];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_agen_sequence(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(c_lim);
    saveas(gcf, [PATH_ERP_RESULTS, 'topo_agen_sequence_', num2str(tpoints(t)), 'ms', '.png']);
end

% Plot effect size topos at selected time points for difficulty
c_lim = [-0.8, 0.8];
tpoints = [120, 450, 1600];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_difficulty(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(c_lim);
    saveas(gcf, [PATH_ERP_RESULTS, 'topo_difficulty_', num2str(tpoints(t)), 'ms', '.png']);
end
