clear all;

% Path variables
PATH_EEGLAB       = '/home/plkn/eeglab2023.1/';
PATH_AUTOCLEANED  = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_COR_RESULTS  = '/mnt/data_dump/pixelflip/5_correlation_results/';
PATH_RATINGS      = '/mnt/data_dump/pixelflip/veusz/subjecctive_ratings/';  
PATH_ERP_RESULTS  = '/mnt/data_dump/pixelflip/5_erp_results/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP04', 'VP05', 'VP06', 'VP07', 'VP08', 'VP09', 'VP10',...
                'VP11', 'VP12', 'VP13', 'VP14', 'VP15', 'VP16', 'VP17', 'VP18', 'VP19', 'VP20',...
                'VP21', 'VP22', 'VP23', 'VP24', 'VP25', 'VP26', 'VP27', 'VP28', 'VP29', 'VP30',...
                'VP31', 'VP32', 'VP33', 'VP34', 'VP35', 'VP36', 'VP37', 'VP38', 'VP39', 'VP40'};

% Exclude from analysis
subject_list = setdiff(subject_list, {'VP07'}); % Age outlier

% Load ratings
load([PATH_RATINGS, 'table_ratings.mat']);

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
erp_times_idx = EEG.times >= -200 & EEG.times <= 1200;
erp_times = EEG.times(erp_times_idx);

% Get chanlocs
chanlocs = EEG.chanlocs;

% Matrices to collect data. Dimensionality: subjects x channels x times
erp_agen00 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_agen10 = zeros(length(subject_list), EEG.nbchan, length(erp_times));

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

    % Calculate erps
    erp_agen00(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen00)), 3);
    erp_agen10(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_agen10)), 3);

end

% Apply moving average
%winlength = 31;
%erp_agen00 = movmean(erp_agen00, winlength, 3);
%erp_agen10 = movmean(erp_agen10, winlength, 3);

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

% GA struct noflip
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen00(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_agen00 = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct flip
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen10(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_agen10= ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct difference ERP
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen00(s, :, :)) - squeeze(erp_agen10(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_diff = ft_timelockgrandaverage(cfg, GA{1, :});

% Correlations statistic config
cfg.statistic = 'ft_statfun_correlationT';
cfg.alpha = 0.025;
cfg.neighbours = neighbours;
cfg.minnbchan = 2;
cfg.method = 'montecarlo';
cfg.correctm = 'cluster';
cfg.clustertail = 0;
cfg.clusteralpha = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = 1000;
cfg.computecritval = 'yes';
cfg.ivar = 1;

% Tests no differences
cfg.design = T.focus_accu;
[stat_focus_accu_00] = ft_timelockstatistics(cfg, GA_agen00);
cfg.design = T.moti_accu;
[stat_moti_accu_00] = ft_timelockstatistics(cfg, GA_agen00);
cfg.design = T.mw_accu;
[stat_mw_accu_00] = ft_timelockstatistics(cfg, GA_agen00);
cfg.design = T.focus_flip;
[stat_focus_flip_10] = ft_timelockstatistics(cfg, GA_agen10);
cfg.design = T.moti_flip;
[stat_moti_flip_10] = ft_timelockstatistics(cfg, GA_agen10);
cfg.design = T.mw_flip;
[stat_mw_flip_10] = ft_timelockstatistics(cfg, GA_agen10);

% Tests difference
cfg.design = T.focus_accu - T.focus_flip;
[stat_diff_focus] = ft_timelockstatistics(cfg, GA_diff);
cfg.design = T.moti_accu - T.moti_flip;
[stat_diff_moti] = ft_timelockstatistics(cfg, GA_diff);
cfg.design = T.mw_accu - T.mw_flip;
[stat_diff_mw] = ft_timelockstatistics(cfg, GA_diff);

% Save masks and rho
dlmwrite([PATH_COR_RESULTS, 'rho_focus_accu_00.csv'], stat_focus_accu_00.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_focus_accu_00.csv'], stat_focus_accu_00.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_moti_accu_00.csv'], stat_moti_accu_00.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_moti_accu_00.csv'], stat_moti_accu_00.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_mw_accu_00.csv'], stat_mw_accu_00.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_mw_accu_00.csv'], stat_mw_accu_00.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_focus_flip_10.csv'], stat_focus_flip_10.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_focus_flip_10.csv'], stat_focus_flip_10.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_moti_flip_10.csv'], stat_moti_flip_10.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_moti_flip_10.csv'], stat_moti_flip_10.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_mw_flip_10.csv'], stat_mw_flip_10.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_mw_flip_10.csv'], stat_mw_flip_10.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_focus_diff.csv'], stat_diff_focus.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_focus_diff.csv'], stat_diff_focus.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_moti_diff.csv'], stat_diff_moti.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_moti_diff.csv'], stat_diff_moti.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_mw_diff.csv'], stat_diff_mw.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_mw_diff.csv'], stat_diff_mw.mask);

% plot
figure()

subplot(3, 3, 1)
pd = stat_focus_accu_00.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_focus_accu_00.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('focus accu 00')

subplot(3, 3, 2)
pd = stat_moti_accu_00.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_moti_accu_00.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('moti accu 00')

subplot(3, 3, 3)
pd = stat_mw_accu_00.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_mw_accu_00.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('mw accu 00')

subplot(3, 3, 4)
pd = stat_focus_flip_10.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_focus_flip_10.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('focus flip 10')

subplot(3, 3, 5)
pd = stat_moti_flip_10.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_moti_flip_10.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('moti flip 10')

subplot(3, 3, 6)
pd = stat_mw_flip_10.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_mw_flip_10.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('mw flip 10')

subplot(3, 3, 7)
pd = stat_diff_focus.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_diff_focus.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('focus diff')

subplot(3, 3, 8)
pd = stat_diff_moti.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_diff_moti.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('moti diff')

subplot(3, 3, 9)
pd = stat_diff_mw.rho;
contourf(erp_times, [1 : 65], pd, 40, 'linecolor','none')
clim([-0.7, 0.7])
hold on
pd = stat_diff_mw.mask;
contour(erp_times, [1 : 65], pd, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap(jet)
title('mw diff')




aa = bb;


% Load erp-mask
erp_mask = logical(dlmread([PATH_ERP_RESULTS, 'contour_agen_state.csv']));
erp_mask = erp_mask(:, erp_times_idx);


% Get significant cluster averages
motivation_diffs = T.moti_accu - T.moti_flip;
ave_clust_diffs = [];
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen00(s, :, :)) - squeeze(erp_agen10(s, :, :));
    ave_clust_diffs(s) = mean(tmp(erp_mask));
end
figure()
scatter(ave_clust_diffs, motivation_diffs)
corrcoef(ave_clust_diffs, motivation_diffs')






% Plot topos at selected time points
clim = [-0.05, 0.3];
tpoints = [200, 500, 800];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_agency(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(flipud(bone));
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_agency_', num2str(tpoints(t)), 'ms', '.png']);
end


