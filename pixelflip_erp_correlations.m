clear all;

% Path variables
PATH_EEGLAB       = '/home/plkn/eeglab2023.1/';
PATH_AUTOCLEANED  = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_COR_RESULTS  = '/mnt/data_dump/pixelflip/5_correlation_results/';
PATH_RATINGS      = '/mnt/data_dump/pixelflip/veusz/subjecctive_ratings/';  

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
    tmp = squeeze(erp_agen00(s, :, :)) - squeeze(erp_agen10(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_diff_state = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct agen10
GA = {};
for s = 1 : length(subject_list)
    tmp = squeeze(erp_agen10(s, :, :)) - squeeze(erp_agen11(s, :, :));
    ga_template.avg = tmp;
    GA{s} = ga_template;
end 
GA_diff_sequence = ft_timelockgrandaverage(cfg, GA{1, :});

% Correlations focus
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
cfg.design = T.focus_accu - T.focus_flip;

% The test
[stat_focus_state]    = ft_timelockstatistics(cfg, GA_diff_state);
[stat_focus_sequence] = ft_timelockstatistics(cfg, GA_diff_sequence);

% Correlations motivation
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
cfg.design = T.moti_accu - T.moti_flip;

% The test
[stat_moti_state]    = ft_timelockstatistics(cfg, GA_diff_state);
[stat_moti_sequence] = ft_timelockstatistics(cfg, GA_diff_sequence);

% Correlations mind wandering
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
cfg.design = T.mw_accu - T.mw_flip;

% The test
[stat_mw_state]    = ft_timelockstatistics(cfg, GA_diff_state);
[stat_mw_sequence] = ft_timelockstatistics(cfg, GA_diff_sequence);

% Save masks and rho
dlmwrite([PATH_COR_RESULTS, 'rho_focus_state.csv'], stat_focus_state.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_focus_state.csv'], stat_focus_state.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_focus_sequence.csv'], stat_focus_sequence.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_focus_sequence.csv'], stat_focus_sequence.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_moti_state.csv'], stat_moti_state.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_moti_state.csv'], stat_moti_state.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_moti_sequence.csv'], stat_moti_sequence.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_moti_sequence.csv'], stat_moti_sequence.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_mw_state.csv'], stat_mw_state.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_mw_state.csv'], stat_mw_state.mask);
dlmwrite([PATH_COR_RESULTS, 'rho_mw_sequence.csv'], stat_mw_sequence.rho);
dlmwrite([PATH_COR_RESULTS, 'mask_mw_sequence.csv'], stat_mw_sequence.mask);

aa=bb











% Save lineplots at Fz
dlmwrite([PATH_VEUSZ, 'lineplots_fz.csv'],  [mean(squeeze(erp_asis(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip(:, 11, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_VEUSZ, 'lineplots_fcz.csv'], [mean(squeeze(erp_asis(:, 20, :)), 1);...
                                             mean(squeeze(erp_flip(:, 20, :)), 1)]);

dlmwrite([PATH_VEUSZ, 'lineplots_pz.csv'],  [mean(squeeze(erp_asis(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip(:, 48, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_VEUSZ, 'lineplots_poz.csv'], [mean(squeeze(erp_asis(:, 57, :)), 1);...
                                             mean(squeeze(erp_flip(:, 57, :)), 1)]);

% Save erp-times
dlmwrite([PATH_VEUSZ, 'erp_times.csv'], erp_times);

% Plot topos at selected time points for agency
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


