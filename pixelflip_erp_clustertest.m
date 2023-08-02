clear all;

% PATH VARS - PLEASE ADJUST!!!!!
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_VEUSZ       = '/mnt/data_dump/pixelflip/veusz/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP04', 'VP05', 'VP06', 'VP07', 'VP08', 'VP09', 'VP10',...
                'VP11', 'VP12', 'VP13', 'VP14', 'VP15', 'VP16', 'VP17', 'VP18', 'VP19', 'VP20',...
                'VP21', 'VP22', 'VP23', 'VP24', 'VP25', 'VP26', 'VP27', 'VP28', 'VP29', 'VP30',...
                'VP31', 'VP32', 'VP33', 'VP34', 'VP35', 'VP36', 'VP37', 'VP38'};

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
erp_times_idx = EEG.times >= -200 & EEG.times <= 1200;
erp_times = EEG.times(erp_times_idx);

% Get chanlocs
chanlocs = EEG.chanlocs;

% Matrices to collect data. Dimensionality: subjects x channels x times
erp_easy_accu = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_easy_flip = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_hard_accu = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_hard_flip = zeros(length(subject_list), EEG.nbchan, length(erp_times));

% Loop subjects
ids = [];
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Collect IDs as number
    ids(s) = str2num(subject(3 : 4));

    % Load subject data. EEG data has dimensionality channels x times x trials
    EEG = pop_loadset('filename',    [subject, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');
    
    % Get trial-indices of conditions
    idx_easy_accu = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 1;
    idx_easy_flip = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 0;
    idx_hard_accu = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 1;
    idx_hard_flip = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 0;
    
    % Calculate subject ERPs by averaging across trials for each condition.
    erp_easy_accu(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_easy_accu)), 3);
    erp_easy_flip(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_easy_flip)), 3);
    erp_hard_accu(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_hard_accu)), 3);
    erp_hard_flip(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_hard_flip)), 3);

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
erp_easy_accu = erp_easy_accu(:, new_order_idx, :);
erp_easy_flip = erp_easy_flip(:, new_order_idx, :);
erp_hard_accu = erp_hard_accu(:, new_order_idx, :);
erp_hard_flip = erp_hard_flip(:, new_order_idx, :);
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

% GA struct easy trials
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_easy_accu(s, :, :)) + squeeze(erp_easy_flip(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_easy = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct easy trials
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_hard_accu(s, :, :)) + squeeze(erp_hard_flip(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_hard = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct accu trials
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_easy_accu(s, :, :)) + squeeze(erp_hard_accu(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_accu = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct flip trials
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_easy_flip(s, :, :)) + squeeze(erp_hard_flip(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_flip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct interaction accu
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_easy_accu(s, :, :)) - squeeze(erp_hard_accu(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_interaction_accu = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct interaction flip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_easy_flip(s, :, :)) - squeeze(erp_hard_flip(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_interaction_flip = ft_timelockgrandaverage(cfg, GA{1, :});

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
[stat_difficulty]  = ft_timelockstatistics(cfg, GA_easy, GA_hard);
[stat_agency]      = ft_timelockstatistics(cfg, GA_accu, GA_flip);
[stat_interaction] = ft_timelockstatistics(cfg, GA_interaction_accu, GA_interaction_flip);

% Significant clusters difficulty
sig = find([stat_difficulty.posclusters.prob] <= testalpha);
for cl = 1 : numel(sig)
    idx = stat_difficulty.posclusterslabelmat == sig(cl);
    pval = round(stat_difficulty.posclusters(sig(cl)).prob, 3);
    dlmwrite([PATH_VEUSZ, 'difficulty_cluster_', num2str(cl), '_contour.csv'], idx);
end

% Significant clusters agency
sig = find([stat_agency.posclusters.prob] <= testalpha);
for cl = 1 : numel(sig)
    idx = stat_agency.posclusterslabelmat == sig(cl);
    pval = round(stat_agency.posclusters(sig(cl)).prob, 3);
    dlmwrite([PATH_VEUSZ, 'agency_cluster_', num2str(cl), '_contour.csv'], idx);
end

% Significant clusters interaction
sig = find([stat_interaction.posclusters.prob] <= testalpha);
for cl = 1 : numel(sig)
    idx = stat_interaction.posclusterslabelmat == sig(cl);
    pval = round(stat_interaction.posclusters(sig(cl)).prob, 3);
    dlmwrite([PATH_VEUSZ, 'interactioncluster_', num2str(cl), '_contour.csv'], idx);
end

% Calculate effect sizes
n_chans = numel(chanlocs);
apes_difficulty  = [];
apes_agency      = [];
apes_interaction = [];
df_effect = 1;
for ch = 1 : n_chans
    petasq = (squeeze(stat_difficulty.stat(ch, :)) * df_effect) ./ ((squeeze(stat_difficulty.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_difficulty(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
    petasq = (squeeze(stat_agency.stat(ch, :)) * df_effect) ./ ((squeeze(stat_agency.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_agency(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
    petasq = (squeeze(stat_interaction.stat(ch, :)) * df_effect) ./ ((squeeze(stat_interaction.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_interaction(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
end

% Save effect sizes
dlmwrite([PATH_VEUSZ, 'apes_difficulty.csv'], apes_difficulty);
dlmwrite([PATH_VEUSZ, 'apes_agency.csv'], apes_agency);
dlmwrite([PATH_VEUSZ, 'apes_interaction.csv'], apes_interaction);

% Get cluster-times for agency
sig = find([stat_agency.posclusters.prob] <= testalpha);
agency_clust_times = {};
for cl = 1 : numel(sig)
    figure('Visible', 'off'); clf;
    idx = stat_agency.posclusterslabelmat == sig(cl);
    idx_time = logical(mean(idx, 1));
    agency_clust_times{cl} = erp_times(idx_time);
end



% Save lineplots at Fz
dlmwrite([PATH_VEUSZ, 'lineplots_fz.csv'],  [mean(squeeze(erp_easy_accu(:, 11, :)), 1);...
                                             mean(squeeze(erp_easy_flip(:, 11, :)), 1);...
                                             mean(squeeze(erp_hard_accu(:, 11, :)), 1);...
                                             mean(squeeze(erp_hard_flip(:, 11, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_VEUSZ, 'lineplots_fcz.csv'], [mean(squeeze(erp_easy_accu(:, 20, :)), 1);...
                                             mean(squeeze(erp_easy_flip(:, 20, :)), 1);...
                                             mean(squeeze(erp_hard_accu(:, 20, :)), 1);...
                                             mean(squeeze(erp_hard_flip(:, 20, :)), 1)]);

dlmwrite([PATH_VEUSZ, 'lineplots_pz.csv'],  [mean(squeeze(erp_easy_accu(:, 48, :)), 1);...
                                             mean(squeeze(erp_easy_flip(:, 48, :)), 1);...
                                             mean(squeeze(erp_hard_accu(:, 48, :)), 1);...
                                             mean(squeeze(erp_hard_flip(:, 48, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_VEUSZ, 'lineplots_poz.csv'], [mean(squeeze(erp_easy_accu(:, 57, :)), 1);...
                                             mean(squeeze(erp_easy_flip(:, 57, :)), 1);...
                                             mean(squeeze(erp_hard_accu(:, 57, :)), 1);...
                                             mean(squeeze(erp_hard_flip(:, 57, :)), 1)]);

% Save erp-times
dlmwrite([PATH_VEUSZ, 'erp_times.csv'], erp_times);

% Plot effect size topos at selected time points for agency
clim = [-0.05, 0.3];
tpoints = [340, 570, 700, 880, 1100];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_agency(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(flipud(bone));
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_agency_', num2str(tpoints(t)), 'ms', '.png']);
end

% Plot effect size topos at selected time points for difficulty
clim = [-0.05, 0.3];
tpoints = [120, 220, 420, 700, 1100];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_difficulty(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(flipud(bone));
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_difficulty_', num2str(tpoints(t)), 'ms', '.png']);
end

