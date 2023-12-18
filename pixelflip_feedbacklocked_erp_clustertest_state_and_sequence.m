clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_VEUSZ       = '/mnt/data_dump/pixelflip/veusz/feedback_erp_state/';  

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
EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_feedback_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Get erp times
erp_times_idx = EEG.times >= -200 & EEG.times <= 1000;
erp_times = EEG.times(erp_times_idx);

% Get chanlocs
chanlocs = EEG.chanlocs;

% Matrices to collect data. Dimensionality: subjects x channels x times
erp_flip0_easy_flip0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip0_hard_flip0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_easy_flip0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_hard_flip0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_easy_flip1 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_hard_flip1 = zeros(length(subject_list), EEG.nbchan, length(erp_times));

% Loop subjects
ids = [];
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Collect IDs as number
    ids(s) = str2num(subject(3 : 4));

    % Load subject data. EEG data has dimensionality channels x times x trials
    EEG = pop_loadset('filename', [subject, '_cleaned_feedback_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Load cuelocked data for baseline values
    BLVALS = pop_loadset('filename', [subject, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'check', 'on');

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

    % Loop epochs
    for e = 1 : size(EEG.trialinfo, 1)

        % If in flip block
        if EEG.trialinfo(e, 3) == 0

            % Check in previous trial if available
            if sum(EEG.trialinfo(:, 1) == EEG.trialinfo(e, 1) - 1) > 0 

                % Get index of previous trial
                idx_prev = find(EEG.trialinfo(:, 1) == EEG.trialinfo(e, 1) - 1);

                % Check if different blocks
                if EEG.trialinfo(e, 2) ~= EEG.trialinfo(idx_prev, 2)

                    % Not a good trial...
                    EEG.trialinfo(e, 12) = -1;
                    EEG.trialinfo(e, 13) = -1;

                    % Next
                    continue;

                end

                % Check if previous incorrect
                if EEG.trialinfo(idx_prev, 11) ~= 1

                    % Not a good trial...
                    EEG.trialinfo(e, 12) = -1;
                    EEG.trialinfo(e, 13) = -1;

                    % Next
                    continue;

                end

                % Check if previous was flipped
                if EEG.trialinfo(idx_prev, 5) == 1

                    % Yes
                    EEG.trialinfo(e, 12) = 1;
                    EEG.trialinfo(e, 13) = EEG.trialinfo(idx_prev, 4);

                elseif EEG.trialinfo(idx_prev, 5) == 0

                    % No
                    EEG.trialinfo(e, 12) = 0;
                    EEG.trialinfo(e, 13) = EEG.trialinfo(idx_prev, 4);
                
                end
            end

        % If not a flip block
        else

            % Not a good trial...
            EEG.trialinfo(e, 12) = -1;
            EEG.trialinfo(e, 13) = -1;

            % Next
            continue;
    
        end
    end

    % Find trials that are present in both, feedbacklocked and cuelocked, datasets
    common_trials = intersect(BLVALS.trialinfo(:, 1), EEG.trialinfo(:, 1));

    % Reduce datasets to common trials
    to_keep = ismember(EEG.trialinfo(:, 1), common_trials); 
    EEG.data = EEG.data(:, :, to_keep);
    EEG.trialinfo = EEG.trialinfo(to_keep, :);
    to_keep = ismember(BLVALS.trialinfo(:, 1), common_trials); 
    BLVALS.data = BLVALS.data(:, :, to_keep);
    BLVALS.trialinfo = BLVALS.trialinfo(to_keep, :);
    
    % Get trial-indices of conditions
    idx_flip0_easy_flip0 = EEG.trialinfo(:, 11) == 1 & EEG.trialinfo(:, 3) == 1 & EEG.trialinfo(:, 4) == 0;
    idx_flip0_hard_flip0 = EEG.trialinfo(:, 11) == 1 & EEG.trialinfo(:, 3) == 1 & EEG.trialinfo(:, 4) == 1;
    idx_flip1_easy_flip0 = EEG.trialinfo(:, 11) == 1 & EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 5) == 0;
    idx_flip1_hard_flip0 = EEG.trialinfo(:, 11) == 1 & EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 5) == 0;
    idx_flip1_easy_flip1 = EEG.trialinfo(:, 11) == 1 & EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 5) == 1;
    idx_flip1_hard_flip1 = EEG.trialinfo(:, 11) == 1 & EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 5) == 1;

    % Get condition specific baseline values for feedbacklocked erps from before cue-onset
    idx_bl = BLVALS.times >= -200 & BLVALS.times <= 0;
    bl_flip0_easy_flip0(s, :, :) = squeeze(mean(BLVALS.data(:, idx_bl, idx_flip0_easy_flip0), [2, 3]));
    bl_flip0_hard_flip0(s, :, :) = squeeze(mean(BLVALS.data(:, idx_bl, idx_flip0_hard_flip0), [2, 3]));
    bl_flip1_easy_flip0(s, :, :) = squeeze(mean(BLVALS.data(:, idx_bl, idx_flip1_easy_flip0), [2, 3]));
    bl_flip1_hard_flip0(s, :, :) = squeeze(mean(BLVALS.data(:, idx_bl, idx_flip1_hard_flip0), [2, 3]));
    bl_flip1_easy_flip1(s, :, :) = squeeze(mean(BLVALS.data(:, idx_bl, idx_flip1_easy_flip1), [2, 3]));
    bl_flip1_hard_flip1(s, :, :) = squeeze(mean(BLVALS.data(:, idx_bl, idx_flip1_hard_flip1), [2, 3]));   

    % Calculate subject ERPs by averaging across trials for each condition.
    erp_flip0_easy_flip0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip0_easy_flip0)), 3);
    erp_flip0_hard_flip0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip0_hard_flip0)), 3);
    erp_flip1_easy_flip0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_easy_flip0)), 3);
    erp_flip1_hard_flip0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_hard_flip0)), 3);
    erp_flip1_easy_flip1(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_easy_flip1)), 3);
    erp_flip1_hard_flip1(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_hard_flip1)), 3);

    % Subtract baseline
    for ch = 1 : EEG.nbchan
        
        erp_flip0_easy_flip0(s, ch, :) = squeeze(erp_flip0_easy_flip0(s, ch, :)) - squeeze(bl_flip0_easy_flip0(s, ch, :));
        erp_flip0_hard_flip0(s, ch, :) = squeeze(erp_flip0_hard_flip0(s, ch, :)) - squeeze(bl_flip0_hard_flip0(s, ch, :));
        erp_flip1_easy_flip0(s, ch, :) = squeeze(erp_flip1_easy_flip0(s, ch, :)) - squeeze(bl_flip1_easy_flip0(s, ch, :));
        erp_flip1_hard_flip0(s, ch, :) = squeeze(erp_flip1_hard_flip0(s, ch, :)) - squeeze(bl_flip1_hard_flip0(s, ch, :));
        erp_flip1_easy_flip1(s, ch, :) = squeeze(erp_flip1_easy_flip1(s, ch, :)) - squeeze(bl_flip1_easy_flip1(s, ch, :));
        erp_flip1_hard_flip1(s, ch, :) = squeeze(erp_flip1_hard_flip1(s, ch, :)) - squeeze(bl_flip1_hard_flip1(s, ch, :)); 

    end

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
erp_flip0_easy_flip0 = erp_flip0_easy_flip0(:, new_order_idx, :);
erp_flip0_hard_flip0 = erp_flip0_hard_flip0(:, new_order_idx, :);
erp_flip1_easy_flip0 = erp_flip1_easy_flip0(:, new_order_idx, :);
erp_flip1_hard_flip0 = erp_flip1_hard_flip0(:, new_order_idx, :);
erp_flip1_easy_flip1 = erp_flip1_easy_flip1(:, new_order_idx, :);
erp_flip1_hard_flip1 = erp_flip1_hard_flip1(:, new_order_idx, :);
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

% GA struct easy noflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip0_easy_flip0(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_easy_noflip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct easy flip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip1_easy_flip0(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_easy_flip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct easy flipflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip1_easy_flip1(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_easy_flipflip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct hard noflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip0_hard_flip0(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_hard_noflip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct hard flip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip1_hard_flip0(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_hard_flip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct hard flipflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip1_hard_flip1(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_hard_flipflip = ft_timelockgrandaverage(cfg, GA{1, :});

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
[stat_agency_easy] = ft_timelockstatistics(cfg, GA_easy_noflip, GA_easy_flip);
[stat_agency_hard] = ft_timelockstatistics(cfg, GA_hard_noflip, GA_hard_flip);
[stat_agency_sequence_easy] = ft_timelockstatistics(cfg, GA_easy_flip, GA_easy_flipflip);
[stat_agency_sequence_hard] = ft_timelockstatistics(cfg, GA_hard_flip, GA_hard_flipflip);

% Save masks
dlmwrite([PATH_VEUSZ, 'contour_agency_easy.csv'], stat_agency_easy.mask);
dlmwrite([PATH_VEUSZ, 'contour_agency_hard.csv'], stat_agency_hard.mask);

% Calculate effect sizes
n_chans = numel(chanlocs);
apes_agency_easy = [];
apes_agency_hard = [];
apes_agency_sequence_easy = [];
apes_agency_sequence_hard = [];
df_effect = 1;
for ch = 1 : n_chans
    petasq = (squeeze(stat_agency_easy.stat(ch, :)) * df_effect) ./ ((squeeze(stat_agency_easy.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_agency_easy(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
    petasq = (squeeze(stat_agency_hard.stat(ch, :)) * df_effect) ./ ((squeeze(stat_agency_hard.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_agency_hard(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
    petasq = (squeeze(stat_agency_sequence_easy.stat(ch, :)) * df_effect) ./ ((squeeze(stat_agency_sequence_easy.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_agency_sequence_easy(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
    petasq = (squeeze(stat_agency_sequence_hard.stat(ch, :)) * df_effect) ./ ((squeeze(stat_agency_sequence_hard.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_agency_sequence_hard(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
end

% Plot masks
figure()
subplot(2, 1, 1)
contourf(erp_times,[1:65], apes_agency_easy, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_agency_easy.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('agency state easy')

subplot(2, 1, 2)
contourf(erp_times,[1:65], apes_agency_hard, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_agency_hard.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('agency state hard')

% Plot masks
figure()
subplot(2, 1, 1)
contourf(erp_times,[1:65], apes_agency_sequence_easy, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_agency_sequence_easy.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('agency sequence easy')

subplot(2, 1, 2)
contourf(erp_times,[1:65], apes_agency_sequence_hard, 50, 'LineColor', 'none')
caxis([-0.5, 0.5])
colormap(jet)
hold on
contour(erp_times,[1:65], stat_agency_sequence_hard.mask, 'levels', 1, 'LineColor', 'k', 'LineWidth', 1.5)
title('agency sequence hard')

% Save effect sizes
dlmwrite([PATH_VEUSZ, 'apes_agency_easy.csv'], apes_agency_easy);
dlmwrite([PATH_VEUSZ, 'apes_agency_hard.csv'], apes_agency_hard);

% Save lineplots at Fz
dlmwrite([PATH_VEUSZ, 'lineplots_fz.csv'],  [mean(squeeze(erp_flip0_easy_flip0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_flip0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_flip0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_flip0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_flip1(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_flip1(:, 11, :)), 1)]);

% Save lineplots at FCz
dlmwrite([PATH_VEUSZ, 'lineplots_fcz.csv'],  [mean(squeeze(erp_flip0_easy_flip0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_flip0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_flip0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_flip0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_flip1(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_flip1(:, 20, :)), 1)]);

% Save lineplots at Cz
dlmwrite([PATH_VEUSZ, 'lineplots_cz.csv'],  [mean(squeeze(erp_flip0_easy_flip0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_flip0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_flip0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_flip0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_flip1(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_flip1(:, 29, :)), 1)]);

% Save lineplots at CPz
dlmwrite([PATH_VEUSZ, 'lineplots_cpz.csv'],  [mean(squeeze(erp_flip0_easy_flip0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_flip0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_flip0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_flip0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_flip1(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_flip1(:, 38, :)), 1)]);
% Save lineplots at Pz
dlmwrite([PATH_VEUSZ, 'lineplots_pz.csv'],  [mean(squeeze(erp_flip0_easy_flip0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_flip0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_flip0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_flip0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_flip1(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_flip1(:, 48, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_VEUSZ, 'lineplots_poz.csv'],  [mean(squeeze(erp_flip0_easy_flip0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_flip0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_flip0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_flip0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_flip1(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_flip1(:, 57, :)), 1)]);


% Save erp-times
dlmwrite([PATH_VEUSZ, 'erp_times.csv'], erp_times);

% Plot erps
pd_fz = [mean(squeeze(erp_flip0_easy_flip0(:, 11, :)), 1);...
         mean(squeeze(erp_flip0_hard_flip0(:, 11, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip0(:, 11, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip0(:, 11, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip1(:, 11, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip1(:, 11, :)), 1)];

pd_fcz = [mean(squeeze(erp_flip0_easy_flip0(:, 20, :)), 1);...
          mean(squeeze(erp_flip0_hard_flip0(:, 20, :)), 1);...
          mean(squeeze(erp_flip1_easy_flip0(:, 20, :)), 1);...
          mean(squeeze(erp_flip1_hard_flip0(:, 20, :)), 1);...
          mean(squeeze(erp_flip1_easy_flip1(:, 20, :)), 1);...
          mean(squeeze(erp_flip1_hard_flip1(:, 20, :)), 1)];

pd_cz = [mean(squeeze(erp_flip0_easy_flip0(:, 29, :)), 1);...
         mean(squeeze(erp_flip0_hard_flip0(:, 29, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip0(:, 29, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip0(:, 29, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip1(:, 29, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip1(:, 29, :)), 1)];

pd_cpz = [mean(squeeze(erp_flip0_easy_flip0(:, 38, :)), 1);...
         mean(squeeze(erp_flip0_hard_flip0(:, 38, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip0(:, 38, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip0(:, 38, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip1(:, 38, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip1(:, 38, :)), 1)];

pd_pz = [mean(squeeze(erp_flip0_easy_flip0(:, 48, :)), 1);...
         mean(squeeze(erp_flip0_hard_flip0(:, 48, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip0(:, 48, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip0(:, 48, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip1(:, 48, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip1(:, 48, :)), 1)];

pd_poz = [mean(squeeze(erp_flip0_easy_flip0(:, 57, :)), 1);...
         mean(squeeze(erp_flip0_hard_flip0(:, 57, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip0(:, 57, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip0(:, 57, :)), 1);...
         mean(squeeze(erp_flip1_easy_flip1(:, 57, :)), 1);...
         mean(squeeze(erp_flip1_hard_flip1(:, 57, :)), 1)];

figure()
subplot(3, 2, 1)
plot(erp_times, pd_fz([1,3,5], :), 'LineWidth', 1.5)
title('easy Fz')
xlim([-200, 1000])
subplot(3, 2, 2)
plot(erp_times, pd_fcz([1,3,5], :), 'LineWidth', 1.5)
title('FCz')
xlim([-200, 1000])
subplot(3, 2, 3)
plot(erp_times, pd_cz([1,3,5], :), 'LineWidth', 1.5)
title('Cz')
xlim([-200, 1000])
subplot(3, 2, 4)
plot(erp_times, pd_cpz([1,3,5], :), 'LineWidth', 1.5)
title('CPz')
xlim([-200, 1000])
subplot(3, 2, 5)
plot(erp_times, pd_pz([1,3,5], :), 'LineWidth', 1.5)
title('Pz')
xlim([-200, 1000])
subplot(3, 2, 6)
plot(erp_times, pd_poz([1,3,5], :), 'LineWidth', 1.5)
title('POz')
xlim([-200, 1000])
legend({'00', '10', '11'})

figure()
subplot(3, 2, 1)
plot(erp_times, pd_fz([2, 4, 6], :), 'LineWidth', 1.5)
title('hard Fz')
xlim([-200, 1000])
subplot(3, 2, 2)
plot(erp_times, pd_fcz([2, 4, 6], :), 'LineWidth', 1.5)
title('FCz')
xlim([-200, 1000])
subplot(3, 2, 3)
plot(erp_times, pd_cz([2, 4, 6], :), 'LineWidth', 1.5)
title('Cz')
xlim([-200, 1000])
subplot(3, 2, 4)
plot(erp_times, pd_cpz([2, 4, 6], :), 'LineWidth', 1.5)
title('CPz')
xlim([-200, 1000])
subplot(3, 2, 5)
plot(erp_times, pd_pz([2, 4, 6], :), 'LineWidth', 1.5)
title('Pz')
xlim([-200, 1000])
subplot(3, 2, 6)
plot(erp_times, pd_poz([2, 4, 6], :), 'LineWidth', 1.5)
title('POz')
xlim([-200, 1000])
legend({'00', '10', '11'})

aa=bb
% Plot effect size topos at selected time points for agency
clim = [-0.3, 0.3];
tpoints = [100, 280, 800];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_agency(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_agency_', num2str(tpoints(t)), 'ms', '.png']);
end

% Plot effect size topos at selected time points for difficulty
clim = [-0.3, 0.3];
tpoints = [0, 300, 500];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_difficulty(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_difficulty_', num2str(tpoints(t)), 'ms', '.png']);
end

% Plot effect size topos at selected time points for interaction
clim = [-0.3, 0.3];
tpoints = [300, 600, 700];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_interaction(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_interaction_', num2str(tpoints(t)), 'ms', '.png']);
end