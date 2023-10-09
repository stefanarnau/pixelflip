clear all;

% PATH VARS - PLEASE ADJUST!!!!!
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_VEUSZ       = '/mnt/data_dump/pixelflip/veusz/cue_erp_sequence/';  

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
erp_flip0_easy_post0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip0_hard_post0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_easy_post0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_hard_post0 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_easy_post1 = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip1_hard_post1 = zeros(length(subject_list), EEG.nbchan, length(erp_times));

% Loop subjects
ids = [];
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Collect IDs as number
    ids(s) = str2num(subject(3 : 4));

    % Load subject data. EEG data has dimensionality channels x times x trials
    EEG = pop_loadset('filename',    [subject, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

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
    
    % Get trial-indices of conditions
    idx_flip0_easy_post0 = EEG.trialinfo(:, 3) == 1 & EEG.trialinfo(:, 4) == 0;
    idx_flip0_hard_post0 = EEG.trialinfo(:, 3) == 1 & EEG.trialinfo(:, 4) == 1;
    idx_flip1_easy_post0 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 12) == 0;
    idx_flip1_hard_post0 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 12) == 0;
    idx_flip1_easy_post1 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 12) == 1;
    idx_flip1_hard_post1 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 12) == 1;

    % Calculate subject ERPs by averaging across trials for each condition.
    erp_flip0_easy_post0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip0_easy_post0)), 3);
    erp_flip0_hard_post0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip0_hard_post0)), 3);
    erp_flip1_easy_post0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_easy_post0)), 3);
    erp_flip1_hard_post0(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_hard_post0)), 3);
    erp_flip1_easy_post1(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_easy_post1)), 3);
    erp_flip1_hard_post1(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip1_hard_post1)), 3);  

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
erp_flip0_easy_post0 = erp_flip0_easy_post0(:, new_order_idx, :);
erp_flip0_hard_post0 = erp_flip0_hard_post0(:, new_order_idx, :);
erp_flip1_easy_post0 = erp_flip1_easy_post0(:, new_order_idx, :);
erp_flip1_hard_post0 = erp_flip1_hard_post0(:, new_order_idx, :);
erp_flip1_easy_post1 = erp_flip1_easy_post1(:, new_order_idx, :);
erp_flip1_hard_post1 = erp_flip1_hard_post1(:, new_order_idx, :);
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


% ############################### flip-postnoflip vs. flip-postflip ###########################################

% GA struct easy
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_flip1_easy_post1(s, :, :)) + squeeze(erp_flip1_easy_post0(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_easy = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct hard
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_flip1_hard_post1(s, :, :)) + squeeze(erp_flip1_hard_post0(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_hard = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct flip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_flip1_easy_post0(s, :, :)) + squeeze(erp_flip1_hard_post0(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_noflip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct noflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(erp_flip1_easy_post1(s, :, :)) + squeeze(erp_flip1_hard_post1(s, :, :))) ./ 2;
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_flip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct hard minus easy flip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip1_hard_post0(s, :, :)) - squeeze(erp_flip1_easy_post0(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_interaction_noflip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct hard minus easy noflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip1_hard_post1(s, :, :)) - squeeze(erp_flip1_easy_post1(s, :, :));
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
[stat_agency]      = ft_timelockstatistics(cfg, GA_noflip, GA_flip);
[stat_interaction] = ft_timelockstatistics(cfg, GA_interaction_noflip, GA_interaction_flip);

% Save masks
dlmwrite([PATH_VEUSZ, 'contour_difficulty.csv'], stat_difficulty.mask);
dlmwrite([PATH_VEUSZ, 'contour_agency.csv'], stat_agency.mask);
dlmwrite([PATH_VEUSZ, 'contour_interaction.csv'], stat_interaction.mask);

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

% Save lineplots at Fz
dlmwrite([PATH_VEUSZ, 'lineplots_fz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post1(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post1(:, 11, :)), 1)]);

% Save lineplots at FCz
dlmwrite([PATH_VEUSZ, 'lineplots_fcz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post1(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post1(:, 20, :)), 1)]);
% Save lineplots at Fz
dlmwrite([PATH_VEUSZ, 'lineplots_cz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post1(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post1(:, 29, :)), 1)]);

% Save lineplots at FCz
dlmwrite([PATH_VEUSZ, 'lineplots_cpz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post1(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post1(:, 38, :)), 1)]);

% Save lineplots at Pz
dlmwrite([PATH_VEUSZ, 'lineplots_pz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post1(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post1(:, 48, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_VEUSZ, 'lineplots_poz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post1(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post1(:, 57, :)), 1)]);


% Save erp-times
dlmwrite([PATH_VEUSZ, 'erp_times.csv'], erp_times);

% Plot effect size topos at selected time points for agency
clim = [-0.3, 0.3];
tpoints = [550, 900, 1400];
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
tpoints = [120, 420, 1600];
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
tpoints = [150, 450, 1700];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_interaction(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_interaction_', num2str(tpoints(t)), 'ms', '.png']);
end