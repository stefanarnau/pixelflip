clear all;

% PATH VARS - PLEASE ADJUST!!!!!
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_RATINGS     = '/mnt/data_dump/pixelflip/veusz/subjecctive_ratings/';  
PATH_VEUSZ       = '/mnt/data_dump/pixelflip/veusz/cue_erp_correlations/';  

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

% Load ratings
load([PATH_RATINGS, 'table_ratings.mat'])

% Load info
EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Get erp times
erp_times_idx = EEG.times >= -200 & EEG.times <= 1400;
erp_times = EEG.times(erp_times_idx);

% Get chanlocs
chanlocs = EEG.chanlocs;

% Matrices to collect data. Dimensionality: subjects x channels x times
erp_asis = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_flip = zeros(length(subject_list), EEG.nbchan, length(erp_times));

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
    % 12: previous flipped
    % 13: previous difficulty

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
    idx_asis = EEG.trialinfo(:, 3) == 0;
    idx_flip = EEG.trialinfo(:, 3) == 1;
    
    % Calculate subject ERPs by averaging across trials for each condition.
    erp_asis(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_asis)), 3);
    erp_flip(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_flip)), 3);

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
erp_asis = erp_asis(:, new_order_idx, :);
erp_flip = erp_flip(:, new_order_idx, :);
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

% GA struct asis trials
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_asis(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_asis = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct flip trials
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_flip(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_flip = ft_timelockgrandaverage(cfg, GA{1, :});

% GA struct diferences
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(erp_asis(s, :, :)) - squeeze(erp_flip(s, :, :));
    ga_template.avg = chan_time_data;
    GA{s} = ga_template;
end 
GA_diff = ft_timelockgrandaverage(cfg, GA{1, :});


% Correlation motivation
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
[stat] = ft_timelockstatistics(cfg, GA_diff);

% Save mask and rho
dlmwrite([PATH_VEUSZ, 'rho_motivation.csv'], stat.rho);
dlmwrite([PATH_VEUSZ, 'mask:motivation.csv'], stat.mask);

figure()
contourf( erp_times,[1 : 65], stat.rho, 50, 'linecolor','none')
colorbar()
hold on
contour( erp_times,[1 : 65], stat.mask, 1, 'linecolor', 'k', 'LineWidth', 2)
colormap('jet')

% Correlation motivation
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
[stat] = ft_timelockstatistics(cfg, GA_diff);

% Save mask and rho
dlmwrite([PATH_VEUSZ, 'rho_motivation.csv'], stat.rho);
dlmwrite([PATH_VEUSZ, 'mask:motivation.csv'], stat.mask);

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