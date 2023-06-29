clear all;

% PATH VARS
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP12', 'VP11',...
                'VP09', 'VP16', 'VP17', 'VP19', 'VP21', 'VP23', 'VP25', 'VP27',...
                'VP29', 'VP31', 'VP18', 'VP20', 'VP22', 'VP24', 'VP26'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

%% ========================= CALCULATE ERPs ======================================================================================================================

% Load info
EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Get erp times
erp_times = EEG.times(EEG.times >= -200 & EEG.times <= 1500);

% Matrices to collect data. Dimensionality: subjects x channels x times
erp_easy_accu = zeros(length(subject_list), EEG.nbchan, length(EEG.times));
erp_easy_flip = zeros(length(subject_list), EEG.nbchan, length(EEG.times));
erp_hard_accu = zeros(length(subject_list), EEG.nbchan, length(EEG.times));
erp_hard_flip = zeros(length(subject_list), EEG.nbchan, length(EEG.times));

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
    erp_easy_accu(s, :, :) = mean(squeeze(EEG.data(:, :, idx_easy_accu)), 3);
    erp_easy_flip(s, :, :) = mean(squeeze(EEG.data(:, :, idx_easy_flip)), 3);
    erp_hard_accu(s, :, :) = mean(squeeze(EEG.data(:, :, idx_hard_accu)), 3);
    erp_hard_flip(s, :, :) = mean(squeeze(EEG.data(:, :, idx_hard_flip)), 3);

end

% Select frontal channels
frontal_channel_idx = [65];

% Calculate frontal ERPs. Data is then subject x times dimensionality.
erp_frontal_easy_accu = squeeze(mean(erp_easy_accu(:, frontal_channel_idx, :), 2));
erp_frontal_easy_flip = squeeze(mean(erp_easy_flip(:, frontal_channel_idx, :), 2));
erp_frontal_hard_accu = squeeze(mean(erp_hard_accu(:, frontal_channel_idx, :), 2));
erp_frontal_hard_flip = squeeze(mean(erp_hard_flip(:, frontal_channel_idx, :), 2));

% Define time window for CNV-parameterization
cnv_times = [900, 1200];

% Get cnv-time indices
cnv_time_idx = EEG.times >= cnv_times(1) & EEG.times <= cnv_times(2);

%% ========================= PLOTTING ======================================================================================================================

% Plot frontal ERPs
figure()
plot(EEG.times, mean(erp_frontal_easy_accu, 1), 'k-', 'LineWidth', 2)
hold on;
plot(EEG.times, mean(erp_frontal_easy_flip, 1), 'k:', 'LineWidth', 2)
plot(EEG.times, mean(erp_frontal_hard_accu, 1), 'r-', 'LineWidth', 2)
plot(EEG.times, mean(erp_frontal_hard_flip, 1), 'r:', 'LineWidth', 2)
legend({'easy accu', 'easy flip', 'hard accu', 'hard flip'})
title('frontal channels')
xline([0, 1200])
ylims = [-5, 3];
ylim(ylims)
xlim([-200, 2000])
rectangle('Position', [cnv_times(1), ylims(1), cnv_times(2) - cnv_times(1), ylims(2) - ylims(1)], 'FaceColor',[0.5, 1, 0.5, 0.2], 'EdgeColor', 'none')

% Average over subjects and cnv-times to plot topographies
topo_easy_accu = squeeze(mean(erp_easy_accu(:, :, cnv_time_idx), [1, 3]));
topo_easy_flip = squeeze(mean(erp_easy_flip(:, :, cnv_time_idx), [1, 3]));
topo_hard_accu = squeeze(mean(erp_hard_accu(:, :, cnv_time_idx), [1, 3]));
topo_hard_flip = squeeze(mean(erp_hard_flip(:, :, cnv_time_idx), [1, 3]));

% Create difference topographies for main-effects
topo_difficulty = ((topo_hard_accu + topo_hard_flip) / 2) - ((topo_easy_accu + topo_easy_flip) / 2);
topo_reliability = ((topo_easy_accu + topo_hard_accu) / 2) - ((topo_easy_flip + topo_hard_flip) / 2) ;

% Plot topos
figure()

subplot(3, 2, 1)
topoplot(topo_easy_accu, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy accu'], 'FontSize', 10)

subplot(3, 2, 2)
topoplot(topo_easy_flip, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy flip'], 'FontSize', 10)

subplot(3, 2, 3)
topoplot(topo_hard_accu, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['hard accu'], 'FontSize', 10)

subplot(3, 2, 4)
topoplot(topo_hard_flip, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['hard flip'], 'FontSize', 10)

subplot(3, 2, 5)
topoplot(topo_difficulty, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-1, 1])
colorbar;
title(['hard - easy'], 'FontSize', 10)

subplot(3, 2, 6)
topoplot(topo_reliability, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-1, 1])
colorbar;
title(['accu - flip'], 'FontSize', 10)

%% ========================= STATISTICS ======================================================================================================================

% Average CNV valus in time window for each subject
cnv_values = [mean(erp_frontal_easy_accu(:, cnv_time_idx), 2),...
              mean(erp_frontal_easy_flip(:, cnv_time_idx), 2),...
              mean(erp_frontal_hard_accu(:, cnv_time_idx), 2),...
              mean(erp_frontal_hard_flip(:, cnv_time_idx), 2)];

% Perform rmANOVA for CNV-values
varnames = {'subject', 'easy_accu', 'easy_flip', 'hard_accu', 'hard_flip'};
t = table(ids', cnv_values(:, 1), cnv_values(:, 2), cnv_values(:, 3), cnv_values(:, 4), 'VariableNames', varnames);
within = table({'easy'; 'easy'; 'hard'; 'hard'}, {'accu'; 'flip'; 'accu'; 'flip'}, 'VariableNames', {'difficulty', 'reliability'});
rm = fitrm(t, 'easy_accu-hard_flip~1', 'WithinDesign', within);
anova_cnv = ranova(rm, 'WithinModel', 'difficulty + reliability + difficulty*reliability');
anova_cnv

