clear all;

% PATH VARS - PLEASE ADJUST!!!!!
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP12', 'VP07',...
                'VP11', 'VP09', 'VP16', 'VP17', 'VP19', 'VP21', 'VP23', 'VP25',...
                'VP27', 'VP29', 'VP31', 'VP18', 'VP20', 'VP22', 'VP24', 'VP26',...
                'VP28', 'VP13', 'VP15'};

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
frontal_channel_idx = [65, 19, 20];

% Calculate frontal ERPs. Data is then subject x times dimensionality.
erp_frontal_easy_accu = squeeze(mean(erp_easy_accu(:, frontal_channel_idx, :), 2));
erp_frontal_easy_flip = squeeze(mean(erp_easy_flip(:, frontal_channel_idx, :), 2));
erp_frontal_hard_accu = squeeze(mean(erp_hard_accu(:, frontal_channel_idx, :), 2));
erp_frontal_hard_flip = squeeze(mean(erp_hard_flip(:, frontal_channel_idx, :), 2));

% Define time window for CNV-parameterization
cnv_times_1 = [700, 900];
cnv_times_2 = [900, 1100];

% Get cnv-time indices
cnv_time_idx_1 = EEG.times >= cnv_times_1(1) & EEG.times <= cnv_times_1(2);
cnv_time_idx_2 = EEG.times >= cnv_times_2(1) & EEG.times <= cnv_times_2(2);

%% ========================= PLOTTING ======================================================================================================================

% Plot frontal ERPs
figure()
plot(EEG.times, mean(erp_frontal_easy_accu, 1), 'k-', 'LineWidth', 2)
hold on;
plot(EEG.times, mean(erp_frontal_easy_flip, 1), 'k:', 'LineWidth', 2)
plot(EEG.times, mean(erp_frontal_hard_accu, 1), 'm-', 'LineWidth', 2)
plot(EEG.times, mean(erp_frontal_hard_flip, 1), 'm:', 'LineWidth', 2)
legend({'easy accu', 'easy flip', 'hard accu', 'hard flip'})
title('ERP at: FCz, FC1, FC2')
xline([0, 1200])
ylims = [-5, 3];
ylim(ylims)
xlim([-200, 2000])
rectangle('Position', [cnv_times_1(1), ylims(1), cnv_times_1(2) - cnv_times_1(1), ylims(2) - ylims(1)], 'FaceColor',[0.5, 1, 0.5, 0.3], 'EdgeColor', 'none')
rectangle('Position', [cnv_times_2(1), ylims(1), cnv_times_2(2) - cnv_times_2(1), ylims(2) - ylims(1)], 'FaceColor',[0.5, 0.5, 1, 0.3], 'EdgeColor', 'none')

% Average over subjects and cnv-times to plot topographies
topo_easy_accu_1 = squeeze(mean(erp_easy_accu(:, :, cnv_time_idx_1), [1, 3]));
topo_easy_flip_1 = squeeze(mean(erp_easy_flip(:, :, cnv_time_idx_1), [1, 3]));
topo_hard_accu_1 = squeeze(mean(erp_hard_accu(:, :, cnv_time_idx_1), [1, 3]));
topo_hard_flip_1 = squeeze(mean(erp_hard_flip(:, :, cnv_time_idx_1), [1, 3]));
topo_easy_accu_2 = squeeze(mean(erp_easy_accu(:, :, cnv_time_idx_2), [1, 3]));
topo_easy_flip_2 = squeeze(mean(erp_easy_flip(:, :, cnv_time_idx_2), [1, 3]));
topo_hard_accu_2 = squeeze(mean(erp_hard_accu(:, :, cnv_time_idx_2), [1, 3]));
topo_hard_flip_2 = squeeze(mean(erp_hard_flip(:, :, cnv_time_idx_2), [1, 3]));

% Plot topo timewin 1
figure()
subplot(2, 2, 1)
topoplot(topo_easy_accu_1, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy reliable'], 'FontSize', 10)
subplot(2, 2, 2)
topoplot(topo_easy_flip_1, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy non-reliable'], 'FontSize', 10)
subplot(2, 2, 3)
topoplot(topo_hard_accu_1, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['hard reliable'], 'FontSize', 10)
subplot(2, 2, 4)
topoplot(topo_hard_flip_1, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['hard non-reliable'], 'FontSize', 10)
sgtitle('Topographies CNV (700-900 ms)')

% Plot topo timewin 2
figure()
subplot(2, 2, 1)
topoplot(topo_easy_accu_2, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy reliable'], 'FontSize', 10)
subplot(2, 2, 2)
topoplot(topo_easy_flip_2, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy non-reliable'], 'FontSize', 10)
subplot(2, 2, 3)
topoplot(topo_hard_accu_2, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['hard reliable'], 'FontSize', 10)
subplot(2, 2, 4)
topoplot(topo_hard_flip_2, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['hard non-reliable'], 'FontSize', 10)
sgtitle('Topographies CNV (900-1100 ms)')

%% ========================= STATISTICS ======================================================================================================================

% Average CNV valus in time window for each subject
cnv_values_1 = [mean(erp_frontal_easy_accu(:, cnv_time_idx_1), 2),...
                mean(erp_frontal_easy_flip(:, cnv_time_idx_1), 2),...
                mean(erp_frontal_hard_accu(:, cnv_time_idx_1), 2),...
                mean(erp_frontal_hard_flip(:, cnv_time_idx_1), 2)];
cnv_values_2 = [mean(erp_frontal_easy_accu(:, cnv_time_idx_2), 2),...
                mean(erp_frontal_easy_flip(:, cnv_time_idx_2), 2),...
                mean(erp_frontal_hard_accu(:, cnv_time_idx_2), 2),...
                mean(erp_frontal_hard_flip(:, cnv_time_idx_2), 2)];

% Perform rmANOVA for CNV-values
varnames = {'subject', 'easy_accu', 'easy_flip', 'hard_accu', 'hard_flip'};
t = table(ids', cnv_values_1(:, 1), cnv_values_1(:, 2), cnv_values_1(:, 3), cnv_values_1(:, 4), 'VariableNames', varnames);
within = table({'easy'; 'easy'; 'hard'; 'hard'}, {'accu'; 'flip'; 'accu'; 'flip'}, 'VariableNames', {'difficulty', 'reliability'});
rm = fitrm(t, 'easy_accu-hard_flip~1', 'WithinDesign', within);
anova_cnv_1 = ranova(rm, 'WithinModel', 'difficulty + reliability + difficulty*reliability');
anova_cnv_1

varnames = {'subject', 'easy_accu', 'easy_flip', 'hard_accu', 'hard_flip'};
t = table(ids', cnv_values_2(:, 1), cnv_values_2(:, 2), cnv_values_2(:, 3), cnv_values_2(:, 4), 'VariableNames', varnames);
within = table({'easy'; 'easy'; 'hard'; 'hard'}, {'accu'; 'flip'; 'accu'; 'flip'}, 'VariableNames', {'difficulty', 'reliability'});
rm = fitrm(t, 'easy_accu-hard_flip~1', 'WithinDesign', within);
anova_cnv_2 = ranova(rm, 'WithinModel', 'difficulty + reliability + difficulty*reliability');
anova_cnv_2

