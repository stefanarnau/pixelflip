clear all;

% PATH VARS
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_RESULTS     = '/mnt/data_dump/pixelflip/results_jana/';

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

% Load info
EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Get erp times
erp_times_idx = EEG.times >= -200 & EEG.times <= 1800;
erp_times = EEG.times(erp_times_idx);

%% ========================= CALCULATE ERPs ======================================================================================================================

% Load info
EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

% Matrices to collect data. Dimensionality: subjects x channels x times
erp_easy_asis = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_easy_flip = zeros(length(subject_list), EEG.nbchan, length(erp_times));
erp_hard_asis = zeros(length(subject_list), EEG.nbchan, length(erp_times));
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

            % If not a flip block
            else

                % Not a good trial...
                EEG.trialinfo(e, 12) = -1;
                EEG.trialinfo(e, 13) = -1;

                % Next
                continue;

            end

        % If no flip block
        else

            % Not a good trial...
            EEG.trialinfo(e, 12) = -1;
            EEG.trialinfo(e, 13) = -1;

        end
    end
    
    % Get trial-indices of conditions
    idx_easy_asis = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 1;
    idx_easy_flip = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 0;
    idx_hard_asis = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 1;
    idx_hard_flip = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 0;
    
    % Calculate subject ERPs by averaging across trials for each condition.
    erp_easy_asis(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_easy_asis)), 3);
    erp_easy_flip(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_easy_flip)), 3);
    erp_hard_asis(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_hard_asis)), 3);
    erp_hard_flip(s, :, :) = mean(squeeze(EEG.data(:, erp_times_idx, idx_hard_flip)), 3);

end

% Select frontal channels
frontal_channel_idx = [65, 15];

% Calculate frontal ERPs. Data is then subject x times dimensionality.
erp_frontal_easy_asis = squeeze(mean(erp_easy_asis(:, frontal_channel_idx, :), 2));
erp_frontal_easy_flip = squeeze(mean(erp_easy_flip(:, frontal_channel_idx, :), 2));
erp_frontal_hard_asis = squeeze(mean(erp_hard_asis(:, frontal_channel_idx, :), 2));
erp_frontal_hard_flip = squeeze(mean(erp_hard_flip(:, frontal_channel_idx, :), 2));

% Define time window for CNV-parameterization
cnv_times = [600, 1100];

% Get cnv-time indices
cnv_time_idx = erp_times >= cnv_times(1) & erp_times <= cnv_times(2);

%% ========================= PLOTTING ======================================================================================================================

% Plot frontal ERPs
figure()
plot(erp_times, mean(erp_frontal_easy_asis, 1), 'k-', 'LineWidth', 2)
hold on;
plot(erp_times, mean(erp_frontal_easy_flip, 1), 'k:', 'LineWidth', 2)
plot(erp_times, mean(erp_frontal_hard_asis, 1), 'r-', 'LineWidth', 2)
plot(erp_times, mean(erp_frontal_hard_flip, 1), 'r:', 'LineWidth', 2)
legend({'easy non-flip', 'easy flip', 'hard non-flip', 'hard flip'})
title('Fz FCz FC1 FC2')
xline([0, 1200])
ylims = [-5, 3];
ylim(ylims)
xlim([-200, 1800])
rectangle('Position', [cnv_times(1), ylims(1), cnv_times(2) - cnv_times(1), ylims(2) - ylims(1)], 'FaceColor',[0.5, 1, 0.5, 0.2], 'EdgeColor', 'none')

% Average over subjects and cnv-times to plot topographies
topo_easy_asis= squeeze(mean(erp_easy_asis(:, :, cnv_time_idx), [1, 3]));
topo_easy_flip = squeeze(mean(erp_easy_flip(:, :, cnv_time_idx), [1, 3]));
topo_hard_asis= squeeze(mean(erp_hard_asis(:, :, cnv_time_idx), [1, 3]));
topo_hard_flip = squeeze(mean(erp_hard_flip(:, :, cnv_time_idx), [1, 3]));

% Create difference topographies for main-effects
topo_difficulty = ((topo_hard_asis+ topo_hard_flip) / 2) - ((topo_easy_asis+ topo_easy_flip) / 2);
topo_reliability = ((topo_easy_asis+ topo_hard_asis) / 2) - ((topo_easy_flip + topo_hard_flip) / 2) ;

% Plot topos
figure()

subplot(3, 2, 1)
topoplot(topo_easy_asis, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy non-flip'], 'FontSize', 10)

subplot(3, 2, 2)
topoplot(topo_easy_flip, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['easy flip'], 'FontSize', 10)

subplot(3, 2, 3)
topoplot(topo_hard_asis, EEG.chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
colormap('jet')
set(gca, 'clim', [-3, 3])
colorbar;
title(['hard non-flip'], 'FontSize', 10)

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
title(['non-flip - flip'], 'FontSize', 10)

%% ========================= STATISTICS ======================================================================================================================

% Average CNV valus in time window for each subject
cnv_values = [mean(erp_frontal_easy_asis(:, cnv_time_idx), 2),...
              mean(erp_frontal_easy_flip(:, cnv_time_idx), 2),...
              mean(erp_frontal_hard_asis(:, cnv_time_idx), 2),...
              mean(erp_frontal_hard_flip(:, cnv_time_idx), 2)];

% Perform rmANOVA for CNV-values
varnames = {'subject', 'easy_asis', 'easy_flip', 'hard_asis', 'hard_flip'};
t = table(ids', cnv_values(:, 1), cnv_values(:, 2), cnv_values(:, 3), cnv_values(:, 4), 'VariableNames', varnames);
within = table({'easy'; 'easy'; 'hard'; 'hard'}, {'asis'; 'flip'; 'asis'; 'flip'}, 'VariableNames', {'difficulty', 'reliability'});
rm = fitrm(t, 'easy_asis-hard_flip~1', 'WithinDesign', within);
anova_cnv = ranova(rm, 'WithinModel', 'difficulty + reliability + difficulty*reliability');
anova_cnv

% Convert to long format and save as csv
res = [];
counter = 0;
for s = 1 : size(cnv_values, 1)

    % Get subject id as string
    subject = subject_list{s};

    % Collect IDs as number
    id = str2num(subject(3 : 4));

    % Loop condition means
    for cond = 1 : 4

        counter = counter + 1;

        % Set levels
        if cond < 3
            difficulty = 0;
        else
            difficulty = 1;
        end
        if mod(cond, 2) == 0
            flip = 0;
        else
            flip = 1;
        end

        % Fill
        res(counter, :) = [id, difficulty, flip, cnv_values(s, cond)];

    end
end

writematrix(res, [PATH_RESULTS, 'cnv_values.csv']);

