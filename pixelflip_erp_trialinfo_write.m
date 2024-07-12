clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2023.1/';
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

% Loop subjects
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Load subject data. EEG data has dimensionality channels x times x trials
    EEG = pop_loadset('filename',    [subject, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

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

    % Convert trialinfo to table
    trialinfo_table = array2table(trialinfo, 'VariableNames', {'trial_nr', 'block_nr', 'reliability', 'difficulty', 'flipped', ...
                                                               'key_pressed', 'rt', 'color_pressed', 'feedback_accuracy', 'feedback_color', ...
                                                               'accuracy', 'prev_accuracy', 'prev_flipped'});
    writetable(trialinfo_table, [PATH_AUTOCLEANED, subject, '_erp_trialinfo.csv']);


end