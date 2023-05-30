clear all;

% PATH VARS
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP12', 'VP07', 'VP11', 'VP09', 'VP16', 'VP17'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% SWITCH: Switch parts of script on/off
to_execute = {'part1'};

% Result matrices
rts_all = [];
acc_all = [];
rts_correct = [];

% Part 1: Calculate ersp
if ismember('part1', to_execute)

    % Loop subjects
    for s = 1 : length(subject_list)

        % Get id stuff
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject, '_cleaned_cue_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

        % To double precision
        eeg_data = double(EEG.data);

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

        % Get condition idx
        idx_easy_accu = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 0;
        idx_easy_flip = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 1;
        idx_hard_accu = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 0;
        idx_hard_flip = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 1;

        % Get correct idx
        idx_correct = EEG.trialinfo(:, 11) == 1;

        % Get mean accuracy
        acc_easy_accu = sum(EEG.trialinfo(idx_easy_accu, 11) == 1) / sum(idx_easy_accu);
        acc_easy_flip = sum(EEG.trialinfo(idx_easy_flip, 11) == 1) / sum(idx_easy_flip);
        acc_hard_accu = sum(EEG.trialinfo(idx_hard_accu, 11) == 1) / sum(idx_hard_accu);
        acc_hard_flip = sum(EEG.trialinfo(idx_hard_flip, 11) == 1) / sum(idx_hard_flip);
        acc_all(s, :) = [acc_easy_accu, acc_easy_flip, acc_hard_accu, acc_hard_flip];

        % Get mean RT all trials
        rt_easy_accu = nanmean(EEG.trialinfo(idx_easy_accu, 7));
        rt_easy_flip = nanmean(EEG.trialinfo(idx_easy_flip, 7));
        rt_hard_accu = nanmean(EEG.trialinfo(idx_hard_accu, 7));
        rt_hard_flip = nanmean(EEG.trialinfo(idx_hard_flip, 7));
        rts_all(s, :) = [rt_easy_accu, rt_easy_flip, rt_hard_accu, rt_hard_flip];

        % Get mean RT of correct trials
        rt_easy_accu = nanmean(EEG.trialinfo(idx_easy_accu & idx_correct, 7));
        rt_easy_flip = nanmean(EEG.trialinfo(idx_easy_flip & idx_correct, 7));
        rt_hard_accu = nanmean(EEG.trialinfo(idx_hard_accu & idx_correct, 7));
        rt_hard_flip = nanmean(EEG.trialinfo(idx_hard_flip & idx_correct, 7));
        rts_correct(s, :) = [rt_easy_accu, rt_easy_flip, rt_hard_accu, rt_hard_flip];

    end % End subject iteration

    figure()

    subplot(2, 2, 1)
    bar([1 : 4], mean(acc_all, 1))
    title('accuracy')

    subplot(2, 2, 2)
    bar([1 : 4], mean(rts_all, 1))
    title('rt - all trials')

    subplot(2, 2, 3)
    bar([1 : 4], mean(rts_correct, 1))
    title('rt - correct trials')


end % End part1
