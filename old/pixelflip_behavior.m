clear all;

% PATH VARS - PLEASE ADJUST!!!!!
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_OUT         = '/mnt/data_dump/pixelflip/veusz/behavior/';  

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

% SWITCH: Switch parts of script on/off
to_execute = {'part1'};

% Result matrices
rts_all = [];
acc_all = [];
rts_correct = [];

% Part 1: Calculate ersp
if ismember('part1', to_execute)

    % Loop subjects
    ids = [];
    for s = 1 : length(subject_list)

        % Get id stuff
        subject = subject_list{s};

        % Collect IDs as number
        ids(s) = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

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
        idx_easy_accu = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 1;
        idx_easy_flip = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 0;
        idx_hard_accu = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 1;
        idx_hard_flip = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 0;

        % Get correct idx
        idx_correct = EEG.trialinfo(:, 11) == 1;

        % Get mean accuracy
        acc_easy_accu = sum(EEG.trialinfo(idx_easy_accu, 11) == 1) / sum(idx_easy_accu);
        acc_easy_flip = sum(EEG.trialinfo(idx_easy_flip, 11) == 1) / sum(idx_easy_flip);
        acc_hard_accu = sum(EEG.trialinfo(idx_hard_accu, 11) == 1) / sum(idx_hard_accu);
        acc_hard_flip = sum(EEG.trialinfo(idx_hard_flip, 11) == 1) / sum(idx_hard_flip);
        acc_all(s, :) = [acc_easy_accu, acc_easy_flip, acc_hard_accu, acc_hard_flip];

        % Get mean RT of correct trials
        rt_easy_accu = nanmean(EEG.trialinfo(idx_easy_accu & idx_correct, 7));
        rt_easy_flip = nanmean(EEG.trialinfo(idx_easy_flip & idx_correct, 7));
        rt_hard_accu = nanmean(EEG.trialinfo(idx_hard_accu & idx_correct, 7));
        rt_hard_flip = nanmean(EEG.trialinfo(idx_hard_flip & idx_correct, 7));
        rts_correct(s, :) = [rt_easy_accu, rt_easy_flip, rt_hard_accu, rt_hard_flip];

    end % End subject iteration

    figure()

    subplot(1, 2, 1)
    bar([1 : 4], mean(acc_all, 1))
    set(gca,'xticklabel', {'easy accu', 'easy flip', 'hard accu', 'hard flip'})
    ylim([0.4, 1.2])
    title('accuracy')

    subplot(1, 2, 2)
    bar([1 : 4], mean(rts_correct, 1))
    set(gca,'xticklabel', {'easy accu', 'easy flip', 'hard accu', 'hard flip'})
    ylim([400, 650])
    title('rt - correct trials')

    % Perform rmANOVA for accuracy
    varnames = {'subject', 'easy_accu', 'easy_flip', 'hard_accu', 'hard_flip'};
    t = table(ids', acc_all(:, 1), acc_all(:, 2), acc_all(:, 3), acc_all(:, 4), 'VariableNames', varnames);
    within = table({'easy'; 'easy'; 'hard'; 'hard'}, {'accu'; 'flip'; 'accu'; 'flip'}, 'VariableNames', {'difficulty', 'reliability'});
    rm = fitrm(t, 'easy_accu-hard_flip~1', 'WithinDesign', within);
    anova_acc = ranova(rm, 'WithinModel', 'difficulty + reliability + difficulty*reliability');
    anova_acc


    % Perform rmANOVA for RT
    varnames = {'subject', 'easy_accu', 'easy_flip', 'hard_accu', 'hard_flip'};
    t = table(ids', rts_correct(:, 1), rts_correct(:, 2), rts_correct(:, 3), rts_correct(:, 4), 'VariableNames', varnames);
    within = table({'easy'; 'easy'; 'hard'; 'hard'}, {'accu'; 'flip'; 'accu'; 'flip'}, 'VariableNames', {'difficulty', 'reliability'});
    rm = fitrm(t, 'easy_accu-hard_flip~1', 'WithinDesign', within);
    anova_rt = ranova(rm, 'WithinModel', 'difficulty + reliability + difficulty*reliability');
    anova_rt


    % Save behavioral data for veusz
    rt_mean = mean(rts_correct, 1);
    rt_sd = std(rts_correct, [], 1);
    rt_out = [rt_mean(1), rt_sd(1), rt_mean(3), rt_sd(3); rt_mean(2), rt_sd(2), rt_mean(4), rt_sd(4)];
    dlmwrite([PATH_OUT, 'rt_veusz.csv'], rt_out, 'delimiter', '\t');

    acc_mean = mean(acc_all, 1);
    acc_sd = std(acc_all, [], 1);
    acc_out = [acc_mean(1), acc_sd(1), acc_mean(3), acc_sd(3); acc_mean(2), acc_sd(2), acc_mean(4), acc_sd(4)];
    dlmwrite([PATH_OUT, 'acc_veusz.csv'], acc_out, 'delimiter', '\t');

end % End part1
