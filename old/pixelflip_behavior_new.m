clear all;

% PATH VARS - PLEASE ADJUST!!!!!
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_OUT         = '/mnt/data_dump/pixelflip/6_behavioral_results/';  

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

        % Get trial-indices for factor combinations
        idx_agen00_easy = trialinfo(:, 3) == 1 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 0 & trialinfo(:, 13) == 0;
        idx_agen00_hard = trialinfo(:, 3) == 1 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 1 & trialinfo(:, 13) == 0;
        idx_agen10_easy = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 0 & trialinfo(:, 13) == 0;
        idx_agen10_hard = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 1 & trialinfo(:, 13) == 0;
        idx_agen11_easy = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 0 & trialinfo(:, 13) == 1;
        idx_agen11_hard = trialinfo(:, 3) == 0 & trialinfo(:, 12) == 1 & trialinfo(:, 4) == 1 & trialinfo(:, 13) == 1;

        % Get correct idx
        idx_correct = EEG.trialinfo(:, 11) == 1;

        % Get mean accuracy
        acc_flip0_easy_post0 = sum(EEG.trialinfo(idx_agen00_easy, 11) == 1) / sum(idx_agen00_easy);
        acc_flip0_hard_post0 = sum(EEG.trialinfo(idx_agen00_hard, 11) == 1) / sum(idx_agen00_hard);
        acc_flip1_easy_post0 = sum(EEG.trialinfo(idx_agen10_easy, 11) == 1) / sum(idx_agen10_easy);
        acc_flip1_hard_post0 = sum(EEG.trialinfo(idx_agen10_hard, 11) == 1) / sum(idx_agen10_hard);
        acc_flip1_easy_post1 = sum(EEG.trialinfo(idx_agen11_easy, 11) == 1) / sum(idx_agen11_easy);
        acc_flip1_hard_post1 = sum(EEG.trialinfo(idx_agen11_hard, 11) == 1) / sum(idx_agen11_hard);
        acc_all(s, :) = [acc_flip0_easy_post0,...
                         acc_flip0_hard_post0,...
                         acc_flip1_easy_post0,...
                         acc_flip1_hard_post0,...
                         acc_flip1_easy_post1,...                  
                         acc_flip1_hard_post1];

        % Get mean RT of correct trials
        rt_flip0_easy_post0 = nanmean(EEG.trialinfo(idx_agen00_easy & idx_correct, 7));
        rt_flip0_hard_post0 = nanmean(EEG.trialinfo(idx_agen00_hard & idx_correct, 7));
        rt_flip1_easy_post0 = nanmean(EEG.trialinfo(idx_agen10_easy & idx_correct, 7));
        rt_flip1_hard_post0 = nanmean(EEG.trialinfo(idx_agen10_hard & idx_correct, 7));
        rt_flip1_easy_post1 = nanmean(EEG.trialinfo(idx_agen11_easy & idx_correct, 7));
        rt_flip1_hard_post1 = nanmean(EEG.trialinfo(idx_agen11_hard & idx_correct, 7));
        rt_all(s, :) = [rt_flip0_easy_post0,...
                        rt_flip0_hard_post0,...
                        rt_flip1_easy_post0,...
                        rt_flip1_hard_post0,...
                        rt_flip1_easy_post1,...
                        rt_flip1_hard_post1];

    end % End subject iteration

    figure()

    subplot(1, 2, 1)
    bar([1 : 6], mean(acc_all, 1))
    set(gca,'xticklabel', {'fl0 ez 0', 'fl0 hrd 0', 'fl1 ez 0', 'fl1 hrd 0', 'fl1 ez 1', 'fl1 hrd 1'})
    ylim([0.4, 1.2])
    title('asisracy')

    subplot(1, 2, 2)
    bar([1 : 6], mean(rt_all, 1))
    set(gca,'xticklabel', {'fl0 ez 0', 'fl0 hrd 0', 'fl1 ez 0', 'fl1 hrd 0', 'fl1 ez 1', 'fl1 hrd 1'})
    ylim([400, 650])
    title('rt - correct trials')

    % Perform rmANOVA for accuracy
    varnames = {'subject', 'fl0_ez_0', 'fl0_hrd_0', 'fl1_ez_0', 'fl1_ez_1', 'fl1_hrd_0', 'fl1_hrd_1'};
    t = table(ids', acc_all(:, 1), acc_all(:, 2), acc_all(:, 3), acc_all(:, 4), acc_all(:, 5), acc_all(:, 6), 'VariableNames', varnames);
    within = table({'easy'; 'hard'; 'easy'; 'hard'; 'easy'; 'hard'}, {'0_0'; '0_0'; '1_0'; '1_0'; '1_1'; '1_1'}, 'VariableNames', {'difficulty', 'flip'});
    rm = fitrm(t, 'fl0_ez_0-fl1_hrd_1~1', 'WithinDesign', within);
    anova_acc = ranova(rm, 'WithinModel', 'difficulty + flip + difficulty*flip');
    anova_acc

    % Perform rmANOVA for RT
    varnames = {'subject', 'fl0_ez_0', 'fl0_hrd_0', 'fl1_ez_0', 'fl1_ez_1', 'fl1_hrd_0', 'fl1_hrd_1'};
    t = table(ids', rt_all(:, 1), rt_all(:, 2), rt_all(:, 3), rt_all(:, 4), rt_all(:, 5), rt_all(:, 6), 'VariableNames', varnames);
    within = table({'easy'; 'hard'; 'easy'; 'hard'; 'easy'; 'hard'}, {'0_0'; '0_0'; '1_0'; '1_0'; '1_1'; '1_1'}, 'VariableNames', {'difficulty', 'flip'});
    rm = fitrm(t, 'fl0_ez_0-fl1_hrd_1~1', 'WithinDesign', within);
    anova_rt = ranova(rm, 'WithinModel', 'difficulty + flip + difficulty*flip');
    anova_rt

    % Average across difficulty conditions for post hoc t-tests
    rt_0_0 = mean(rt_all(:, [1, 2]), 2);
    rt_1_0 = mean(rt_all(:, [3, 4]), 2);
    rt_1_1 = mean(rt_all(:, [5, 6]), 2);
    acc_0_0 = mean(acc_all(:, [1, 2]), 2);
    acc_1_0 = mean(acc_all(:, [3, 4]), 2);
    acc_1_1 = mean(acc_all(:, [5, 6]), 2);

    % Calculate t-tests
    [H1, P1, CI1, STATS1] = ttest(rt_0_0, rt_1_0);
    [H2, P2, CI2, STATS2] = ttest(rt_0_0, rt_1_1);
    [H3, P3, CI3, STATS3] = ttest(rt_1_0, rt_1_1);

    % Correct for multiple comparisons
    p = [P1, P2, P3];
    p_corr_rt = bonf_holm(p, 0.05);

    % Calculate t-tests
    [H1, P1, CI1, STATS1] = ttest(acc_0_0, acc_1_0);
    [H2, P2, CI2, STATS2] = ttest(acc_0_0, acc_1_1);
    [H3, P3, CI3, STATS3] = ttest(acc_1_0, acc_1_1);

    % Correct for multiple comparisons
    p = [P1, P2, P3];
    p_corr_acc = bonf_holm(p, 0.05);

    % Save behavioral data for veusz
    rt_mean = mean(rt_all, 1);
    rt_sd = std(rt_all, [], 1);
    rt_out = [rt_mean(1), rt_sd(1), rt_mean(3), rt_sd(3), rt_mean(5), rt_sd(5); rt_mean(2), rt_sd(2), rt_mean(4), rt_sd(4), rt_mean(6), rt_sd(6)];
    dlmwrite([PATH_OUT, 'rt_veusz.csv'], rt_out, 'delimiter', '\t');

    acc_mean = mean(acc_all, 1);
    acc_sd = std(acc_all, [], 1);
    acc_out = [acc_mean(1), acc_sd(1), acc_mean(3), acc_sd(3), acc_mean(5), acc_sd(5); acc_mean(2), acc_sd(2), acc_mean(4), acc_sd(4), acc_mean(6), acc_sd(6)];
    dlmwrite([PATH_OUT, 'acc_veusz.csv'], acc_out, 'delimiter', '\t');

end % End part1
