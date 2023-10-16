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

% Part 1: Calculate ersp
if ismember('part1', to_execute)

    trial_counts = [];

    % Loop subjects
    for s = 1 : length(subject_list)

        % Get id stuff
        subject = subject_list{s};

        % Collect IDs as number
        id = str2num(subject(3 : 4));

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
                    % if EEG.trialinfo(idx_prev, 11) ~= 1

                    %     % Not a good trial...
                    %     EEG.trialinfo(e, 12) = -1;
                    %     EEG.trialinfo(e, 13) = -1;

                    %     % Next
                    %     continue;

                    % end

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

        % Exclude omission errors
        EEG.trialinfo(EEG.trialinfo(:, 11) == -1, :) = [];
        
        % Init out matrix for subject
        out_subject = zeros(size(EEG.trialinfo, 1), 6);

        % Fill matrix
        out_subject(:, 1) = id; % id
        out_subject(:, 2) = EEG.trialinfo(:, 7); % rt
        out_subject(:, 3) = EEG.trialinfo(:, 11); % accuracy
        out_subject(:, 4) = EEG.trialinfo(:, 11); % key pressed
        out_subject(EEG.trialinfo(:, 4) == 0, 5) = 0; % easy
        out_subject(EEG.trialinfo(:, 4) == 1, 5) = 1; % hard
        out_subject(EEG.trialinfo(:, 3) == 1, 6) = 1; % no-flip block
        out_subject(EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 12) == 0, 6) = 2; % flip-block no-flip
        out_subject(EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 12) == 1, 6) = 3; % flip-block flip

        % Remove trials without valid sequence coding
        out_subject(out_subject(:, 6) == 0, :) = [];

        % Count trials
        trial_counts(s, :) = [s,...
                              sum(out_subject(:, 5) == 0 & out_subject(:, 6) == 1),...
                              sum(out_subject(:, 5) == 0 & out_subject(:, 6) == 2),...
                              sum(out_subject(:, 5) == 0 & out_subject(:, 6) == 3),...
                              sum(out_subject(:, 5) == 1 & out_subject(:, 6) == 1),...
                              sum(out_subject(:, 5) == 1 & out_subject(:, 6) == 2),...
                              sum(out_subject(:, 5) == 1 & out_subject(:, 6) == 3)];

        % Collect
        if s == 1
            out = out_subject;
        else 
            out = [out; out_subject];
        end

    end % End subject iteration



end % End part1
