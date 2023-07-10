clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2022.1/';
PATH_RAW = '/mnt/data_dump/pixelflip/0_raw/';
PATH_ICSET = '/mnt/data_dump/pixelflip/1_icset/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';

% Subject list (stating the obvious here...)
%subject_list = {'VP01', 'VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP12', 'VP07',...
%                'VP11', 'VP09', 'VP16', 'VP17', 'VP19', 'VP21', 'VP23', 'VP25',...
%                'VP27', 'VP29', 'VP31', 'VP18', 'VP20', 'VP22', 'VP24', 'VP26',...
%                'VP28', 'VP13', 'VP15'};

subject_list = {'VP15'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Get chanlocfile
channel_location_file = which('standard-10-5-cap385.elp');

% Loop subjects
for s = 1 : length(subject_list)

    % Get id stuff
    subject = subject_list{s};
    id = str2num(subject(3 : 4));

    % Load
    EEG = pop_loadbv(PATH_RAW, [subject, '.vhdr'], [], []);

    % Iterate events
    trialinfo = [];
    block_nr = 0;
    trial_nr = 0;
    enums = zeros(256, 1);
    for e = 1 : length(EEG.event)

        % If an S event
        if strcmpi(EEG.event(e).type(1), 'S')

            % Get event number
            enum = str2num(EEG.event(e).type(2 : end));

            enums(enum) = enums(enum) + 1;

            % Set block number
            if enum >= 101 & enum <= 207 & trial_nr >= 20
                block_nr = str2num(EEG.event(e).type(end));
            end

            % If cue (trial)
            if enum >= 11 & enum <= 18

                % Increase!!!!!
                trial_nr = trial_nr + 1;

                % Get reliability
                if ismember(enum, [11, 12, 13, 14])
                    reliability = 1; % Reliable
                else
                    reliability = 0; % unreliable
                end

                % Get difficulty
                if ismember(enum, [11, 12, 15, 16])
                    difficulty = 0; % easy
                else
                    difficulty = 1; % hard
                end

                % Get flipped
                if ismember(enum, [12, 14, 16, 18])
                    flipped = 1; % flipped
                else
                    flipped = 0; % not flipped
                end

                % Loop to feedback
                f = e;
                rt = NaN;
                key_pressed = 0;
                while ~ismember(str2num(EEG.event(f).type(2 : end)), [30, 31, 40, 41, 50])

                    % If response left
                    if str2num(EEG.event(f).type(2 : end)) == 5 
                        key_pressed = 1;
                        rt = EEG.event(f).latency - EEG.event(e).latency - 1200;
                    end

                    % If response right
                    if str2num(EEG.event(f).type(2 : end)) == 4
                        key_pressed = 2;
                        rt = EEG.event(f).latency - EEG.event(e).latency - 1200;
                    end

                    % Move on
                    f = f + 1;

                end

                % Check color pressed
                color_pressed = 0;
                if key_pressed == 1 & mod(id, 2) == 1 % Pressed left odd id
                    color_pressed = 1; % cyan
                elseif key_pressed == 1 & mod(id, 2) == 0 % Pressed left even id
                    color_pressed = 2; % orange
                elseif key_pressed == 2 & mod(id, 2) == 1 % Pressed right odd id
                    color_pressed = 2; % orange
                elseif key_pressed == 2 & mod(id, 2) == 0 % Pressed right even id
                    color_pressed = 1; % cyan
                end

                % Get feedback color and feedback accuracy
                if str2num(EEG.event(f).type(2 : end)) == 30
                    feedback_accuracy = 1; % correct
                    feedback_color = 2; % orange
                elseif str2num(EEG.event(f).type(2 : end)) == 31
                    feedback_accuracy = 1; % correct
                    feedback_color = 1; % cyan
                elseif str2num(EEG.event(f).type(2 : end)) == 40
                    feedback_accuracy = 0; % false
                    feedback_color = 2; % orange
                elseif str2num(EEG.event(f).type(2 : end)) == 41
                    feedback_accuracy = 0; % false
                    feedback_color = 1; % cyan
                elseif str2num(EEG.event(f).type(2 : end)) == 50
                    feedback_accuracy = -1; % missing
                    feedback_color = 0; % none
                end

                % Get actual accuracy
                if feedback_accuracy == 1 & flipped == 0
                    accuracy = 1;
                elseif feedback_accuracy == 1 & flipped == 1
                    accuracy = 0;
                elseif feedback_accuracy == 0 & flipped == 0
                    accuracy = 0;
                elseif feedback_accuracy == 0 & flipped == 1
                    accuracy = 1;
                elseif feedback_accuracy == -1
                    accuracy = -1; % missing
                end

                % Save info
                trialinfo(trial_nr, :) = [trial_nr, block_nr, reliability, difficulty, flipped, key_pressed, rt, color_pressed, feedback_accuracy, feedback_color, accuracy];

                % Rename events
                EEG.event(e).type = 'cue';
                EEG.event(e).code = 'cue';
                EEG.event(f).type = 'feedback';
                EEG.event(f).code = 'feedback';

            end
        end
    end

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

    % Move trialinfo to EEG
    EEG.trialinfo = trialinfo;

    % Add FCz as empty channel
    EEG.data(end + 1, :) = 0;
    EEG.nbchan = size(EEG.data, 1);
    EEG.chanlocs(end + 1).labels = 'FCz';

    % Add channel locations
    EEG = pop_chanedit(EEG, 'lookup', channel_location_file);

    % Save original channel locations (for later interpolation)
    EEG.chanlocs_original = EEG.chanlocs;

    % Reref to CPz, so that FCz obtains non-interpolated data
    EEG = pop_reref(EEG, 'CPz');

    % Resample data
    EEG    = pop_resample(EEG, 200);
    EEG_TF = pop_resample(EEG, 200);

    % Filter
    EEG    = pop_basicfilter(EEG,    [1 : EEG.nbchan],    'Cutoff', [0.01, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary'); 
    EEG_TF = pop_basicfilter(EEG_TF, [1 : EEG_TF.nbchan], 'Cutoff', [   2, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary');
        
    % Bad channel detection
    [EEG, EEG.chans_rejected]       = pop_rejchan(EEG,    'elec', [1 : EEG.nbchan],    'threshold', 5, 'norm', 'on', 'measure', 'kurt');
    [EEG_TF, EEG_TF.chans_rejected] = pop_rejchan(EEG_TF, 'elec', [1 : EEG_TF.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'kurt');

    % Interpolate channels
    EEG    = pop_interp(EEG,    EEG.chanlocs_original,    'spherical');
    EEG_TF = pop_interp(EEG_TF, EEG_TF.chanlocs_original, 'spherical');

    % Reref common average
    EEG_CONT    = pop_reref(EEG,    []);
    EEG_TF_CONT = pop_reref(EEG_TF, []);

    % Determine rank of data
    dataRank = sum(eig(cov(double(EEG_TF.data'))) > 1e-6);

    % -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    % Epoch EEG data cue-locked
    EEG    = pop_epoch(EEG_CONT, {'cue'}, [-0.3, 2], 'newname', [subject '_epoched'], 'epochinfo', 'yes');
    EEG    = pop_rmbase(EEG, [-200, 0]);
    EEG_TF = pop_epoch(EEG_TF_CONT, {'cue'}, [-0.8, 2.5], 'newname', [subject '_epoched'],  'epochinfo', 'yes');
    EEG_TF = pop_rmbase(EEG_TF, [-200, 0]);

    % Autoreject trials
    [EEG,    EEG.rejected_epochs]    = pop_autorej(EEG,    'nogui', 'on');
    [EEG_TF, EEG_TF.rejected_epochs] = pop_autorej(EEG_TF, 'nogui', 'on');

    % Remove from trialinfo
    EEG.trialinfo(EEG.rejected_epochs, :) = [];
    EEG_TF.trialinfo(EEG_TF.rejected_epochs, :) = [];

    % Runica & ICLabel
    EEG_TF = pop_runica(EEG_TF, 'extended', 1, 'interrupt', 'on', 'PCA', dataRank - 5);
    EEG_TF = iclabel(EEG_TF);

    % Find nobrainer
    EEG_TF.nobrainer = find(EEG_TF.etc.ic_classification.ICLabel.classifications(:, 3) > 0.3);

    % Copy ICs to erpset
    EEG = pop_editset(EEG, 'icachansind', 'EEG_TF.icachansind', 'icaweights', 'EEG_TF.icaweights', 'icasphere', 'EEG_TF.icasphere');
    EEG.etc = EEG_TF.etc;
    EEG.nobrainer = EEG_TF.nobrainer;

    % Save IC set
    pop_saveset(EEG,    'filename', [subject, '_icset_cue_erp.set'], 'filepath', PATH_ICSET, 'check', 'on');
    pop_saveset(EEG_TF, 'filename', [subject, '_icset_cue_tf.set'],  'filepath', PATH_ICSET, 'check', 'on');

    % Remove components
    EEG    = pop_subcomp(EEG, EEG.nobrainer, 0);
    EEG_TF = pop_subcomp(EEG_TF, EEG_TF.nobrainer, 0);

    % Save clean data
    pop_saveset(EEG, 'filename',    [subject, '_cleaned_cue_erp.set'], 'filepath', PATH_AUTOCLEANED, 'check', 'on');
    pop_saveset(EEG_TF, 'filename', [subject, '_cleaned_cue_tf.set'],  'filepath', PATH_AUTOCLEANED, 'check', 'on');

    % -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    % Epoch EEG data feedback-locked
    EEG    = pop_epoch(EEG_CONT, {'feedback'}, [-0.3, 1.2], 'newname', [subject '_epoched'], 'epochinfo', 'yes');
    EEG    = pop_rmbase(EEG, [-200, 0]);
    EEG_TF = pop_epoch(EEG_TF_CONT, {'feedback'}, [-0.8, 1.7], 'newname', [subject '_epoched'],  'epochinfo', 'yes');
    EEG_TF = pop_rmbase(EEG_TF, [-200, 0]);

    % Autoreject trials
    [EEG,    EEG.rejected_epochs]    = pop_autorej(EEG,    'nogui', 'on');
    [EEG_TF, EEG_TF.rejected_epochs] = pop_autorej(EEG_TF, 'nogui', 'on');

    % Remove from trialinfo
    EEG.trialinfo(EEG.rejected_epochs, :) = [];
    EEG_TF.trialinfo(EEG_TF.rejected_epochs, :) = [];

    % Runica & ICLabel
    EEG_TF = pop_runica(EEG_TF, 'extended', 1, 'interrupt', 'on', 'PCA', dataRank - 5);
    EEG_TF = iclabel(EEG_TF);

    % Find nobrainer
    EEG_TF.nobrainer = find(EEG_TF.etc.ic_classification.ICLabel.classifications(:, 3) > 0.3);

    % Copy ICs to erpset
    EEG = pop_editset(EEG, 'icachansind', 'EEG_TF.icachansind', 'icaweights', 'EEG_TF.icaweights', 'icasphere', 'EEG_TF.icasphere');
    EEG.etc = EEG_TF.etc;
    EEG.nobrainer = EEG_TF.nobrainer;

    % Save IC set
    pop_saveset(EEG,    'filename', [subject, '_icset_feedback_erp.set'], 'filepath', PATH_ICSET, 'check', 'on');
    pop_saveset(EEG_TF, 'filename', [subject, '_icset_feedback_tf.set'],  'filepath', PATH_ICSET, 'check', 'on');

    % Remove components
    EEG    = pop_subcomp(EEG, EEG.nobrainer, 0);
    EEG_TF = pop_subcomp(EEG_TF, EEG_TF.nobrainer, 0);

    % Save clean data
    pop_saveset(EEG, 'filename',    [subject, '_cleaned_feedback_erp.set'], 'filepath', PATH_AUTOCLEANED, 'check', 'on');
    pop_saveset(EEG_TF, 'filename', [subject, '_cleaned_feedback_tf.set'],  'filepath', PATH_AUTOCLEANED, 'check', 'on');

end % End subject loop


