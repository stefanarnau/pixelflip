clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2022.1/';
PATH_RAW = '/mnt/data_dump/pixelflip/0_raw/';
PATH_ICSET = '/mnt/data_dump/pixelflip/1_icset/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';

% Initialize eeglab
addpath(PATH_EEGLAB);
eeglab;

%% =============== LOAD DATA ======================================================================================================================

% Select subject to process
% Subjects available: 'VP01', 'VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP12', 'VP07', 'VP11', 'VP09', 'VP16', 'VP17'
subject = 'VP17';

% Get subject identifier as a number
id = str2num(subject(3 : 4));

% Load the raw data as recorded
EEG = pop_loadbv(PATH_RAW, [subject, '.vhdr'], [], []);

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Look at continuous data 
% -Inspect EEG struct (metadata like number of channels, datapoints etc...)

%% =============== EVENT CODING ===================================================================================================================

% Iterate events in "EEG.event"
trialinfo = [];
block_nr = 0;
trial_nr = 0;
for e = 1 : length(EEG.event)

    % If an S event
    if strcmpi(EEG.event(e).type(1), 'S')

        % Get event number
        enum = str2num(EEG.event(e).type(2 : end));

        % Set block number
        if enum >= 101 & enum <= 207 & trial_nr >= 20
            block_nr = str2num(EEG.event(e).type(end));
        end

        % If cue-stimulus
        if enum >= 11 & enum <= 18

            % Increase trial counter
            trial_nr = trial_nr + 1;

            % Get reliability
            if ismember(enum, [11, 12, 13, 14])
                reliability = 1; % Relaible
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

% Add trialinfo to EEG struct
EEG.trialinfo = trialinfo;

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Look at trialinfo in EEG-struct

%% =============== LOAD CHANNEL INFO ===================================================================================================================

% Add FCz, the online reference channel, as an empty channel
EEG.data(end + 1, :) = 0;
EEG.nbchan = size(EEG.data, 1);
EEG.chanlocs(end + 1).labels = 'FCz';

% Select the file containing info about electrode positions
channel_location_file = which('standard-10-5-cap385.elp');

% Add channel locations from file
EEG = pop_chanedit(EEG, 'lookup', channel_location_file);

% Save original channel locations (for later interpolation)
EEG.chanlocs_original = EEG.chanlocs;

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Look at chanlocs in EEG-struct
% -Plot topography using the GUI (plot labels/numbers)

%% =============== RESAMPLING & FILTERING ===================================================================================================================

% Rereference data to CPz, so that FCz contains non-interpolated data
EEG = pop_reref(EEG, 'CPz');

% Resample data
EEG = pop_resample(EEG, 200);

% Filter data for ICA
ICA = pop_basicfilter(EEG, [1 : EEG.nbchan], 'Cutoff', [2, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary');

% Filter data for ERP analysis
EEG = pop_basicfilter(EEG, [1 : EEG.nbchan], 'Cutoff', [0.01, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 6, 'RemoveDC', 'on', 'Boundary', 'boundary'); 

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Inspect the filtered continuous data using the GUI

%% =============== REREFERENCING ===================================================================================================================


% Detect and reject bad channels
[EEG, EEG.chans_rejected] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'kurt');
[ICA, ICA.chans_rejected] = pop_rejchan(ICA, 'elec', [1 : ICA.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'kurt');

% Interpolate rejected channels
EEG = pop_interp(EEG, EEG.chanlocs_original, 'spherical');
ICA = pop_interp(ICA, ICA.chanlocs_original, 'spherical');

% Rereference data to common average reference
EEG = pop_reref(EEG, []);
ICA = pop_reref(ICA, []);

% Determine rank of data
dataRank = sum(eig(cov(double(ICA.data'))) > 1e-6);

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Inspect effects of average reference in channel data (e.g. blink distribution)

%% =============== EPOCH DATA ===================================================================================================

% Epoch EEG data cue-locked
EEG = pop_epoch(EEG, {'cue'}, [-0.3, 2], 'newname', [subject '_epoched'], 'epochinfo', 'yes');
ICA = pop_epoch(ICA, {'cue'}, [-0.3, 2], 'newname', [subject '_epoched'], 'epochinfo', 'yes');

% Remove baseline
EEG = pop_rmbase(EEG, [-200, 0]);
ICA = pop_rmbase(ICA, [-200, 0]);

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Plot an ERP image (via GUI) of all trials. Try several electrodes.

%% =============== CLEAN EPOCHED DATA ===================================================================================================

% Automatically reject trials
[EEG, EEG.rejected_epochs] = pop_autorej(EEG, 'nogui', 'on');
[ICA, ICA.rejected_epochs] = pop_autorej(ICA, 'nogui', 'on');

% Remove rejected trials from trialinfo as well
EEG.trialinfo(EEG.rejected_epochs, :) = [];
ICA.trialinfo(ICA.rejected_epochs, :) = [];

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Plot an ERP image again (via GUI) of the cleaned dataset.

%% =============== RUN INDEPENDENT COMPONENT ANALYSIS ===================================================================================================

% Run independent component analysis (ICA) on ICA-data
ICA = pop_runica(ICA, 'extended', 1, 'interrupt', 'on', 'PCA', dataRank - 5); % PCA compression

% Classify the independent components (ICs)
ICA = iclabel(ICA);

% Find eye-activity related ICs
ICA.eye_movements = find(ICA.etc.ic_classification.ICLabel.classifications(:, 3) > 0.3);

% Copy ICs to ERP-dataset
EEG = pop_editset(EEG, 'icachansind', 'ICA.icachansind', 'icaweights', 'ICA.icaweights', 'icasphere', 'ICA.icasphere');
EEG.etc = ICA.etc;
EEG.eye_movements = ICA.eye_movements;

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Inspect ICs via GUI
% -Inspect epoched channel data. Look at blinks.
% -Inspect epoched ICA data.

% Remove eye-related ICs
EEG = pop_subcomp(EEG, EEG.eye_movements, 0);

% Redraw eeglab to inspect data
eeglab redraw;

% TASKS:
% -Inspect remaining ICs via GUI
% -Inspect cleaned epoched channel data. Can you still see blinks?

