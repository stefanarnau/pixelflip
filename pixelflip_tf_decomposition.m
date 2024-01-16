clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_TF_DATA     = '/mnt/data_dump/pixelflip/3_tf_data/';

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

% Init fieldtrip
ft_path = '/home/plkn/fieldtrip-master/';
addpath(ft_path);
ft_defaults;

% Load data
EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

% Set complex Morlet wavelet parameters
n_frq = 29;
frqrange = [2, 30];
tfres_range = [600, 200];

% Set wavelet time
wtime = -2 : 1 / EEG.srate : 2;

% Determine fft frqs
hz = linspace(0, EEG.srate, length(wtime));

% Create wavelet frequencies and tapering Gaussian widths in temporal domain
%tf_freqs = logspace(log10(frqrange(1)), log10(frqrange(2)), n_frq);
%fwhmTs = logspace(log10(tfres_range(1)), log10(tfres_range(2)), n_frq);
tf_freqs = linspace(frqrange(1), frqrange(2), n_frq);
fwhmTs = linspace(tfres_range(1), tfres_range(2), n_frq);


% Init matrices for wavelets
cmw = zeros(length(tf_freqs), length(wtime));
cmwX = zeros(length(tf_freqs), length(wtime));
tlim = zeros(1, length(tf_freqs));

% These will contain the wavelet widths as full width at 
% half maximum in the temporal and spectral domain
obs_fwhmT = zeros(1, length(tf_freqs));
obs_fwhmF = zeros(1, length(tf_freqs));

% Create the wavelets
for frq = 1 : length(tf_freqs)

    % Create wavelet with tapering gaussian corresponding to desired width in temporal domain
    cmw(frq, :) = exp(2 * 1i * pi * tf_freqs(frq) .* wtime) .* exp((-4 * log(2) * wtime.^2) ./ (fwhmTs(frq) / 1000)^2);

    % Normalize wavelet
    cmw(frq, :) = cmw(frq, :) ./ max(cmw(frq, :));

    % Create normalized freq domain wavelet
    cmwX(frq, :) = fft(cmw(frq, :)) ./ max(fft(cmw(frq, :)));

    % Determine observed fwhmT
    midt = dsearchn(wtime', 0);
    cmw_amp = abs(cmw(frq, :)) ./ max(abs(cmw(frq, :))); % Normalize cmw amplitude
    obs_fwhmT(frq) = wtime(midt - 1 + dsearchn(cmw_amp(midt : end)', 0.5)) - wtime(dsearchn(cmw_amp(1 : midt)', 0.5));

    % Determine observed fwhmF
    idx = dsearchn(hz', tf_freqs(frq));
    cmwx_amp = abs(cmwX(frq, :)); 
    obs_fwhmF(frq) = hz(idx - 1 + dsearchn(cmwx_amp(idx : end)', 0.5) - dsearchn(cmwx_amp(1 : idx)', 0.5));

end

% Define time window of analysis
prune_times = [-500, 1800]; 
tf_times = EEG.times(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)));

% Get chanlocs
chanlocs = EEG.chanlocs;

% Restructure coordinates
chanlabs = {};
coords = [];
for c = 1 : numel(chanlocs)
    chanlabs{c} = chanlocs(c).labels;
    coords(c, :) = [chanlocs(c).X, chanlocs(c).Y, chanlocs(c).Z];
end

% A sensor struct
sensors = struct();
sensors.label = chanlabs;
sensors.chanpos = coords;
sensors.elecpos = coords;

% Prepare neighbor struct
cfg                 = [];
cfg.elec            = sensors;
cfg.feedback        = 'no';
cfg.method          = 'triangulation'; 
neighbours          = ft_prepare_neighbours(cfg);

% Save metadata
save([PATH_TF_DATA, 'chanlocs.mat'], 'chanlocs');
save([PATH_TF_DATA, 'tf_freqs.mat'], 'tf_freqs');
save([PATH_TF_DATA, 'tf_times.mat'], 'tf_times');
save([PATH_TF_DATA, 'neighbours.mat'], 'neighbours');
save([PATH_TF_DATA, 'obs_fwhmT.mat'], 'obs_fwhmT');
save([PATH_TF_DATA, 'obs_fwhmF.mat'], 'obs_fwhmF');

% Loop subjects
ids = [];
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Collect IDs as number
    ids(s) = str2num(subject(3 : 4));

    % Load subject data. EEG data has dimensionality channels x times x trials
    EEG = pop_loadset('filename',    [subject, '_cleaned_cue_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

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

    % Save trialinfo
    trialinfo = EEG.trialinfo;
    save([PATH_TF_DATA, 'trialinfo_', subject, '.mat'], 'trialinfo');

    % Loop channels
    for ch = 1 : EEG.nbchan

        % Init tf matrices
        powcube = NaN(length(tf_freqs), EEG.pnts, EEG.trials);

        % Talk
        fprintf('\ntf-decomposition | subject %i/%i | channel %i/%i\n', s, length(subject_list), ch, EEG.nbchan);

        % Get channel data
        channel_data = squeeze(EEG.data(ch, :, :));

        % convolution length
        convlen = size(channel_data, 1) * size(channel_data, 2) + size(cmw, 2) - 1;

        % cmw to freq domain and scale
        cmwX = zeros(length(tf_freqs), convlen);
        for f = 1 : length(tf_freqs)
            cmwX(f, :) = fft(cmw(f, :), convlen);
            cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
        end

        % Get TF-power
        tmp = fft(reshape(channel_data, 1, []), convlen);
        for f = 1 : length(tf_freqs)
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, EEG.pnts, EEG.trials);
            powcube(f, :, :) = abs(as) .^ 2;   
        end
        
        % Cut edges
        powcube = single(powcube(:, dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :));

        % Save
        save([PATH_TF_DATA, 'powcube_', subject, '_chan_', num2str(ch), '.mat'], 'powcube');

    end % end channel loop

end % End subject loop