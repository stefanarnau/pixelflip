clear all;

% PATH VARS
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_TF_DATA     = '/mnt/data_dump/pixelflip/3_tf_data/ersps/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP12', 'VP07',...
                'VP11', 'VP09', 'VP16', 'VP17', 'VP19', 'VP21', 'VP23', 'VP25',...
                'VP27', 'VP29', 'VP31', 'VP18', 'VP20', 'VP22', 'VP24', 'VP26',...
                'VP28', 'VP13', 'VP15'};
    
% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% SWITCH: Switch parts of script on/off
to_execute = {'part2'};

% Part 1: Calculate ersp
if ismember('part1', to_execute)

    % Load data
    EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Set complex Morlet wavelet parameters
    n_frq = 30;
    frqrange = [2, 20];
    tfres_range = [600, 300];

    % Set wavelet time
    wtime = -2 : 1 / EEG.srate : 2;

    % Determine fft frqs
    hz = linspace(0, EEG.srate, length(wtime));

    % Create wavelet frequencies and tapering Gaussian widths in temporal domain
    tf_freqs = logspace(log10(frqrange(1)), log10(frqrange(2)), n_frq);
    fwhmTs = logspace(log10(tfres_range(1)), log10(tfres_range(2)), n_frq);

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
    prune_times = [-500, 2200]; 
    tf_times = EEG.times(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)));

    % Result struct
    chanlocs = EEG.chanlocs;
    ersp_easy_accu = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));
    ersp_easy_flip = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));
    ersp_hard_accu = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));
    ersp_hard_flip = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));

    % Loop subjects
    for s = 1 : length(subject_list)

        % Get id stuff
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject, '_cleaned_cue_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

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
        idx_easy_accu = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 1;
        idx_easy_flip = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 3) == 0;
        idx_hard_accu = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 1;
        idx_hard_flip = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 3) == 0;

        % Loop channels
        for ch = 1 : EEG.nbchan

            % Init tf matrices
            powcube = NaN(length(tf_freqs), EEG.pnts, EEG.trials);

            % Talk
            fprintf('\ntf-decomposition | subject %i/%i | channel %i/%i\n', s, length(subject_list), ch, EEG.nbchan);

            % Get component signal
            channel_data = squeeze(eeg_data(ch, :, :));

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
            powcube = powcube(:, dsearchn(EEG.times', -500) : dsearchn(EEG.times', 2200), :);

            % Get condition general baseline values
            ersp_bl = [-500, -200];
            tmp = squeeze(mean(powcube, 3));
            [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
            [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
            blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

            % Calculate ersp
            ersp_easy_accu(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_easy_accu), 3)), blvals)));
            ersp_easy_flip(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_easy_flip), 3)), blvals)));
            ersp_hard_accu(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_hard_accu), 3)), blvals)));
            ersp_hard_flip(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_hard_flip), 3)), blvals)));

        end % end channel loop

    end % end subject loop

    % Save shit
    save([PATH_TF_DATA, 'chanlocs.mat'], 'chanlocs');
    save([PATH_TF_DATA, 'tf_freqs.mat'], 'tf_freqs');
    save([PATH_TF_DATA, 'tf_times.mat'], 'tf_times');
    save([PATH_TF_DATA, 'ersp_easy_accu.mat'], 'ersp_easy_accu');
    save([PATH_TF_DATA, 'ersp_easy_flip.mat'], 'ersp_easy_flip');
    save([PATH_TF_DATA, 'ersp_hard_accu.mat'], 'ersp_hard_accu');
    save([PATH_TF_DATA, 'ersp_hard_flip.mat'], 'ersp_hard_flip');

end % End part1

% Part 2: Calculate ersp
if ismember('part2', to_execute)

    % Load shit
    load([PATH_TF_DATA, 'chanlocs.mat']);
    load([PATH_TF_DATA, 'tf_freqs.mat']);
    load([PATH_TF_DATA, 'tf_times.mat']);
    load([PATH_TF_DATA, 'ersp_easy_accu.mat']);
    load([PATH_TF_DATA, 'ersp_easy_flip.mat']);
    load([PATH_TF_DATA, 'ersp_hard_accu.mat']);
    load([PATH_TF_DATA, 'ersp_hard_flip.mat']);

    % Get theta idx
    idx_theta = tf_freqs >= 8 & tf_freqs <= 12;

    % Frontal channels idx
    idx_frontal = [15, 19, 20, 65];

    % Average across channels and subjects and theta-frequencies
    theta_easy_accu = squeeze(mean(ersp_easy_accu(:, idx_frontal, idx_theta, :), [1, 2, 3]));
    theta_easy_flip = squeeze(mean(ersp_easy_flip(:, idx_frontal, idx_theta, :), [1, 2, 3]));
    theta_hard_accu = squeeze(mean(ersp_hard_accu(:, idx_frontal, idx_theta, :), [1, 2, 3]));
    theta_hard_flip = squeeze(mean(ersp_hard_flip(:, idx_frontal, idx_theta, :), [1, 2, 3]));

    % Plot
    figure()
    plot(tf_times, theta_easy_accu, 'k-', 'LineWidth', 2)
    hold on;
    plot(tf_times, theta_easy_flip, 'k:', 'LineWidth', 2)
    plot(tf_times, theta_hard_accu, 'm-', 'LineWidth', 2)
    plot(tf_times, theta_hard_flip, 'm:', 'LineWidth', 2)
    legend({'easy-accurate', 'easy-flip', 'hard-accurate', 'hard-flip'})
    title('Frontal Theta Power')
    xline([0, 1200])


end