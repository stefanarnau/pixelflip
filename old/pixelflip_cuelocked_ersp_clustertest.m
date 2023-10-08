clear all;

% PATH VARS
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_TF_DATA     = '/mnt/data_dump/pixelflip/3_tf_data/ersps/';
PATH_OUT         = '/mnt/data_dump/pixelflip/3_tf_data/results/'; 

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
to_execute = {'part3'};

% Part 1: Calculate ersp
if ismember('part1', to_execute)

    % Load data
    EEG = pop_loadset('filename', [subject_list{1}, '_cleaned_cue_tf.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Set complex Morlet wavelet parameters
    n_frq = 50;
    frqrange = [2, 30];
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



    % Loop subjects
    for s = 1 : length(subject_list)

        % Result matrices
        chanlocs = EEG.chanlocs;
        rpow_easy_accu = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        rpow_easy_flip = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        rpow_hard_accu = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        rpow_hard_flip = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        ersp_easy_accu = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        ersp_easy_flip = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        ersp_hard_accu = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        ersp_hard_flip = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        itpc_easy_accu = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        itpc_easy_flip = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        itpc_hard_accu = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));
        itpc_hard_flip = zeros(EEG.nbchan, length(tf_freqs), length(tf_times));

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

        % Balance number of trials
        min_n = min([length(idx_easy_accu), length(idx_easy_flip), length(idx_hard_accu), length(idx_hard_flip)]);
        idx_easy_accu = randsample(idx_easy_accu, min_n);
        idx_easy_flip = randsample(idx_easy_flip, min_n);
        idx_hard_accu = randsample(idx_hard_accu, min_n);
        idx_hard_flip = randsample(idx_hard_flip, min_n);

        % Loop channels
        parfor ch = 1 : EEG.nbchan

            % Init tf matrices
            powcube = NaN(length(tf_freqs), EEG.pnts, EEG.trials);
            phacube = NaN(length(tf_freqs), EEG.pnts, EEG.trials);

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
                phacube(f, :, :) = angle(as);
            end
            
            % Cut edges
            powcube = powcube(:, dsearchn(EEG.times', -500) : dsearchn(EEG.times', 2200), :);
            phacube = phacube(:, dsearchn(EEG.times', -500) : dsearchn(EEG.times', 2200), :);

            % Get condition general baseline values
            ersp_bl = [-500, -200];
            tmp = squeeze(mean(powcube, 3));
            [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
            [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
            blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

            % Calculate ersp
            rpow_easy_accu(ch, :, :) = squeeze(mean(powcube(:, :, idx_easy_accu), 3));
            rpow_easy_flip(ch, :, :) = squeeze(mean(powcube(:, :, idx_easy_flip), 3));
            rpow_hard_accu(ch, :, :) = squeeze(mean(powcube(:, :, idx_hard_accu), 3));
            rpow_hard_flip(ch, :, :) = squeeze(mean(powcube(:, :, idx_hard_flip), 3));

            % Calculate ersp
            ersp_easy_accu(ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_easy_accu), 3)), blvals));
            ersp_easy_flip(ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_easy_flip), 3)), blvals));
            ersp_hard_accu(ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_hard_accu), 3)), blvals));
            ersp_hard_flip(ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_hard_flip), 3)), blvals));

            % Calculate itpc
            itpc_easy_accu(ch, :, :) = abs(squeeze(mean(exp(1i*phacube(:, :, idx_easy_accu)), 3)));
            itpc_easy_flip(ch, :, :) = abs(squeeze(mean(exp(1i*phacube(:, :, idx_easy_flip)), 3)));
            itpc_hard_accu(ch, :, :) = abs(squeeze(mean(exp(1i*phacube(:, :, idx_hard_accu)), 3)));
            itpc_hard_flip(ch, :, :) = abs(squeeze(mean(exp(1i*phacube(:, :, idx_hard_flip)), 3)));

        end % end channel loop

        % Save shit
        save([PATH_TF_DATA, 'chanlocs.mat'], 'chanlocs');
        save([PATH_TF_DATA, 'tf_freqs.mat'], 'tf_freqs');
        save([PATH_TF_DATA, 'tf_times.mat'], 'tf_times');
        save([PATH_TF_DATA, subject, '_rpow_easy_accu.mat'], 'rpow_easy_accu');
        save([PATH_TF_DATA, subject, '_rpow_easy_flip.mat'], 'rpow_easy_flip');
        save([PATH_TF_DATA, subject, '_rpow_hard_accu.mat'], 'rpow_hard_accu');
        save([PATH_TF_DATA, subject, '_rpow_hard_flip.mat'], 'rpow_hard_flip');
        save([PATH_TF_DATA, subject, '_ersp_easy_accu.mat'], 'ersp_easy_accu');
        save([PATH_TF_DATA, subject, '_ersp_easy_flip.mat'], 'ersp_easy_flip');
        save([PATH_TF_DATA, subject, '_ersp_hard_accu.mat'], 'ersp_hard_accu');
        save([PATH_TF_DATA, subject, '_ersp_hard_flip.mat'], 'ersp_hard_flip');
        save([PATH_TF_DATA, subject, '_itpc_easy_accu.mat'], 'itpc_easy_accu');
        save([PATH_TF_DATA, subject, '_itpc_easy_flip.mat'], 'itpc_easy_flip');
        save([PATH_TF_DATA, subject, '_itpc_hard_accu.mat'], 'itpc_hard_accu');
        save([PATH_TF_DATA, subject, '_itpc_hard_flip.mat'], 'itpc_hard_flip');

    end % end subject loop

end % End part1

% Part 2: Calculate cluster test for ersp
if ismember('part2', to_execute)

    % Load shit
    load([PATH_TF_DATA, 'chanlocs.mat']);
    load([PATH_TF_DATA, 'tf_freqs.mat']);
    load([PATH_TF_DATA, 'tf_times.mat']);

    % Build elec struct
    for ch = 1 : length(chanlocs)
        elec.label{ch} = chanlocs(ch).labels;
        elec.elecpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
        elec.chanpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
    end

    % Prepare layout
    cfg      = [];
    cfg.elec = elec;
    cfg.rotate = 90;
    layout = ft_prepare_layout(cfg);

    % Size of subject dimension
    n_subjects = length(subject_list);

    % Re-organize data
    for s = 1 : n_subjects

        % Subject identifier
        subject = subject_list{s};

        % Load ersp data
        load([PATH_TF_DATA, subject, '_ersp_easy_accu.mat']);
        load([PATH_TF_DATA, subject, '_ersp_easy_flip.mat']);
        load([PATH_TF_DATA, subject, '_ersp_hard_accu.mat']);
        load([PATH_TF_DATA, subject, '_ersp_hard_flip.mat']);

        % get dims
        [n_channels, n_freqs, n_times] = size(ersp_easy_accu);

        ersp_accu.powspctrm = (ersp_easy_accu + ersp_hard_accu) / 2;
        ersp_accu.dimord    = 'chan_freq_time';
        ersp_accu.label     = elec.label;
        ersp_accu.freq      = tf_freqs;
        ersp_accu.time      = tf_times;

        ersp_flip.powspctrm = (ersp_easy_flip + ersp_hard_flip) / 2;
        ersp_flip.dimord    = 'chan_freq_time';
        ersp_flip.label     = elec.label;
        ersp_flip.freq      = tf_freqs;
        ersp_flip.time      = tf_times;

        ersp_easy.powspctrm = (ersp_easy_accu + ersp_easy_flip) / 2;
        ersp_easy.dimord    = 'chan_freq_time';
        ersp_easy.label     = elec.label;
        ersp_easy.freq      = tf_freqs;
        ersp_easy.time      = tf_times;

        ersp_hard.powspctrm = (ersp_hard_accu + ersp_hard_flip) / 2;
        ersp_hard.dimord    = 'chan_freq_time';
        ersp_hard.label     = elec.label;
        ersp_hard.freq      = tf_freqs;
        ersp_hard.time      = tf_times;

        ersp_diff_easy.powspctrm = ersp_easy_accu - ersp_easy_flip;
        ersp_diff_easy.dimord    = 'chan_freq_time';
        ersp_diff_easy.label     = elec.label;
        ersp_diff_easy.freq      = tf_freqs;
        ersp_diff_easy.time      = tf_times;

        ersp_diff_hard.powspctrm = ersp_hard_accu - ersp_hard_flip;
        ersp_diff_hard.dimord    = 'chan_freq_time';
        ersp_diff_hard.label     = elec.label;
        ersp_diff_hard.freq      = tf_freqs;
        ersp_diff_hard.time      = tf_times;

        % Collect
        d_ersp_accu{s} = ersp_accu;
        d_ersp_flip{s} = ersp_flip;
        d_ersp_easy{s} = ersp_easy;
        d_ersp_hard{s} = ersp_hard;
        d_ersp_diff_easy{s} = ersp_diff_easy;
        d_ersp_diff_hard{s} = ersp_diff_hard;

    end

    % Calculate grand averages
    cfg = [];
    cfg.keepindividual = 'yes';
    GA_accu = ft_freqgrandaverage(cfg, d_ersp_accu{1, :});
    GA_flip = ft_freqgrandaverage(cfg, d_ersp_flip{1, :});
    GA_easy = ft_freqgrandaverage(cfg, d_ersp_easy{1, :});
    GA_hard = ft_freqgrandaverage(cfg, d_ersp_hard{1, :});
    GA_diff_easy = ft_freqgrandaverage(cfg, d_ersp_diff_easy{1, :});
    GA_diff_hard = ft_freqgrandaverage(cfg, d_ersp_diff_hard{1, :});

    % Define neighbours
    cfg                 = [];
    cfg.layout          = layout;
    cfg.feedback        = 'no';
    cfg.method          = 'triangulation'; 
    cfg.neighbours      = ft_prepare_neighbours(cfg, GA_accu);
    neighbours          = cfg.neighbours;

    % Testparams
    testalpha   = 0.025;
    voxelalpha  = 0.01;
    nperm       = 1000;

    % Set config. Same for all tests
    cfg = [];
    cfg.tail             = 0;
    cfg.statistic        = 'depsamplesT';
    cfg.alpha            = testalpha;
    cfg.neighbours       = neighbours;
    cfg.minnbchan        = 2;
    cfg.method           = 'montecarlo';
    cfg.correctm         = 'cluster';
    cfg.clustertail      = 0;
    cfg.clusteralpha     = voxelalpha;
    cfg.clusterstatistic = 'maxsum';
    cfg.numrandomization = nperm;
    cfg.computecritval   = 'yes'; 
    cfg.ivar             = 1;
    cfg.uvar             = 2;
    cfg.design           = [ones(1, n_subjects), 2 * ones(1, n_subjects); 1 : n_subjects, 1 : n_subjects];

    % The tests
    [stat_agency] = ft_freqstatistics(cfg, GA_accu, GA_flip);
    [stat_difficulty] = ft_freqstatistics(cfg, GA_easy, GA_hard);
    [stat_interaction] = ft_freqstatistics(cfg, GA_diff_easy, GA_diff_hard);

    % Calculate and save effect sizes
    adjpetasq_agency = [];
    adjpetasq_difficulty = [];
    adjpetasq_interaction = [];
    for ch = 1 : n_channels
        petasq = (squeeze(stat_agency.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_agency.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
        adjpetasq_agency(ch, :, :) = adj_petasq;

        petasq = (squeeze(stat_difficulty.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_difficulty.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
        adjpetasq_difficulty(ch, :, :) = adj_petasq;

        petasq = (squeeze(stat_interaction.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_interaction.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
        adjpetasq_interaction(ch, :, :) = adj_petasq;
    end

    % Save cluster struct
    save([PATH_OUT 'adjpetasq_agency.mat'], 'adjpetasq_agency');
    save([PATH_OUT 'adjpetasq_difficulty.mat'], 'adjpetasq_difficulty');
    save([PATH_OUT 'adjpetasq_interaction.mat'], 'adjpetasq_interaction');

    % Identify significant clusters
    clust_thresh = 0.025;
    clusts = struct();
    cnt = 0;
    stat_names = {'stat_agency', 'stat_difficulty', 'stat_interaction'};
    for s = 1 : numel(stat_names)
        stat = eval(stat_names{s});
        if ~isempty(stat.negclusters)
            neg_idx = find([stat.negclusters(1, :).prob] < clust_thresh);
            for c = 1 : numel(neg_idx)
                cnt = cnt + 1;
                clusts(cnt).testlabel = stat_names{s};
                clusts(cnt).clustnum = cnt;
                clusts(cnt).time = stat.time;
                clusts(cnt).freq = stat.freq;
                clusts(cnt).polarity = -1;
                clusts(cnt).prob = stat.negclusters(1, neg_idx(c)).prob;
                clusts(cnt).idx = stat.negclusterslabelmat == neg_idx(c);
                clusts(cnt).stats = clusts(cnt).idx .* stat.stat * -1;
                clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2,3])));
            end
        end
        if ~isempty(stat.posclusters)
            pos_idx = find([stat.posclusters(1, :).prob] < clust_thresh);
            for c = 1 : numel(pos_idx)
                cnt = cnt + 1;
                clusts(cnt).testlabel = stat_names{s};
                clusts(cnt).clustnum = cnt;
                clusts(cnt).time = stat.time;
                clusts(cnt).freq = stat.freq;
                clusts(cnt).polarity = 1;
                clusts(cnt).prob = stat.posclusters(1, pos_idx(c)).prob;
                clusts(cnt).idx = stat.posclusterslabelmat == pos_idx(c);
                clusts(cnt).stats = clusts(cnt).idx .* stat.stat;
                clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2, 3])));
            end
        end
    end

    % Save cluster struct
    save([PATH_OUT 'significant_clusters.mat'], 'clusts');

    % Init eeglab
    addpath(PATH_EEGLAB);
    eeglab;

    % Plot identified cluster
    clinecol = 'k';
    cmap = 'jet';
    for cnt = 1 : numel(clusts)

        figure('Visible', 'off'); clf;

        subplot(2, 2, 1)
        pd = squeeze(sum(clusts(cnt).stats, 1));
        contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
        hold on
        contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
        colormap(cmap)
        set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-max(abs(pd(:))), max(abs(pd(:)))], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
        colorbar;
        title(['sum t across chans, plrt: ' num2str(clusts(cnt).polarity)], 'FontSize', 10)

        subplot(2, 2, 2)
        pd = squeeze(mean(clusts(cnt).idx, 1));
        contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
        hold on
        contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
        colormap(cmap)
        set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-1, 1], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
        colorbar;
        title(['proportion chans significant'], 'FontSize', 10)

        subplot(2, 2, 3)
        pd = squeeze(sum(clusts(cnt).stats, [2, 3]));
        topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
        colormap(cmap)
        set(gca, 'clim', [-max(abs(pd(:))), max(abs(pd(:)))])
        colorbar;
        title(['sum t per electrode'], 'FontSize', 10)

        subplot(2, 2, 4)
        pd = squeeze(mean(clusts(cnt).idx, [2, 3]));
        topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
        colormap(cmap)
        set(gca, 'clim', [-1, 1])
        colorbar;
        title(['proportion tf-points significant'], 'FontSize', 10)

        saveas(gcf, [PATH_OUT 'clustnum_' num2str(clusts(cnt).clustnum) '_' clusts(cnt).testlabel '.png']); 
    end

end % End part 2


% Part 3: Whats going on actually...
if ismember('part3', to_execute)

    % Load shit
    load([PATH_TF_DATA, 'chanlocs.mat']);
    load([PATH_TF_DATA, 'tf_freqs.mat']);
    load([PATH_TF_DATA, 'tf_times.mat']);

    % Size of subject dimension
    n_subjects = length(subject_list);

    ave_traces = zeros(4, length(tf_times));

    % Re-organize data
    for s = 1 : n_subjects

        % Subject identifier
        subject = subject_list{s};

        % Load ersp data
        load([PATH_TF_DATA, subject, '_ersp_easy_accu.mat']);
        load([PATH_TF_DATA, subject, '_ersp_easy_flip.mat']);
        load([PATH_TF_DATA, subject, '_ersp_hard_accu.mat']);
        load([PATH_TF_DATA, subject, '_ersp_hard_flip.mat']);

        % get dims
        [n_channels, n_freqs, n_times] = size(ersp_easy_accu);

        % Get average traces
        idx_chan = [65, 15, 16, 19, 20];
        idx_freq = tf_freqs >= 4 & tf_freqs <= 5.5;
        ave_traces(1, :) = ave_traces(1, :) + squeeze(mean(ersp_easy_accu(idx_chan, idx_freq, :), [1, 2]))';
        ave_traces(2, :) = ave_traces(2, :) + squeeze(mean(ersp_easy_flip(idx_chan, idx_freq, :), [1, 2]))';
        ave_traces(3, :) = ave_traces(3, :) + squeeze(mean(ersp_hard_accu(idx_chan, idx_freq, :), [1, 2]))';
        ave_traces(4, :) = ave_traces(4, :) + squeeze(mean(ersp_hard_flip(idx_chan, idx_freq, :), [1, 2]))';

    end

    % Scale
    ave_traces = ave_traces ./ n_subjects;

    % Plot
    figure()
    plot(tf_times, ave_traces, 'LineWidth', 2)
    legend({'easy-accu', 'easy-flip', 'hard-accu', 'hard-flip'})

end