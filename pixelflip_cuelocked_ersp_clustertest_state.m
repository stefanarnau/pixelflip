clear all;

% PATH VARS - PLEASE ADJUST!!!!!
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';
PATH_VEUSZ       = '/mnt/data_dump/pixelflip/veusz/cue_ersp_state/';  
PATH_TF_DATA     = '/mnt/data_dump/pixelflip/3_tf_data/ersps/';

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
n_frq = 50;
frqrange = [2, 30];
tfres_range = [600, 200];

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
prune_times = [-500, 1800]; 
tf_times = EEG.times(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)));

% Get chanlocs
chanlocs = EEG.chanlocs;

% Matrices to collect data. Dimensionality: subjects x channels x times
ersp_flip0_easy_post0 = zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times));
ersp_flip0_hard_post0 = zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times));
ersp_flip1_easy_post0 = zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times));
ersp_flip1_hard_post0 = zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times));
ersp_flip1_easy_post1 = zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times));
ersp_flip1_hard_post1 = zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times));

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
    
    % Get trial-indices of conditions
    idx_flip0_easy_post0 = EEG.trialinfo(:, 3) == 1 & EEG.trialinfo(:, 4) == 0;
    idx_flip0_hard_post0 = EEG.trialinfo(:, 3) == 1 & EEG.trialinfo(:, 4) == 1;
    idx_flip1_easy_post0 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 12) == 0;
    idx_flip1_hard_post0 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 12) == 0;
    idx_flip1_easy_post1 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 12) == 1;
    idx_flip1_hard_post1 = EEG.trialinfo(:, 3) == 0 & EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 12) == 1;

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
        powcube = powcube(:, dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);

        % Get condition general baseline values
        ersp_bl = [-500, -200];
        tmp = squeeze(mean(powcube, 3));
        [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp
        ersp_flip0_easy_post0(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_flip0_easy_post0), 3)), blvals)));
        ersp_flip0_hard_post0(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_flip0_hard_post0), 3)), blvals)));
        ersp_flip1_easy_post0(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_flip1_easy_post0), 3)), blvals)));
        ersp_flip1_hard_post0(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_flip1_hard_post0), 3)), blvals)));
        ersp_flip1_easy_post1(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_flip1_easy_post1), 3)), blvals)));
        ersp_flip1_hard_post1(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_flip1_hard_post1), 3)), blvals)));

    end % end channel loop

end

% Save shit
save([PATH_TF_DATA, 'chanlocs.mat'], 'chanlocs');
save([PATH_TF_DATA, 'tf_freqs.mat'], 'tf_freqs');
save([PATH_TF_DATA, 'tf_times.mat'], 'tf_times');
save([PATH_TF_DATA, 'ersp_flip0_easy_post0.mat'], 'ersp_flip0_easy_post0');
save([PATH_TF_DATA, 'ersp_flip0_hard_post0.mat'], 'ersp_flip0_hard_post0');
save([PATH_TF_DATA, 'ersp_flip1_easy_post0.mat'], 'ersp_flip1_easy_post0');
save([PATH_TF_DATA, 'ersp_flip1_hard_post0.mat'], 'ersp_flip1_hard_post0');
save([PATH_TF_DATA, 'ersp_flip1_easy_post1.mat'], 'ersp_flip1_easy_post1');
save([PATH_TF_DATA, 'ersp_flip1_hard_post1.mat'], 'ersp_flip1_hard_post1');

% The order of things
new_order_labels = {...
'Fp1',...
'Fp2',...
'AF7',...
'AF3',...
'AF4',...
'AF8',...
'F9',...
'F5',...
'F3',...
'F1',...
'Fz',...
'F2',...
'F4',...
'F6',...
'F10',...
'FT9',...
'FT7',...
'FC3',...
'FC1',...
'FCz',...
'FC2',...
'FC4',...
'FT8',...
'FT10',...
'T7',...
'C5',...
'C3',...
'C1',...
'Cz',...
'C2',...
'C4',...
'C6',...
'T8',...
'TP9',...
'TP7',...
'CP3',...
'CP1',...
'CPz',...
'CP2',...
'CP4',...
'TP8',...
'TP10',...
'P9',...
'P7',...
'P5',...
'P3',...
'P1',...
'Pz',...
'P2',...
'P4',...
'P6',...
'P8',...
'P10',...
'PO9',...
'PO7',...
'PO3',...
'POz',...
'PO4',...
'PO8',...
'PO10',...
'O9',...
'O1',...
'Oz',...
'O2',...
'O10',...
};

% Get original chanloc labels
chanlocs_labels = {};
for ch = 1 : length(chanlocs)
    chanlocs_labels{end + 1} = chanlocs(ch).labels;
end

% Get new order indices
new_order_idx = [];
for ch = 1 : length(chanlocs)
    new_order_idx(end + 1) = find(strcmpi(new_order_labels{ch}, chanlocs_labels));
end

% Install new order
ersp_flip0_easy_post0 = ersp_flip0_easy_post0(:, new_order_idx, :, :);
ersp_flip0_hard_post0 = ersp_flip0_hard_post0(:, new_order_idx, :, :);
ersp_flip1_easy_post0 = ersp_flip1_easy_post0(:, new_order_idx, :, :);
ersp_flip1_hard_post0 = ersp_flip1_hard_post0(:, new_order_idx, :, :);
ersp_flip1_easy_post1 = ersp_flip1_easy_post1(:, new_order_idx, :, :);
ersp_flip1_hard_post1 = ersp_flip1_hard_post1(:, new_order_idx, :, :);
chanlocs = chanlocs(new_order_idx);

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

% A template for GA structs
cfg=[];
cfg.keepindividual = 'yes';
ga_template = [];
ga_template.dimord = 'chan_freq_time';
ga_template.label = chanlabs;
ga_template.freq      = tf_freqs;
ga_template.time      = tf_times;

% ############################### noflip vs. flip-postnoflip ###########################################

% GA struct easy
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(ersp_flip0_easy_post0(s, :, :, :)) + squeeze(ersp_flip1_easy_post0(s, :, :, :))) ./ 2;
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_easy = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct hard
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(ersp_flip0_hard_post0(s, :, :, :)) + squeeze(ersp_flip1_hard_post0(s, :, :, :))) ./ 2;
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_hard = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct noflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(ersp_flip0_easy_post0(s, :, :, :)) + squeeze(ersp_flip0_hard_post0(s, :, :, :))) ./ 2;
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_noflip = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct flip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = (squeeze(ersp_flip1_easy_post0(s, :, :, :)) + squeeze(ersp_flip1_hard_post0(s, :, :, :))) ./ 2;
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_flip = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct hard minus easy noflip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_flip0_hard_post0(s, :, :, :)) - squeeze(ersp_flip0_easy_post0(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_interaction_noflip = ft_freqgrandaverage(cfg, GA{1, :});

% GA struct hard minus easy flip
GA = {};
for s = 1 : length(subject_list)
    chan_time_data = squeeze(ersp_flip1_hard_post0(s, :, :, :)) - squeeze(ersp_flip1_easy_post0(s, :, :, :));
    ga_template.powspctrm = chan_time_data;
    GA{s} = ga_template;
end 
GA_interaction_flip = ft_freqgrandaverage(cfg, GA{1, :});

% Testparams
testalpha  = 0.05;
voxelalpha  = 0.01;
nperm = 1000;

% Set config
cfg = [];
cfg.tail             = 1;
cfg.statistic        = 'depsamplesFmultivariate';
cfg.alpha            = testalpha;
cfg.neighbours       = neighbours;
cfg.minnbchan        = 2;
cfg.method           = 'montecarlo';
cfg.correctm         = 'cluster';
cfg.clustertail      = 1;
cfg.clusteralpha     = voxelalpha;
cfg.clusterstatistic = 'maxsum';
cfg.numrandomization = nperm;
cfg.computecritval   = 'yes'; 
cfg.ivar             = 1;
cfg.uvar             = 2;

% Set up design
n_subjects = length(subject_list);
design = zeros(2, n_subjects * 2);
design(1, :) = [ones(1, n_subjects), 2 * ones(1, n_subjects)];
design(2, :) = [1 : n_subjects, 1 : n_subjects];
cfg.design = design;

% The tests
[stat_difficulty]  = ft_freqstatistics(cfg, GA_easy, GA_hard);
[stat_agency]      = ft_freqstatistics(cfg, GA_noflip, GA_flip);
[stat_interaction] = ft_freqstatistics(cfg, GA_interaction_noflip, GA_interaction_flip);

% Save cluster structs
save([PATH_TF_DATA 'stat_difficulty.mat'], 'stat_difficulty');
save([PATH_TF_DATA 'stat_agency.mat'], 'stat_agency');
save([PATH_TF_DATA 'stat_interaction.mat'], 'stat_interaction');

% Calculate and save effect sizes
adjpetasq_difficulty = [];
adjpetasq_agency = [];
adjpetasq_interaction = [];
for ch = 1 : EEG.nbchan
    petasq = (squeeze(stat_difficulty.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_difficulty.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    adjpetasq_difficulty(ch, :, :) = adj_petasq;

    petasq = (squeeze(stat_agency.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_agency.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    adjpetasq_agency(ch, :, :) = adj_petasq;

    petasq = (squeeze(stat_interaction.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_interaction.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    adjpetasq_interaction(ch, :, :) = adj_petasq;
end

% Save effect sizes
save([PATH_TF_DATA, 'adjpetasq_difficulty.mat'], 'adjpetasq_difficulty');
save([PATH_TF_DATA, 'adjpetasq_agency.mat'], 'adjpetasq_agency');
save([PATH_TF_DATA, 'adjpetasq_interaction.mat'], 'adjpetasq_interaction');

% Plot some effect sizes
figure()
pd = squeeze(adjpetasq_agency(20, :, :));
contourf(tf_times, tf_freqs, pd, 40, 'linecolor','none')
clim([0, 0.5])
colormap(hot)


% Identify significant clusters
clust_thresh = 0.05;
clusts = struct();
cnt = 0;
stat_names = {'stat_difficulty', 'stat_agency', 'stat_interaction'};
for s = 1 : numel(stat_names)
    stat = eval(stat_names{s});
    if ~isempty(stat.posclusters)
        pos_idx = find([stat.posclusters(1, :).prob] < clust_thresh);
        for c = 1 : numel(pos_idx)
            cnt = cnt + 1;
            clusts(cnt).testlabel = stat_names{s};
            clusts(cnt).clustnum = cnt;
            clusts(cnt).time = stat.time;
            clusts(cnt).freq = stat.freq;
            clusts(cnt).prob = stat.posclusters(1, pos_idx(c)).prob;
            clusts(cnt).idx = stat.posclusterslabelmat == pos_idx(c);
            clusts(cnt).stats = clusts(cnt).idx .* stat.stat;
            clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2, 3])));
        end
    end
end

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
    title(['sum t across chans '], 'FontSize', 10)

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

    saveas(gcf, [PATH_VEUSZ 'clustnum_' num2str(clusts(cnt).clustnum) '_' clusts(cnt).testlabel '.png']); 
end

aa=bb





% Save masks
dlmwrite([PATH_VEUSZ, 'contour_difficulty.csv'], stat_difficulty.mask);
dlmwrite([PATH_VEUSZ, 'contour_agency.csv'], stat_agency.mask);
dlmwrite([PATH_VEUSZ, 'contour_interaction.csv'], stat_interaction.mask);

% Calculate effect sizes
n_chans = numel(chanlocs);
apes_difficulty  = [];
apes_agency      = [];
apes_interaction = [];
df_effect = 1;
for ch = 1 : n_chans
    petasq = (squeeze(stat_difficulty.stat(ch, :)) * df_effect) ./ ((squeeze(stat_difficulty.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_difficulty(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
    petasq = (squeeze(stat_agency.stat(ch, :)) * df_effect) ./ ((squeeze(stat_agency.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_agency(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
    petasq = (squeeze(stat_interaction.stat(ch, :)) * df_effect) ./ ((squeeze(stat_interaction.stat(ch, :)) * df_effect) + (n_subjects - 1));
    apes_interaction(ch, :) = petasq - (1 - petasq) .* (df_effect / (n_subjects - 1));
end

% Save effect sizes
dlmwrite([PATH_VEUSZ, 'apes_difficulty.csv'], apes_difficulty);
dlmwrite([PATH_VEUSZ, 'apes_agency.csv'], apes_agency);
dlmwrite([PATH_VEUSZ, 'apes_interaction.csv'], apes_interaction);

% Save lineplots at Fz
dlmwrite([PATH_VEUSZ, 'lineplots_fz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post0(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post1(:, 11, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post1(:, 11, :)), 1)]);

% Save lineplots at FCz
dlmwrite([PATH_VEUSZ, 'lineplots_fcz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post0(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post1(:, 20, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post1(:, 20, :)), 1)]);
% Save lineplots at Fz
dlmwrite([PATH_VEUSZ, 'lineplots_cz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post0(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post1(:, 29, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post1(:, 29, :)), 1)]);

% Save lineplots at FCz
dlmwrite([PATH_VEUSZ, 'lineplots_cpz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post0(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post1(:, 38, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post1(:, 38, :)), 1)]);

% Save lineplots at Pz
dlmwrite([PATH_VEUSZ, 'lineplots_pz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip0_hard_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post0(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_easy_post1(:, 48, :)), 1);...
                                             mean(squeeze(erp_flip1_hard_post1(:, 48, :)), 1)]);

% Save lineplots at POz
dlmwrite([PATH_VEUSZ, 'lineplots_poz.csv'],  [mean(squeeze(erp_flip0_easy_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip0_hard_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post0(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_easy_post1(:, 57, :)), 1);...
                                              mean(squeeze(erp_flip1_hard_post1(:, 57, :)), 1)]);

% Save erp-times
dlmwrite([PATH_VEUSZ, 'erp_times.csv'], erp_times);

% Plot effect size topos at selected time points for agency
clim = [-0.3, 0.3];
tpoints = [550, 900, 1400];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_agency(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_agency_', num2str(tpoints(t)), 'ms', '.png']);
end

% Plot effect size topos at selected time points for difficulty
clim = [-0.3, 0.3];
tpoints = [120, 420, 1600];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_difficulty(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_difficulty_', num2str(tpoints(t)), 'ms', '.png']);
end

% Plot effect size topos at selected time points for interaction
clim = [-0.3, 0.3];
tpoints = [150, 450, 1700];
for t = 1 : length(tpoints)
    figure('Visible', 'off'); clf;
    tidx = erp_times >= tpoints(t) - 5 & erp_times <= tpoints(t) + 5;
    pd = mean(apes_interaction(:, tidx), 2);
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    colormap(jet);
    caxis(clim);
    saveas(gcf, [PATH_VEUSZ, 'topo_interaction_', num2str(tpoints(t)), 'ms', '.png']);
end