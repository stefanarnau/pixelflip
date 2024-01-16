clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_TF_DATA     = '/mnt/data_dump/pixelflip/3_tf_data/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP04', 'VP05', 'VP06', 'VP07', 'VP08', 'VP09', 'VP10',...
                'VP11', 'VP12', 'VP13', 'VP14', 'VP15', 'VP16', 'VP17', 'VP18', 'VP19', 'VP20',...
                'VP21', 'VP22', 'VP23', 'VP24', 'VP25', 'VP26', 'VP27', 'VP28', 'VP29', 'VP30',...
                'VP31', 'VP32', 'VP33', 'VP34', 'VP35', 'VP36', 'VP37', 'VP38', 'VP39', 'VP40'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Init fieldtrip
ft_path = '/home/plkn/fieldtrip-master/';
addpath(ft_path);
ft_defaults;

% Load metadata
load([PATH_TF_DATA, 'chanlocs.mat']);
load([PATH_TF_DATA, 'tf_freqs.mat']);
load([PATH_TF_DATA, 'tf_times.mat']);
load([PATH_TF_DATA, 'neighbours.mat']);
load([PATH_TF_DATA, 'obs_fwhmT.mat']);
load([PATH_TF_DATA, 'obs_fwhmF.mat']);

% Remember number of trials of conditions
n_trials = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Get subject id as string
    subject = subject_list{s};

    % Load trialinfo
    load([PATH_TF_DATA, 'trialinfo_', subject, '.mat']);

    % TODO


end % End subject loop