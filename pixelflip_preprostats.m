clear all;

% PATH VARS
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/pixelflip/2_cleaned/';

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

% Result matrices
prepro_cue_erp      = [];
prepro_cue_tf       = [];
prepro_feedback_erp = [];
prepro_feedback_tf  = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Get id stuff
    subject = subject_list{s};
    id = str2num(subject(3 : 4));

    % Load info
    CUE_ERP      = pop_loadset('filename', [subject, '_cleaned_cue_erp.set'],      'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');
    %CUE_TF       = pop_loadset('filename', [subject, '_cleaned_cue_tf.set'],       'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');
    %FEEDBACK_ERP = pop_loadset('filename', [subject, '_cleaned_feedback_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');
    %FEEDBACK_TF  = pop_loadset('filename', [subject, '_cleaned_feedback_tf.set'],  'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Save stats
    prepro_cue_erp(s, :) = [id, length(CUE_ERP.chans_rejected), length(CUE_ERP.rejected_epochs), length(CUE_ERP.nobrainer)];
    %prepro_cue_tf(s, :) = [id, length(CUE_TF.chans_rejected), length(CUE_TF.rejected_epochs), length(CUE_TF.nobrainer)];
    %prepro_feedback_erp(s, :) = [id, length(FEEDBACK_ERP.chans_rejected), length(FEEDBACK_ERP.rejected_epochs), length(FEEDBACK_ERP.nobrainer)];
    %prepro_feedback_tf(s, :) = [id, length(FEEDBACK_TF.chans_rejected), length(FEEDBACK_TF.rejected_epochs), length(FEEDBACK_TF.nobrainer)];

end


