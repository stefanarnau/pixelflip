clear all;

% PATH VARS
PATH_EEGLAB = '/home/plkn/eeglab2022.1/';
PATH_IN     = '/mnt/data_dump/pixelflip/2_cleaned/';

% Subject list
subject_list = {'VP01', 'VP02', 'VP03', 'VP05', 'VP06', 'VP08', 'VP12', 'VP07',...
               'VP11', 'VP09', 'VP16', 'VP17', 'VP19', 'VP21', 'VP23', 'VP25',...
               'VP27', 'VP29', 'VP31', 'VP18', 'VP20', 'VP22', 'VP24', 'VP26',...
               'VP28'};

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
    CUE_ERP      = pop_loadset('filename', [subject, '_cleaned_cue_erp.set'],      'filepath', PATH_IN, 'loadmode', 'info');
    CUE_TF       = pop_loadset('filename', [subject, '_cleaned_cue_tf.set'],       'filepath', PATH_IN, 'loadmode', 'info');
    FEEDBACK_ERP = pop_loadset('filename', [subject, '_cleaned_feedback_erp.set'], 'filepath', PATH_IN, 'loadmode', 'info');
    FEEDBACK_TF  = pop_loadset('filename', [subject, '_cleaned_feedback_tf.set'],  'filepath', PATH_IN, 'loadmode', 'info');

    % Save stats
    prepro_cue_erp(s, :) = [id, length(CUE_ERP.chans_rejected), length(CUE_ERP.rejected_epochs), length(CUE_ERP.nobrainer)];
    prepro_cue_tf(s, :) = [id, length(CUE_TF.chans_rejected), length(CUE_TF.rejected_epochs), length(CUE_TF.nobrainer)];
    prepro_feedback_erp(s, :) = [id, length(FEEDBACK_ERP.chans_rejected), length(FEEDBACK_ERP.rejected_epochs), length(FEEDBACK_ERP.nobrainer)];
    prepro_feedback_tf(s, :) = [id, length(FEEDBACK_TF.chans_rejected), length(FEEDBACK_TF.rejected_epochs), length(FEEDBACK_TF.nobrainer)];

end


