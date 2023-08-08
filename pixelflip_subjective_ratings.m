% Clear stuff
clear all;

% Path vars
PATH_IN  = '/mnt/data_dump/pixelflip/4_subjective_ratings/';  
PATH_OUT = '/mnt/data_dump/pixelflip/veusz/subjecctive_ratings/';  

% The file
fn = 'subjective_ratings_final.csv';

% Load file
T = readtable([PATH_IN, fn]);

% Rename variables (columns)
T = renamevars(T, ["Frage1", "Frage2", "Frage3", "Frage4", "Frage5", "Frage6", "Frage7", "Frage8"],...
                  ["focus_easy", "focus_hard", "focus_accu", "focus_flip", "moti_accu","moti_flip", "mw_accu","mw_flip"]);

% Exclude age outliers
T(T.age > 35, :) = [];

% T-tests
[focus_sig, focus_p, focus_ci, focus_stat] = ttest(T.focus_accu, T.focus_flip);
[moti_sig, moti_p, moti_ci, moti_stat] = ttest(T.moti_accu, T.moti_flip);
[mw_sig, mw_p, mw_ci, mw_stat] = ttest(T.mw_accu, T.mw_flip);

% Effect sizes
apes_focus = apes_from_t(focus_stat.tstat, 39);
apes_moti = apes_from_t(moti_stat.tstat, 39);
apes_mw = apes_from_t(mw_stat.tstat, 39);

% Average for plot
self_reports_out = [mean(T.focus_accu), std(T.focus_accu), mean(T.moti_accu), std(T.moti_accu), mean(T.mw_accu), std(T.mw_accu);...
                    mean(T.focus_flip), std(T.focus_flip), mean(T.moti_flip), std(T.moti_flip), mean(T.mw_flip), std(T.mw_flip)];

% Save
dlmwrite([PATH_OUT, 'self_reports_veusz.csv'], self_reports_out, 'delimiter', '\t');
dlmwrite([PATH_OUT, 'xax12.csv'], [1, 2], 'delimiter', '\t');

