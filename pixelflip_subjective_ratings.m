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

