#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:35:54 2023

@author: plkn
"""

# Imports
import numpy as np
import os
import pandas as pd
import itertools as it
import random

# Path out
path_out = "/home/plkn/repos/pixelflip/control_files/"

# Number of ids to create files for
ids_pilot = range(1000, 1020)
ids_experiment = range(100)

# Set parameters
n_blocks = 6
n_trials = 100
flip_rate = 0.3

# Iterate participants
for subject_id in it.chain(ids_pilot, ids_experiment):
    
    # Stuff goes here
    all_the_lines = []
    
    # Get cues
    if np.mod(subject_id, 2) == 1:
        cues = {"hard" : "X", "easy" : "O"}
    else:
        cues = {"hard" : "O", "easy" : "X"}
    
    # Iterate blocks
    for block_nr in range(n_blocks):
        
        # Get block reliability and response flip vector
        if np.mod(block_nr + 1, 2) == 1:
            block_reliability = "reliable"
            response_flips = ["no"] * n_trials
        else:
            block_reliability = "unreliable"
            n_noflips = np.int(np.floor(n_trials * (1 - flip_rate)))
            n_flips = n_trials - n_noflips
            response_flips = ["no"] * n_noflips + ["yes"] * n_flips
            random.shuffle(response_flips)
        
        # Line for blockstart
        all_the_lines.append(
            np.array(
                [
                    1,
                    "",
                    "gridBlockStartProc",
                    subject_id + 1,
                    1, # Block start code
                    block_nr + 1,
                    block_reliability,
                    0,
                    "",
                    "",
                    0,
                    0,
                    "",
                ]
            )
        )
        
        # Get list of trial conditions
        trial_difficulties = ["easy"] * np.int((n_trials / 2)) + ["hard"] * np.int((n_trials / 2))
        random.shuffle(trial_difficulties)
        
        # Iterate blocks
        for trial_nr in range(n_trials):
            
            # Get trial difficulty
            if trial_difficulties[trial_nr] == "easy":
                trial_difficulty = np.random.normal(loc=0.3, scale=0.05, size=(1,))[0]
            else:
                trial_difficulty = np.random.normal(loc=0.45, scale=0.05, size=(1,))[0]
                
            # Get color proportions
            color_proportions = [trial_difficulty, 1 - trial_difficulty]
            
            # Randomize colors
            random.shuffle(color_proportions)
            
            # Line for trial
            all_the_lines.append(
                np.array(
                    [
                        1,
                        "",
                        "gridTrialProc",
                        subject_id + 1,
                        3, # Trial code
                        block_nr + 1,
                        block_reliability,
                        trial_nr,
                        trial_difficulties[trial_nr],
                        cues[trial_difficulties[trial_nr]],
                        color_proportions[0],
                        color_proportions[1],
                        response_flips[trial_nr],
                    ]
                )
            )
            
        # Line for blockend
        all_the_lines.append(
            np.array(
                [
                    1,
                    "",
                    "gridBlockEndProc",
                    subject_id + 1,
                    2, # Block end code
                    block_nr + 1,
                    block_reliability,
                    0,
                    "",
                    "",
                    0,
                    0,
                    "",
                ]
            )
        )
            
            
    # Stack lines to array
    all_the_lines = np.stack(all_the_lines)

    # Create data frame
    cols = [
        "Weight",
        "Nested",
        "Procedure",
        "subjectID",
        "event_code",
        "block_nr",
        "block_reliability",
        "trial_nr",
        "trial_difficulty",
        "trial_cue",
        "color_1",
        "color_2",
        "flip_response",
    ]
    df = pd.DataFrame(all_the_lines, columns=cols)
    
    # Save
    fn = os.path.join(path_out, f"pxflp_control_file_{str(subject_id+1)}.csv")
    df.to_csv(fn, sep="\t", index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    