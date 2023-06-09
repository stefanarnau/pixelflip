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
import random

# Path out
path_out = "/home/plkn/repos/pixelflip/control_files/"

# Number of ids to create files for
ids_experiment = range(64)

# Set parameters
n_blocks = 6
n_trials = 80
flip_rate = 0.3

# Iterate participants
for subject_id in ids_experiment:
    
    # Stuff goes here
    all_the_lines = []
    
    # Get cues
    cues = {"hard" : "X", "easy" : "0"}
    
    if (subject_id in range(16)) | (subject_id in range(32, 48)):
        block_reliability_order = ["reliable",  "unreliable", "reliable", "unreliable", "reliable", "unreliable"]
    else:
        block_reliability_order = ["reliable",  "unreliable", "unreliable", "reliable", "unreliable", "reliable"]
    
    # Iterate blocks
    for block_nr in range(n_blocks):
        block_nr
        # Get block reliability and response flip vector
        if block_reliability_order[block_nr] == "reliable":
            response_flips = ["no"] * n_trials
            procstring = "gridBlockReliableStartProc"
        else:
            n_noflips = int(np.floor(n_trials * (1 - flip_rate)))
            n_flips = n_trials - n_noflips
            response_flips = ["no"] * n_noflips + ["yes"] * n_flips
            random.shuffle(response_flips)
            procstring = "gridBlockUnreliableStartProc"
        
        # Line for blockstart
        all_the_lines.append(
            np.array(
                [
                    1,
                    "",
                    procstring,
                    subject_id + 1,
                    1, # Block start code
                    block_nr + 1,
                    block_reliability_order[block_nr],
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
        trial_difficulties = ["easy"] * int((n_trials / 2)) + ["hard"] * int((n_trials / 2))
        random.shuffle(trial_difficulties)
        
        # Iterate blocks
        for trial_nr in range(n_trials):
            
            # Get trial difficulty
            if trial_difficulties[trial_nr] == "easy":
                #trial_difficulty = np.random.normal(loc=0.3, scale=0.05, size=(1,))[0]
                trial_difficulty = np.random.uniform(0.25, 0.32, (1,))[0]
            else:
                #trial_difficulty = np.random.normal(loc=0.45, scale=0.05, size=(1,))[0]
                trial_difficulty = np.random.uniform(0.42, 0.49, (1,))[0]
                
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
                        block_reliability_order[block_nr],
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
                    block_reliability_order[block_nr],
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    