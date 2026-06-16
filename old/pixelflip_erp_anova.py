#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:52:54 2021

@author: Stefan Arnau
"""

# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import mne.stats as stats

# Define paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"
path_res = "/mnt/data_dump/pixelflip/results_anova/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")


# Function for plotting erps and calculate stats
def get_erpplot_and_stats(electrode_selection, stat_label, timewin_stats):

    idx_channel = [
        idx
        for idx, element in enumerate(eeg_epochs.ch_names)
        if element in electrode_selection
    ]
    df_rows = []
    for idx_id, id in enumerate(ids):

        for idx_t, t in enumerate(erp_times):

            if (t >= timewin_stats[0]) & (t <= timewin_stats[1]):
                in_statwin = 1
            else:
                in_statwin = 0

            df_rows.append(
                {
                    "id": id,
                    "SoA": "noflip",
                    "difficulty": "easy",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_easy_00[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "SoA": "postnoflip",
                    "difficulty": "easy",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_easy_10[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "SoA": "postflip",
                    "difficulty": "easy",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_easy_11[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "SoA": "noflip",
                    "difficulty": "hard",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_hard_00[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "SoA": "postnoflip",
                    "difficulty": "hard",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_hard_10[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "SoA": "postflip",
                    "difficulty": "hard",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_hard_11[idx_id, idx_channel, idx_t].mean(),
                }
            )

    # Get dataframe
    df_frontal_erp = pd.DataFrame(df_rows)

    # Create ERP Lineplot
    sns.set_style("darkgrid")
    sns.relplot(
        data=df_frontal_erp,
        x="time (s)",
        y="V",
        hue="SoA",
        style="difficulty",
        kind="line",
        height=3,
        aspect=1.8,
        errorbar=None,
        palette="rocket",
    )

    # Save plot
    plt.savefig(
        os.path.join(path_res, "lineplot_" + stat_label + ".png"),
        dpi=300,
        transparent=True,
    )

    # Get dataframe statistical analysis (average is timewin)
    df_stats = df_frontal_erp.drop(df_frontal_erp[df_frontal_erp.in_statwin != 1].index)
    df_stats = df_stats.groupby(["id", "SoA", "difficulty"])["V"].mean().reset_index()

    # Draw a pointplot
    g = sns.catplot(
        data=df_stats,
        x="difficulty",
        y="V",
        hue="SoA",
        capsize=0.2,
        palette="rocket",
        errorbar="se",
        kind="point",
        height=6,
        aspect=0.75,
    )
    g.despine(left=True)

    # Save plot
    plt.savefig(
        os.path.join(path_res, "interaction_plot_" + stat_label + ".png"),
        dpi=300,
        transparent=True,
    )

    # Save dataframe
    df_stats.to_csv(os.path.join(path_res, "stats_table_" + stat_label + ".csv"))

    # Return stat dataframe
    return df_stats


# Collector lists
matrices_easy_00 = []
matrices_easy_10 = []
matrices_easy_11 = []
matrices_hard_00 = []
matrices_hard_10 = []
matrices_hard_11 = []
ids = []

# Loop datasets
for dataset in datasets:

    # Get id
    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])
    ids.append(subject_id)

    # Skip VP 07 (age outlier)
    if ids[-1] == 7:
        ids.pop()
        continue

    # Load a dataset
    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(-0.2, -0))
        .crop(tmin=-0.2, tmax=1.8)
    )

    # Save times
    erp_times = eeg_epochs.times

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv")

    # Get trial indices
    idx_easy_00 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 1)
        & (trialinfo.prev_accuracy == 1)
    )[0]
    idx_easy_10 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 0)
    )[0]
    idx_easy_11 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 1)
    )[0]
    idx_hard_00 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 1)
        & (trialinfo.prev_accuracy == 1)
    )[0]
    idx_hard_10 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 0)
    )[0]
    idx_hard_11 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 1)
    )[0]

    # Get evokeds
    evoked_easy_00 = eeg_epochs[idx_easy_00].average()
    evoked_easy_10 = eeg_epochs[idx_easy_10].average()
    evoked_easy_11 = eeg_epochs[idx_easy_11].average()
    evoked_hard_00 = eeg_epochs[idx_hard_00].average()
    evoked_hard_10 = eeg_epochs[idx_hard_10].average()
    evoked_hard_11 = eeg_epochs[idx_hard_11].average()

    # Collect as matrices
    matrices_easy_00.append(evoked_easy_00.data)
    matrices_easy_10.append(evoked_easy_10.data)
    matrices_easy_11.append(evoked_easy_11.data)
    matrices_hard_00.append(evoked_hard_00.data)
    matrices_hard_10.append(evoked_hard_10.data)
    matrices_hard_11.append(evoked_hard_11.data)

# Stack matrices
matrices_easy_00 = np.stack(matrices_easy_00)
matrices_easy_10 = np.stack(matrices_easy_10)
matrices_easy_11 = np.stack(matrices_easy_11)
matrices_hard_00 = np.stack(matrices_hard_00)
matrices_hard_10 = np.stack(matrices_hard_10)
matrices_hard_11 = np.stack(matrices_hard_11)


# Get ERP Fp
df_stats_Fp_early = get_erpplot_and_stats(
    electrode_selection=["Fp1", "Fp2"], stat_label="Fp", timewin_stats=(0.4, 0.8)
)
aov_Fp_early = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Fp_early,
    detailed=True,
    effsize="np2",
)


df_stats_Fp_late = get_erpplot_and_stats(
    electrode_selection=["Fp1", "Fp2"], stat_label="Fp", timewin_stats=(0.8, 1.2)
)
aov_Fp_late = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Fp_late,
    detailed=True,
    effsize="np2",
)

# Get ERP Fz
df_stats_Fz_early = get_erpplot_and_stats(
    electrode_selection=["Fz", "F1", "F2"], stat_label="Fz", timewin_stats=(0.4, 0.8)
)
aov_Fz_early = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Fz_early,
    detailed=True,
    effsize="np2",
)


df_stats_Fz_late = get_erpplot_and_stats(
    electrode_selection=["Fz", "F1", "F2"], stat_label="Fz", timewin_stats=(0.8, 1.2)
)
aov_Fz_late = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Fz_late,
    detailed=True,
    effsize="np2",
)

# Get ERP FCz
df_stats_FCz_early = get_erpplot_and_stats(
    electrode_selection=["FCz", "FC1", "FC2"],
    stat_label="FCz",
    timewin_stats=(0.4, 0.8),
)
aov_FCz_early = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_FCz_early,
    detailed=True,
    effsize="np2",
)


df_stats_FCz_late = get_erpplot_and_stats(
    electrode_selection=["FCz", "FC1", "FC2"],
    stat_label="FCz",
    timewin_stats=(0.8, 1.2),
)
aov_FCz_late = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_FCz_late,
    detailed=True,
    effsize="np2",
)


# Get ERP Cz
df_stats_Cz_early = get_erpplot_and_stats(
    electrode_selection=["Cz", "C1", "C2"],
    stat_label="Cz",
    timewin_stats=(0.4, 0.8),
)
aov_Cz_early = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Cz_early,
    detailed=True,
    effsize="np2",
)


df_stats_Cz_late = get_erpplot_and_stats(
    electrode_selection=["Cz", "C1", "C2"],
    stat_label="Cz",
    timewin_stats=(0.8, 1.2),
)
aov_Cz_late = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Cz_late,
    detailed=True,
    effsize="np2",
)

# Get ERP Pz
df_stats_Pz_early = get_erpplot_and_stats(
    electrode_selection=["Pz", "P1", "P2"],
    stat_label="Pz",
    timewin_stats=(0.4, 0.8),
)
aov_Pz_early = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Pz_early,
    detailed=True,
    effsize="np2",
)


df_stats_Pz_late = get_erpplot_and_stats(
    electrode_selection=["Pz", "P1", "P2"],
    stat_label="Pz",
    timewin_stats=(0.8, 1.2),
)
aov_Pz_late = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_Pz_late,
    detailed=True,
    effsize="np2",
)

# Get ERP POz
df_stats_POz_early = get_erpplot_and_stats(
    electrode_selection=["POz", "PO1", "PO2"],
    stat_label="POz",
    timewin_stats=(0.4, 0.8),
)
aov_POz_early = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_POz_early,
    detailed=True,
    effsize="np2",
)


df_stats_POz_late = get_erpplot_and_stats(
    electrode_selection=["POz", "PO1", "PO2"],
    stat_label="POz",
    timewin_stats=(0.8, 1.2),
)
aov_POz_late = pg.rm_anova(
    dv="V",
    within=["SoA", "difficulty"],
    subject="id",
    data=df_stats_POz_late,
    detailed=True,
    effsize="np2",
)
