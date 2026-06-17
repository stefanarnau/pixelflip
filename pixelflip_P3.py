# Imports
import mne
import glob
import pandas as pd
import numpy as np
import mne.stats 
import mne

# Bins
all_erp_rows = []

# Define paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")


# ============================================================================================

# Loop datasets
for dataset in datasets:

    # Get id
    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])

    # Skip VP 07 (age outlier)
    if subject_id == 7:
        continue

    # Load a dataset
    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(1, 1.2))
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
    
    conditions = {
        "easy_00": idx_easy_00,
        "easy_10": idx_easy_10,
        "easy_11": idx_easy_11,
        "hard_00": idx_hard_00,
        "hard_10": idx_hard_10,
        "hard_11": idx_hard_11,
    }
    
    # Get fs
    sfreq = eeg_epochs.info["sfreq"]
    
    # Get channel labels
    ch_names = eeg_epochs.ch_names
    
    for cond_name, idx in conditions.items():
    
        if len(idx) == 0:
            continue
        
        # Calculate evoked for condition
        evoked = eeg_epochs[idx].average()
        
        # Get erp times
        erp_times = evoked.times
        
        # Append erp waveforms
        all_erp_rows.append({
            "subject": subject_id,
            "condition": cond_name,
            "n_trials": len(idx),
            "slow_wave_uv": evoked.data * 1e6,
        })
            
# Create erp df
erp_df = pd.DataFrame(all_erp_rows)

# Recode
erp_df["difficulty"] = erp_df["condition"].str.split("_").str[0]

erp_df["difficulty"] = pd.Categorical(
    erp_df["difficulty"],
    categories=["easy", "hard"],
    ordered=True,
)

erp_df["flip"] = (
    erp_df["condition"]
    .str.split("_")
    .str[1]
    .map({
        "00": "contingent",
        "10": "non-contingent",
        "11": "post-flip",
    })
)

erp_df["flip"] = pd.Categorical(
    erp_df["flip"],
    categories=["contingent", "non-contingent", "post-flip"],
    ordered=True,
)

df_state = erp_df[
    erp_df["flip"].isin(["contingent", "non-contingent"])
].copy()

df_sequence = erp_df[
    erp_df["flip"].isin(["non-contingent", "post-flip"])
].copy()

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

target_onset = 1.2

flip_order = [
    "contingent",
    "non-contingent",
    "post-flip",
]

difficulty_order = [
    "easy",
    "hard",
]

palette = {
    "contingent": "#1b9e77",
    "non-contingent": "#d81b60",
    "post-flip": "#7570b3",
}

roi = ["Pz", "P1","P2"]  # N2/frontocentral first
roi_idx = mne.pick_channels(
    ch_names,
    include=[ch for ch in roi if ch in ch_names],
)

plot_time = (1, 1.8)
time_mask = (
    (erp_times >= plot_time[0])
    & (erp_times <= plot_time[1])
)

plot_times = erp_times[time_mask] - target_onset  # target-locked display


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def get_roi_waveforms(sub_df):

    waves = []

    for arr in sub_df["slow_wave_uv"]:
        roi_wave = arr[roi_idx, :].mean(axis=0)
        waves.append(roi_wave[time_mask])

    waves = np.stack(waves, axis=0)

    mean = waves.mean(axis=0)
    sem = waves.std(axis=0, ddof=1) / np.sqrt(waves.shape[0])

    return mean, sem


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4),
    sharex=True,
    sharey=True,
)

for ax, difficulty in zip(axes, difficulty_order):

    for flip in flip_order:

        sub = erp_df[
            (erp_df["difficulty"] == difficulty)
            & (erp_df["flip"] == flip)
        ]

        if len(sub) == 0:
            continue

        mean, sem = get_roi_waveforms(sub)

        ax.plot(
            plot_times,
            mean,
            color=palette[flip],
            linewidth=2.5,
            label=flip,
        )

        ax.fill_between(
            plot_times,
            mean - sem,
            mean + sem,
            color=palette[flip],
            alpha=0.20,
            linewidth=0,
        )

    ax.axvline(
        0,
        color="black",
        linestyle=":",
        linewidth=2,
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(difficulty)
    ax.set_xlabel("Time from target (s)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Amplitude (µV)")

handles, labels = axes[1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
)

plt.tight_layout()
plt.show()