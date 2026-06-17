# Imports
import mne
import glob
import pandas as pd
import numpy as np
import mne.stats 
import mne

# Bins
all_lrp_rows = []

# Define paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")

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
        .crop(tmin=1, tmax=1.8)
    )
    
    # Save times
    erp_times = eeg_epochs.times
    
    # Find motor channels
    c3_idx = eeg_epochs.ch_names.index("C1")
    c4_idx = eeg_epochs.ch_names.index("C2")
    
    # Single-trial EEG in µV
    data_uv = eeg_epochs.get_data() * 1e6
    
    # Extract channels
    c3 = data_uv[:, c3_idx, :]
    c4 = data_uv[:, c4_idx, :]

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv")
    
    # Initialize
    trial_lrp = np.full_like(c3, np.nan)
    
    # Left-hand responses
    left_mask = trialinfo["key_pressed"] == 1
    
    trial_lrp[left_mask] = (
        c4[left_mask]      # contralateral
        - c3[left_mask]    # ipsilateral
    )
    
    # Right-hand responses
    right_mask = trialinfo["key_pressed"] == 2
    
    trial_lrp[right_mask] = (
        c3[right_mask]     # contralateral
        - c4[right_mask]   # ipsilateral
    )
        
    # Get trial indices
    idx_easy_00 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 1)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.key_pressed > 0)
    )[0]
    idx_easy_10 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 0)
        & (trialinfo.key_pressed > 0)
    )[0]
    idx_easy_11 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 1)
        & (trialinfo.key_pressed > 0)
    )[0]
    idx_hard_00 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 1)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.key_pressed > 0)
    )[0]
    idx_hard_10 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 0)
        & (trialinfo.key_pressed > 0)
    )[0]
    idx_hard_11 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 1)
        & (trialinfo.key_pressed > 0)
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
    
    for cond_name, idx in conditions.items():
    
        if len(idx) == 0:
            continue
        
        # Calculate lrp for condition
        lrp = trial_lrp[idx].mean(axis=0)
        
        # Append erp waveforms
        all_lrp_rows.append({
            "subject": subject_id,
            "condition": cond_name,
            "n_trials": len(idx),
            "lrp_uv": lrp,
        })
            
# Create erp df
lrp_df = pd.DataFrame(all_lrp_rows)

# Recode
lrp_df["difficulty"] = lrp_df["condition"].str.split("_").str[0]

lrp_df["difficulty"] = pd.Categorical(
    lrp_df["difficulty"],
    categories=["easy", "hard"],
    ordered=True,
)

lrp_df["flip"] = (
    lrp_df["condition"]
    .str.split("_")
    .str[1]
    .map({
        "00": "contingent",
        "10": "non-contingent",
        "11": "post-flip",
    })
)

lrp_df["flip"] = pd.Categorical(
    lrp_df["flip"],
    categories=["contingent", "non-contingent", "post-flip"],
    ordered=True,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def mean_sem_waveforms(df, waveform_col="lrp_uv"):

    waves = np.stack(
        df[waveform_col].to_numpy(),
        axis=0,
    )

    mean = waves.mean(axis=0)

    sem = (
        waves.std(axis=0, ddof=1)
        / np.sqrt(waves.shape[0])
    )

    return mean, sem


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
target_onset = 1.2
plot_times = erp_times - target_onset

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4),
    sharex=True,
    sharey=True,
)

for ax, difficulty in zip(axes, difficulty_order):

    for flip in flip_order:

        sub = lrp_df[
            (lrp_df["difficulty"] == difficulty)
            & (lrp_df["flip"] == flip)
        ]

        if len(sub) == 0:
            continue

        mean, sem = mean_sem_waveforms(
            sub,
            waveform_col="lrp_uv",
        )

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
    ax.set_xlabel("Time from response / feedback (s)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("LRP amplitude (µV)")

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