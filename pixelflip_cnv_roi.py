# Imports
import mne
import glob
import pandas as pd
import numpy as np
import mne.stats 
import mne
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Bins
all_erp_rows = []

# Define paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")

# Store single-trial rows here
all_trial_rows = []

# Define ROI once before loop
roi_space = ["FCz", "Fz", "Cz", "FC1", "FC2"]
roi_time = (0.7, 1.2)

# Helpers ====================================================================================
def plot_measure(
    plot_df,
    y,
    ylabel,
    ax,
    title=None,
    ylim=None,
):

    palette = {
        "easy": "#1b9e77",
        "hard": "#d81b60",
    }

    sns.pointplot(
        data=plot_df,
        x="condition",
        y=y,
        hue="difficulty_label",
        errorbar=("ci", 95),
        capsize=.1,
        err_kws={"linewidth": 1.8},
        dodge=0.25,
        markers="o",
        linestyles="-",
        linewidth=2.5,
        markersize=7,
        palette=palette,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x="condition",
        y=y,
        hue="difficulty_label",
        dodge=True,
        jitter=0.08,
        alpha=0.35,
        size=3,
        palette=palette,
        ax=ax,
    )

    # Save legend handles before removing legends
    handles, labels = ax.get_legend_handles_labels()

    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    sns.despine(ax=ax)

    return handles[:2]

# Loop datasets
for dataset in datasets:

    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])

    if subject_id == 7:
        continue

    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(-0.2, 0))
        .crop(tmin=-0.2, tmax=1.8)
    )

    erp_times = eeg_epochs.times

    trialinfo = pd.read_csv(
        dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv"
    )

    assert len(trialinfo) == len(eeg_epochs), dataset

    # ROI indices must be defined after loading, because channel names come from this file
    roi_chs = [ch for ch in roi_space if ch in eeg_epochs.ch_names]

    missing = set(roi_space) - set(roi_chs)
    if missing:
        print(f"VP{subject_id}: missing ROI channels: {missing}")

    roi_idx = mne.pick_channels(
        eeg_epochs.ch_names,
        include=roi_chs,
    )

    roi_time_mask = (
        (eeg_epochs.times >= roi_time[0])
        & (eeg_epochs.times <= roi_time[1])
    )

    # Single-trial CNV
    data_uv = eeg_epochs.get_data() * 1e6
    trial_cnv = data_uv[:, roi_idx, :][:, :, roi_time_mask].mean(axis=(1, 2))

    # Optional sign convention: larger = stronger CNV negativity
    trial_cnv_strength = -trial_cnv

    # Trial-level condition coding
    trialinfo = trialinfo.copy()
    trialinfo["subject"] = subject_id
    trialinfo["trial_idx"] = np.arange(len(trialinfo))
    
    trialinfo["difficulty_label"] = np.where(
        trialinfo["difficulty"] == 0,
        "easy",
        "hard",
    )
    
    trialinfo["condition"] = np.nan
    
    trialinfo.loc[
        trialinfo["reliability"] == 1,
        "condition",
    ] = "contingent"
    
    trialinfo.loc[
        (trialinfo["reliability"] == 0)
        & (trialinfo["prev_flipped"] == 0),
        "condition",
    ] = "non-contingent"
    
    trialinfo.loc[
        (trialinfo["reliability"] == 0)
        & (trialinfo["prev_flipped"] == 1),
        "condition",
    ] = "post-flip"
    
    trialinfo["cnv_value"] = trial_cnv
    trialinfo["cnv_strength"] = trial_cnv_strength
    
    keep = (
        (trialinfo["prev_accuracy"] == 1)
        & (trialinfo["condition"].notna())
    )
    
    all_trial_rows.append(
        trialinfo.loc[keep].copy()
    )

# Final single-trial dataframe
trial_cnv_df = pd.concat(
    all_trial_rows,
    ignore_index=True,
)

trial_cnv_df["difficulty_label"] = pd.Categorical(
    trial_cnv_df["difficulty_label"],
    categories=["easy", "hard"],
    ordered=True,
)

trial_cnv_df["condition"] = pd.Categorical(
    trial_cnv_df["condition"],
    categories=["contingent", "non-contingent", "post-flip"],
    ordered=True,
)

df = trial_cnv_df.copy()

# CNV inference =========================================================================

fit_cnv = smf.mixedlm(
    "cnv_value ~ difficulty_label * condition",
    data=df,
    groups=df["subject"],
).fit(
    method="powell",
    reml=False,
)

print(fit_cnv.summary())

b = (
    fit_cnv.params["condition[T.post-flip]"]
    - fit_cnv.params["condition[T.non-contingent]"]
)

cov = fit_cnv.cov_params()

var = (
    cov.loc["condition[T.post-flip]", "condition[T.post-flip]"]
    + cov.loc["condition[T.non-contingent]", "condition[T.non-contingent]"]
    - 2 * cov.loc["condition[T.post-flip]", "condition[T.non-contingent]"]
)

se = np.sqrt(var)
z = b / se
p = 2 * stats.norm.sf(abs(z))

print(
    f"CNV post-flip vs non-contingent: "
    f"b={b:.3f}, SE={se:.3f}, z={z:.3f}, p={p:.5f}"
)


# Plotting

plot_cnv = (
    df.groupby(
        ["subject", "difficulty_label", "condition"],
        observed=True,
        as_index=False,
    )
    .agg(
        cnv=("cnv_value", "mean"),
    )
)

plot_cnv["difficulty_label"] = pd.Categorical(
    plot_cnv["difficulty_label"],
    categories=["easy", "hard"],
    ordered=True,
)

plot_cnv["condition"] = pd.Categorical(
    plot_cnv["condition"],
    categories=[
        "contingent",
        "non-contingent",
        "post-flip",
    ],
    ordered=True,
)

fig, ax = plt.subplots(
    figsize=(5.5, 4),
)

handles = plot_measure(
    plot_cnv,
    y="cnv",
    ylabel="CNV amplitude (µV)",
    ax=ax,
)

ax.set_title("CNV")

fig.legend(
    handles,
    ["Easy", "Hard"],
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
)

plt.tight_layout()
plt.show()