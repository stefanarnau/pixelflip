# Imports
import os
import pandas as pd
import glob
from scipy.stats import spearmanr
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

# Define paths
path_ratings = "/mnt/data_dump/pixelflip/4_subjective_ratings/"
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")

# Load ratings file
fn = "subjective_ratings_final.csv"
df_ratings = pd.read_csv(os.path.join(path_ratings, fn))

# Rename columns
df_ratings = df_ratings.rename(
    columns={
        "id": "subject",
        "Frage1": "focus_easy",
        "Frage2": "focus_hard",
        "Frage3": "focus_accu",
        "Frage4": "focus_flip",
        "Frage5": "moti_accu",
        "Frage6": "moti_flip",
        "Frage7": "mw_accu",
        "Frage8": "mw_flip",
    }
)

# Loop datasets
subject_rows = []
for dataset in datasets:

    # Get id
    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])

    # Get rid of outlier
    if subject_id == 7:
        continue
    
    # Load EEG
    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(-0.2, 0))
        .crop(tmin=-0.2, tmax=1.8)
    )
    
    # ROI channels
    roi_idx = mne.pick_channels(
        eeg_epochs.ch_names,
        include=["FCz", "Fz", "Cz", "FC1", "FC2"],
    )
    
    # ROI time window
    roi_time_mask = (
        (eeg_epochs.times >= 0.7)
        & (eeg_epochs.times <= 1.2)
    )
    
    # Single-trial CNV (µV)
    data_uv = eeg_epochs.get_data() * 1e6
    
    trial_cnv = (
        data_uv[:, roi_idx, :]
        [:, :, roi_time_mask]
        .mean(axis=(1, 2))
    )

    # Get trialinfo
    trialinfo = pd.read_csv(
        dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv"
    )
    
    # For RT and CNV: correct trials only
    valid_correct = (
        (trialinfo["prev_accuracy"] == 1)
        & (trialinfo["accuracy"] == 1)
    )
    
    # For accuracy: all trials after previous correct trial
    valid_all = (
        trialinfo["prev_accuracy"] == 1
    )
    
    # RT/CNV masks
    accu_mask = valid_correct & (trialinfo["reliability"] == 1)
    flip_mask = valid_correct & (trialinfo["reliability"] == 0)
    
    # Accuracy masks
    accu_acc_mask = valid_all & (trialinfo["reliability"] == 1)
    flip_acc_mask = valid_all & (trialinfo["reliability"] == 0)
        
    # DataFrames for RT/accuracy summaries
    accu = trialinfo[accu_mask]
    flip = trialinfo[flip_mask]
    
    # Get measures
    rt_accu = trialinfo.loc[accu_mask, "rt"].mean()
    rt_flip = trialinfo.loc[flip_mask, "rt"].mean()
    
    cnv_accu = trial_cnv[accu_mask.to_numpy()].mean()
    cnv_flip = trial_cnv[flip_mask.to_numpy()].mean()
    
    acc_accu = trialinfo.loc[accu_acc_mask, "accuracy"].mean()
    acc_flip = trialinfo.loc[flip_acc_mask, "accuracy"].mean()
        
    # Collect
    subject_rows.append({
        "subject": subject_id,
    
        "rt_accu": rt_accu,
        "rt_flip": rt_flip,
        "rt_diff": rt_accu - rt_flip,
    
        "acc_accu": acc_accu,
        "acc_flip": acc_flip,
        "acc_diff": acc_accu - acc_flip,
    
        "cnv_accu": cnv_accu,
        "cnv_flip": cnv_flip,
        "cnv_diff": cnv_accu - cnv_flip,
    })

# Create behavior df
df_behavior = pd.DataFrame(subject_rows)

# State effects
df_behavior["rt_diff"] = (
    df_behavior["rt_accu"]
    - df_behavior["rt_flip"]
)

df_behavior["acc_diff"] = (
    df_behavior["acc_accu"]
    - df_behavior["acc_flip"]
)

df_behavior["cnv_diff"] = (
    df_behavior["cnv_accu"]
    - df_behavior["cnv_flip"]
)

df_ratings["moti_diff"] = (
    df_ratings["moti_accu"]
    - df_ratings["moti_flip"]
)
df_ratings["focus_diff"] = (
    df_ratings["focus_accu"]
    - df_ratings["focus_flip"]
)
df_ratings["mw_diff"] = (
    df_ratings["mw_accu"]
    - df_ratings["mw_flip"]
)

# Merge with ratings
df = df_ratings.merge(
    df_behavior,
    on="subject",
    how="inner",
)

# Define correlation variables
corr_vars = [
    "focus_diff",
    "moti_diff",
    "mw_diff",
    "rt_diff",
    "acc_diff",
    "cnv_diff"
]



corr_mat = pd.DataFrame(
    index=corr_vars,
    columns=corr_vars,
    dtype=float,
)

p_mat = pd.DataFrame(
    index=corr_vars,
    columns=corr_vars,
    dtype=float,
)

for v1 in corr_vars:
    for v2 in corr_vars:

        rho, p = spearmanr(
            df[v1],
            df[v2],
            nan_policy="omit",
        )

        corr_mat.loc[v1, v2] = rho
        p_mat.loc[v1, v2] = p


plt.figure(figsize=(6, 5))

sns.heatmap(
    corr_mat,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
)

plt.title("Spearman correlation matrix")
plt.tight_layout()
plt.show()




# Wilcoxon tests
questionnaires = [
    ("focus", "focus_accu", "focus_flip"),
    ("moti",  "moti_accu",  "moti_flip"),
    ("mw",    "mw_accu",    "mw_flip"),
]

for label, accu_col, flip_col in questionnaires:

    t, p_t = ttest_rel(
        df_ratings[accu_col],
        df_ratings[flip_col],
        nan_policy="omit",
    )

    w, p_w = wilcoxon(
        df_ratings[accu_col],
        df_ratings[flip_col],
    )

    print(f"\n{label}")
    print("-" * 30)
    print(
        f"Accurate: {df_ratings[accu_col].mean():.2f} ± "
        f"{df_ratings[accu_col].std():.2f}"
    )
    print(
        f"Flip:     {df_ratings[flip_col].mean():.2f} ± "
        f"{df_ratings[flip_col].std():.2f}"
    )
    print(f"paired t-test: t={t:.2f}, p={p_t:.4f}")
    print(f"Wilcoxon:      p={p_w:.4f}")
    
    
    
    diff = (
        df_ratings[accu_col]
        - df_ratings[flip_col]
    )
    
    cohens_dz = diff.mean() / diff.std(ddof=1)
    
    print(f"Cohen's dz = {cohens_dz:.2f}")
    




rating_specs = [
    ("task focus", "focus_accu", "focus_flip"),
    ("motivation", "moti_accu", "moti_flip"),
    ("mind wandering", "mw_accu", "mw_flip"),
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(9, 3.2),
    sharey=True,
)

x = np.array([0, 1])
xlabels = ["contingent", "non-contingent"]

cond_colors = ["#1b9e77", "#d81b60"]

for ax, (title, accu_col, flip_col) in zip(axes, rating_specs):

    sub = df_ratings[[accu_col, flip_col]].dropna()

    # individual paired lines
    for _, row in sub.iterrows():
        ax.plot(
            x,
            [row[accu_col], row[flip_col]],
            color="0.75",
            alpha=0.45,
            linewidth=1,
            zorder=1,
        )

    # individual points
    ax.scatter(
        np.repeat(0, len(sub)),
        sub[accu_col],
        color=cond_colors[0],
        alpha=0.35,
        s=18,
        zorder=2,
    )

    ax.scatter(
        np.repeat(1, len(sub)),
        sub[flip_col],
        color=cond_colors[1],
        alpha=0.35,
        s=18,
        zorder=2,
    )

    means = np.array([
        sub[accu_col].mean(),
        sub[flip_col].mean(),
    ])

    sems = np.array([
        sub[accu_col].sem(),
        sub[flip_col].sem(),
    ])

    ci95 = 1.96 * sems

    # group mean
    ax.plot(
        x,
        means,
        color="black",
        linewidth=3,
        zorder=4,
    )

    ax.errorbar(
        x,
        means,
        yerr=ci95,
        fmt="o",
        color="black",
        markersize=7,
        capsize=4,
        linewidth=2,
        zorder=5,
    )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.15, 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("rating (mm)")
axes[0].set_ylim(0, 105)

plt.tight_layout()
plt.show()