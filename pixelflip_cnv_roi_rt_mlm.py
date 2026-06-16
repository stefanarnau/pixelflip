# Imports
import mne
import glob
import pandas as pd
import numpy as np
import mne.stats 
import mne
import statsmodels.formula.api as smf
from scipy import stats


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

    trialinfo["flip"] = np.nan

    trialinfo.loc[
        trialinfo["reliability"] == 1,
        "flip",
    ] = "Stable"

    trialinfo.loc[
        (trialinfo["reliability"] == 0)
        & (trialinfo["prev_flipped"] == 0),
        "flip",
    ] = "Volatile"

    trialinfo.loc[
        (trialinfo["reliability"] == 0)
        & (trialinfo["prev_flipped"] == 1),
        "flip",
    ] = "Post-Flip"

    trialinfo["condition"] = (
        trialinfo["difficulty_label"]
        + "_"
        + trialinfo["flip"].map({
            "Stable": "00",
            "Volatile": "10",
            "Post-Flip": "11",
        })
    )

    trialinfo["cnv_value"] = trial_cnv
    trialinfo["cnv_strength"] = trial_cnv_strength

    # Keep same inclusion criteria as your trial-level RT analysis
    keep = (
        (trialinfo["prev_accuracy"] == 1)
        & (trialinfo["accuracy"] == 1)
        & (trialinfo["flip"].notna())
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

trial_cnv_df["flip"] = pd.Categorical(
    trial_cnv_df["flip"],
    categories=["Stable", "Volatile", "Post-Flip"],
    ordered=True,
)







df = trial_cnv_df.copy()

# Keep valid RT trials
df = df[
    df["rt"].notna()
    & df["cnv_strength"].notna()
    & df["difficulty_label"].notna()
    & df["flip"].notna()
].copy()

# Optional RT transform
df["log_rt"] = np.log(df["rt"])

# Coding
df["difficulty_label"] = pd.Categorical(
    df["difficulty_label"],
    categories=["easy", "hard"],
    ordered=True,
)

df["flip"] = pd.Categorical(
    df["flip"],
    categories=["Stable", "Volatile", "Post-Flip"],
    ordered=True,
)

# Within-subject center CNV
df["cnv_strength_c"] = (
    df["cnv_strength"]
    - df.groupby("subject")["cnv_strength"].transform("mean")
)

# Subject mean CNV, optional between-subject term
df["cnv_strength_subject_mean"] = (
    df.groupby("subject")["cnv_strength"].transform("mean")
)

df["cnv_strength_subject_mean_c"] = (
    df["cnv_strength_subject_mean"]
    - df["cnv_strength_subject_mean"].mean()
)







model = smf.mixedlm(
    "rt ~ difficulty_label + flip + cnv_strength_c + cnv_strength_subject_mean_c",
    data=df,
    groups="subject",
)

fit = model.fit(
    method="powell",
    reml=False,
)

print(fit.summary())



# Base model: task predictors only
m0 = smf.mixedlm(
    "rt ~ difficulty_label + flip",
    data=df,
    groups="subject",
)

m0_fit = m0.fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

# Neural model: task predictors + CNV
m1 = smf.mixedlm(
    "rt ~ difficulty_label + flip "
    "+ cnv_strength_c + cnv_strength_subject_mean_c",
    data=df,
    groups="subject",
)

m1_fit = m1.fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

print(m0_fit.summary())
print(m1_fit.summary())

# Model comparison
lr_stat = 2 * (m1_fit.llf - m0_fit.llf)
df_diff = m1_fit.df_modelwc - m0_fit.df_modelwc
p_value = stats.chi2.sf(lr_stat, df_diff)

print("\nModel comparison")
print("----------------")
print(f"M0 AIC: {m0_fit.aic:.2f}")
print(f"M1 AIC: {m1_fit.aic:.2f}")
print(f"ΔAIC:  {m1_fit.aic - m0_fit.aic:.2f}")

print(f"M0 BIC: {m0_fit.bic:.2f}")
print(f"M1 BIC: {m1_fit.bic:.2f}")
print(f"ΔBIC:  {m1_fit.bic - m0_fit.bic:.2f}")

print(f"LR χ²({df_diff}) = {lr_stat:.2f}, p = {p_value:.4g}")