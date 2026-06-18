# Imports
import glob
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")

# Store single-trial rows
all_trial_rows = []

# CNV ROI
roi_space = ["FCz", "Fz", "Cz", "FC1", "FC2"]
roi_time = (0.7, 1.2)


# ======================================================================================
# Build single-trial CNV dataframe
# ======================================================================================

for dataset in datasets:

    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])

    if subject_id == 7:
        continue

    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(-0.2, 0))
        .crop(tmin=-0.2, tmax=1.8)
    )

    trialinfo = pd.read_csv(
        dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv"
    )

    assert len(trialinfo) == len(eeg_epochs), dataset

    roi_chs = [
        ch for ch in roi_space
        if ch in eeg_epochs.ch_names
    ]

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

    data_uv = eeg_epochs.get_data() * 1e6

    trial_cnv = (
        data_uv[:, roi_idx, :]
        [:, :, roi_time_mask]
        .mean(axis=(1, 2))
    )

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
    ] = "contingent"

    trialinfo.loc[
        (trialinfo["reliability"] == 0)
        & (trialinfo["prev_flipped"] == 0),
        "flip",
    ] = "non-contingent"

    trialinfo.loc[
        (trialinfo["reliability"] == 0)
        & (trialinfo["prev_flipped"] == 1),
        "flip",
    ] = "post-flip"

    trialinfo["cnv_value"] = trial_cnv

    keep = (
        (trialinfo["prev_accuracy"] == 1)
        & (trialinfo["accuracy"] == 1)
        & (trialinfo["flip"].notna())
    )

    all_trial_rows.append(
        trialinfo.loc[keep].copy()
    )


trial_cnv_df = pd.concat(
    all_trial_rows,
    ignore_index=True,
)


# ======================================================================================
# Prepare analysis dataframe
# ======================================================================================

df = trial_cnv_df.copy()

df = df[
    df["rt"].notna()
    & df["cnv_value"].notna()
    & df["difficulty_label"].notna()
    & df["flip"].notna()
].copy()

df["difficulty_label"] = pd.Categorical(
    df["difficulty_label"],
    categories=["easy", "hard"],
    ordered=True,
)

df["flip"] = pd.Categorical(
    df["flip"],
    categories=["contingent", "non-contingent", "post-flip"],
    ordered=True,
)

# Within-subject CNV centering
df["cnv_value_c"] = (
    df["cnv_value"]
    - df.groupby("subject")["cnv_value"].transform("mean")
)

# Between-subject CNV term
df["cnv_value_subject_mean"] = (
    df.groupby("subject")["cnv_value"].transform("mean")
)

df["cnv_value_subject_mean_c"] = (
    df["cnv_value_subject_mean"]
    - df["cnv_value_subject_mean"].mean()
)


# ======================================================================================
# 1. Does flip explain RT beyond CNV?
# ======================================================================================

# Model with cnv and flip condition
m_cnv = smf.mixedlm(
    "rt ~ difficulty_label + flip "
    "+ cnv_value_c + cnv_value_subject_mean_c",
    data=df,
    groups="subject",
).fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

print(m_cnv.summary())

# Model with only cnv
m_cnv_only = smf.mixedlm(
    "rt ~ difficulty_label "
    "+ cnv_value_c + cnv_value_subject_mean_c",
    data=df,
    groups="subject",
).fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

print(m_cnv_only.summary())

# Comparison of model fit
lr_stat = 2 * (m_cnv.llf - m_cnv_only.llf)

df_diff = (
    m_cnv.df_modelwc
    - m_cnv_only.df_modelwc
)

p_value = stats.chi2.sf(
    lr_stat,
    df_diff,
)

print("\nDoes flip explain RT beyond CNV?")
print("----------------------------------")
print(f"LR χ²({df_diff}) = {lr_stat:.2f}")
print(f"p = {p_value:.6f}")
print(f"AIC no-flip : {m_cnv_only.aic:.2f}")
print(f"AIC flip    : {m_cnv.aic:.2f}")
print(f"ΔAIC        : {m_cnv.aic - m_cnv_only.aic:.2f}")


# ======================================================================================
# 2. Does CNV-RT coupling differ across conditions?
# ======================================================================================

m_cnv_int = smf.mixedlm(
    "rt ~ difficulty_label + flip * cnv_value_c "
    "+ cnv_value_subject_mean_c",
    data=df,
    groups="subject",
).fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

print(m_cnv_int.summary())


term_pf = "flip[T.post-flip]:cnv_value_c"
term_nc = "flip[T.non-contingent]:cnv_value_c"

b = (
    m_cnv_int.params[term_pf]
    - m_cnv_int.params[term_nc]
)

cov = m_cnv_int.cov_params()

var = (
    cov.loc[term_pf, term_pf]
    + cov.loc[term_nc, term_nc]
    - 2 * cov.loc[term_pf, term_nc]
)

se = np.sqrt(var)
z = b / se
p = 2 * stats.norm.sf(abs(z))

print(
    "\nCNV-RT slope difference "
    "(post-flip vs non-contingent)"
)
print("----------------------------------")
print(f"b={b:.3f}, SE={se:.3f}, z={z:.3f}, p={p:.5f}")









# ======================================================================================
# Settings
# ======================================================================================

difficulty_palette = {
    "easy": "#1b9e77",
    "hard": "#d81b60",
}

flip_order = [
    "contingent",
    "non-contingent",
    "post-flip",
]

flip_palette = {
    "contingent": "#1b9e77",
    "non-contingent": "#d81b60",
    "post-flip": "#7570b3",
}


# ======================================================================================
# Prepare CNV condition means
# ======================================================================================

plot_cnv = (
    df.groupby(
        ["subject", "difficulty_label", "flip"],
        observed=True,
        as_index=False,
    )
    .agg(
        cnv_value=("cnv_value", "mean"),
    )
)


# ======================================================================================
# Prepare residualized RT and model predictions
# ======================================================================================

resid_model = smf.ols(
    "rt ~ difficulty_label + cnv_value_subject_mean_c",
    data=df,
).fit()

df_resid = df.copy()
df_resid["rt_resid"] = (
    resid_model.resid
    + df["rt"].mean()
)

cnv_range = np.linspace(
    df_resid["cnv_value_c"].quantile(.01),
    df_resid["cnv_value_c"].quantile(.99),
    100,
)

pred_rows = []

for flip in flip_order:

    tmp = pd.DataFrame({
        "cnv_value_c": cnv_range,
        "flip": flip,
        "difficulty_label": "easy",
        "cnv_value_subject_mean_c": 0,
    })

    tmp["pred_rt"] = m_cnv_int.predict(tmp)

    pred_rows.append(tmp)

pred_df = pd.concat(
    pred_rows,
    ignore_index=True,
)

df_resid["cnv_bin"] = (
    df_resid.groupby("subject")["cnv_value_c"]
    .transform(
        lambda x: pd.qcut(
            x,
            q=5,
            labels=False,
            duplicates="drop",
        )
    )
)

bin_df = (
    df_resid.groupby(
        ["subject", "flip", "cnv_bin"],
        observed=True,
        as_index=False,
    )
    .agg(
        cnv_value_c=("cnv_value_c", "mean"),
        rt_resid=("rt_resid", "mean"),
    )
)


# ======================================================================================
# Figure
# ======================================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4),
)


# --------------------------------------------------------------------------------------
# Left panel: CNV amplitude
# --------------------------------------------------------------------------------------

ax = axes[0]

sns.pointplot(
    data=plot_cnv,
    x="flip",
    y="cnv_value",
    hue="difficulty_label",
    errorbar=("ci", 95),
    capsize=.1,
    err_kws={"linewidth": 1.8},
    dodge=0.25,
    markers="o",
    linestyles="-",
    linewidth=2.5,
    markersize=7,
    palette=difficulty_palette,
    ax=ax,
)

sns.stripplot(
    data=plot_cnv,
    x="flip",
    y="cnv_value",
    hue="difficulty_label",
    dodge=True,
    jitter=0.08,
    alpha=0.35,
    size=3,
    palette=difficulty_palette,
    ax=ax,
)

if ax.legend_ is not None:
    ax.legend_.remove()

ax.set_xlabel("")
ax.set_ylabel("CNV value (µV)")
ax.set_title("CNV amplitude", pad=12)

sns.despine(ax=ax)


# --------------------------------------------------------------------------------------
# Right panel: CNV–RT relationship
# --------------------------------------------------------------------------------------

ax = axes[1]

sns.scatterplot(
    data=bin_df,
    x="cnv_value_c",
    y="rt_resid",
    hue="flip",
    hue_order=flip_order,
    palette=flip_palette,
    alpha=0.25,
    s=25,
    edgecolor=None,
    ax=ax,
    legend=False,
)

sns.lineplot(
    data=pred_df,
    x="cnv_value_c",
    y="pred_rt",
    hue="flip",
    hue_order=flip_order,
    palette=flip_palette,
    linewidth=2.8,
    ax=ax,
)

if ax.legend_ is not None:
    ax.legend_.remove()

ax.set_xlabel("Within-subject CNV value (µV)")
ax.set_ylabel("Residualized RT (ms)")
ax.set_title("CNV–RT relationship", pad=12)

sns.despine(ax=ax)


# ======================================================================================
# Legends
# ======================================================================================

difficulty_handles = [
    Line2D(
        [0],
        [0],
        color=difficulty_palette["easy"],
        marker="o",
        linewidth=2.5,
        markersize=8,
    ),
    Line2D(
        [0],
        [0],
        color=difficulty_palette["hard"],
        marker="o",
        linewidth=2.5,
        markersize=8,
    ),
]

fig.legend(
    difficulty_handles,
    ["easy", "hard"],
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.23, 1.08),
)

flip_handles = [
    Line2D(
        [0],
        [0],
        color=flip_palette[flip],
        linewidth=3,
    )
    for flip in flip_order
]

fig.legend(
    flip_handles,
    flip_order,
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.73, 1.08),
)


# ======================================================================================
# Layout
# ======================================================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.98],
)

plt.show()