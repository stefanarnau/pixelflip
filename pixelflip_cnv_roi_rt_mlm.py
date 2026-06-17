# Imports
import glob
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


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

    # Larger value = stronger CNV negativity
    trial_cnv_strength = -trial_cnv

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
    trialinfo["cnv_strength"] = trial_cnv_strength

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
    & df["cnv_strength"].notna()
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
df["cnv_strength_c"] = (
    df["cnv_strength"]
    - df.groupby("subject")["cnv_strength"].transform("mean")
)

# Between-subject CNV term
df["cnv_strength_subject_mean"] = (
    df.groupby("subject")["cnv_strength"].transform("mean")
)

df["cnv_strength_subject_mean_c"] = (
    df["cnv_strength_subject_mean"]
    - df["cnv_strength_subject_mean"].mean()
)


# ======================================================================================
# 1. Does flip explain RT beyond CNV?
# ======================================================================================

m_cnv = smf.mixedlm(
    "rt ~ difficulty_label + flip "
    "+ cnv_strength_c + cnv_strength_subject_mean_c",
    data=df,
    groups="subject",
).fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

print(m_cnv.summary())


m_cnv_only = smf.mixedlm(
    "rt ~ difficulty_label "
    "+ cnv_strength_c + cnv_strength_subject_mean_c",
    data=df,
    groups="subject",
).fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

print(m_cnv_only.summary())


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
    "rt ~ difficulty_label + flip * cnv_strength_c "
    "+ cnv_strength_subject_mean_c",
    data=df,
    groups="subject",
).fit(
    method="powell",
    reml=False,
    maxiter=1000,
)

print(m_cnv_int.summary())


term_pf = "flip[T.post-flip]:cnv_strength_c"
term_nc = "flip[T.non-contingent]:cnv_strength_c"

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







import pandas as pd
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt

rows = []

for (subject, flip), subdf in df.groupby(
    ["subject", "flip"],
    observed=True,
):

    if len(subdf) < 20:
        continue

    slope, intercept, r, p, se = linregress(
        subdf["cnv_strength_c"],
        subdf["rt"],
    )

    rows.append({
        "subject": subject,
        "flip": flip,
        "slope": slope,
    })

slope_df = pd.DataFrame(rows)

plt.figure(figsize=(6,4))

sns.boxplot(
    data=slope_df,
    x="flip",
    y="slope",
    showfliers=False,
)

sns.stripplot(
    data=slope_df,
    x="flip",
    y="slope",
    color="black",
    alpha=.5,
)

plt.axhline(0, color="black", ls="--")
plt.ylabel("CNV–RT slope")
plt.xlabel("")
plt.tight_layout()
plt.show()







df_plot = df.copy()

df_plot["cnv_bin"] = (
    df_plot.groupby("subject")["cnv_strength_c"]
    .transform(
        lambda x: pd.qcut(
            x,
            q=5,
            labels=False,
            duplicates="drop",
        )
    )
)

summary = (
    df_plot.groupby(
        ["flip", "cnv_bin"],
        observed=True,
    )
    .agg(
        rt=("rt", "mean"),
    )
    .reset_index()
)

plt.figure(figsize=(6,4))

sns.lineplot(
    data=summary,
    x="cnv_bin",
    y="rt",
    hue="flip",
    marker="o",
)

plt.xlabel("CNV strength quintile")
plt.ylabel("RT (ms)")
plt.tight_layout()
plt.show()




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# CNV range
cnv_range = np.linspace(
    df["cnv_strength_c"].quantile(.01),
    df["cnv_strength_c"].quantile(.99),
    100,
)

# Prediction dataframe
pred_df = pd.DataFrame()

for flip in [
    "contingent",
    "non-contingent",
    "post-flip",
]:

    tmp = pd.DataFrame({
        "cnv_strength_c": cnv_range,
        "flip": flip,
        "difficulty_label": "easy",   # hold constant
        "cnv_strength_subject_mean_c": 0,
    })

    tmp["pred_rt"] = m_cnv.predict(tmp)
    pred_df = pd.concat([pred_df, tmp])

# Plot
plt.figure(figsize=(7,5))

sns.lineplot(
    data=pred_df,
    x="cnv_strength_c",
    y="pred_rt",
    hue="flip",
    linewidth=3,
)

plt.xlabel("Within-subject CNV strength (µV)")
plt.ylabel("Predicted RT (ms)")
plt.title("Predicted RT from mixed model")

plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

palette = {
    "contingent": "#1b9e77",
    "non-contingent": "#d81b60",
    "post-flip": "#7570b3",
}

flip_order = ["contingent", "non-contingent", "post-flip"]


# ======================================================================================
# 1. Main RT condition plot
# ======================================================================================

plot_rt = (
    df.groupby(
        ["subject", "difficulty_label", "flip"],
        observed=True,
        as_index=False,
    )
    .agg(
        rt=("rt", "mean"),
    )
)

fig, ax = plt.subplots(figsize=(5.5, 4))

sns.pointplot(
    data=plot_rt,
    x="flip",
    y="rt",
    hue="difficulty_label",
    errorbar=("ci", 95),
    capsize=.1,
    err_kws={"linewidth": 1.8},
    dodge=0.25,
    markers="o",
    linestyles="-",
    linewidth=2.5,
    markersize=7,
    palette={
        "easy": "#1b9e77",
        "hard": "#d81b60",
    },
    ax=ax,
)

sns.stripplot(
    data=plot_rt,
    x="flip",
    y="rt",
    hue="difficulty_label",
    dodge=True,
    jitter=0.08,
    alpha=0.35,
    size=3,
    palette={
        "easy": "#1b9e77",
        "hard": "#d81b60",
    },
    ax=ax,
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["easy", "hard"], frameon=False, title=None)

ax.set_xlabel("")
ax.set_ylabel("Response time (ms)")
ax.set_title("Observed response time")
sns.despine(ax=ax)
plt.tight_layout()
plt.show()


# ======================================================================================
# 2. CNV condition plot
# ======================================================================================

plot_cnv = (
    df.groupby(
        ["subject", "difficulty_label", "flip"],
        observed=True,
        as_index=False,
    )
    .agg(
        cnv_strength=("cnv_strength", "mean"),
    )
)

fig, ax = plt.subplots(figsize=(5.5, 4))

sns.pointplot(
    data=plot_cnv,
    x="flip",
    y="cnv_strength",
    hue="difficulty_label",
    errorbar=("ci", 95),
    capsize=.1,
    err_kws={"linewidth": 1.8},
    dodge=0.25,
    markers="o",
    linestyles="-",
    linewidth=2.5,
    markersize=7,
    palette={
        "easy": "#1b9e77",
        "hard": "#d81b60",
    },
    ax=ax,
)

sns.stripplot(
    data=plot_cnv,
    x="flip",
    y="cnv_strength",
    hue="difficulty_label",
    dodge=True,
    jitter=0.08,
    alpha=0.35,
    size=3,
    palette={
        "easy": "#1b9e77",
        "hard": "#d81b60",
    },
    ax=ax,
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["easy", "hard"], frameon=False, title=None)

ax.set_xlabel("")
ax.set_ylabel("CNV strength (µV)")
ax.set_title("CNV amplitude")
sns.despine(ax=ax)
plt.tight_layout()
plt.show()


# ======================================================================================
# 3. Residualized RT vs CNV plot
# ======================================================================================
# Residualize RT for difficulty and subject mean CNV, but keep flip effects visible.
# This isolates the within-subject CNV relationship while preserving condition offsets.

resid_model = smf.ols(
    "rt ~ difficulty_label + cnv_strength_subject_mean_c",
    data=df,
).fit()

df_resid = df.copy()
df_resid["rt_resid"] = resid_model.resid + df["rt"].mean()

# Model-predicted lines from interaction MLM
cnv_range = np.linspace(
    df_resid["cnv_strength_c"].quantile(.01),
    df_resid["cnv_strength_c"].quantile(.99),
    100,
)

pred_rows = []

for flip in flip_order:
    tmp = pd.DataFrame({
        "cnv_strength_c": cnv_range,
        "flip": flip,
        "difficulty_label": "easy",
        "cnv_strength_subject_mean_c": 0,
    })

    tmp["pred_rt"] = m_cnv_int.predict(tmp)
    pred_rows.append(tmp)

pred_df = pd.concat(pred_rows, ignore_index=True)

# Optional: subject-level binning for readable points
df_resid["cnv_bin"] = (
    df_resid.groupby("subject")["cnv_strength_c"]
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
        cnv_strength_c=("cnv_strength_c", "mean"),
        rt_resid=("rt_resid", "mean"),
    )
)

fig, ax = plt.subplots(figsize=(6, 4))

sns.scatterplot(
    data=bin_df,
    x="cnv_strength_c",
    y="rt_resid",
    hue="flip",
    hue_order=flip_order,
    palette=palette,
    alpha=0.35,
    s=25,
    edgecolor=None,
    ax=ax,
    legend=False,
)

sns.lineplot(
    data=pred_df,
    x="cnv_strength_c",
    y="pred_rt",
    hue="flip",
    hue_order=flip_order,
    palette=palette,
    linewidth=2.8,
    ax=ax,
)

ax.set_xlabel("Within-subject CNV strength (µV)")
ax.set_ylabel("Residualized RT (ms)")
ax.set_title("CNV–RT relationship")

sns.despine(ax=ax)
plt.tight_layout()
plt.show()