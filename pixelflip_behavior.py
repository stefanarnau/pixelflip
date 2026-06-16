# Imports
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Bins
all_rows = []

# Define paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")

# Helpers ====================================================================================
def plot_behavior_measure(
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


# Loop datasets ============================================================================
for dataset in datasets:

    # Get id
    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])

    # Skip VP 07 (age outlier)
    if subject_id == 7:
        continue

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv")
    
    trialinfo = trialinfo.copy()
    
    trialinfo["subject"] = subject_id
    
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
    
    # Same inclusion criteria as ERP analyses
    keep = (
        (trialinfo["prev_accuracy"] == 1)
    )
    
    all_rows.append(
        trialinfo.loc[
            keep,
            [
                "subject",
                "difficulty_label",
                "condition",
                "accuracy",
                "rt",
            ]
        ].copy()
    )
        
    
df = pd.concat(
    all_rows,
    ignore_index=True,
)

df["difficulty_label"] = pd.Categorical(
    df["difficulty_label"],
    categories=["easy", "hard"],
    ordered=True,
)

df["condition"] = pd.Categorical(
    df["condition"],
    categories=[
        "contingent",
        "non-contingent",
        "post-flip",
    ],
    ordered=True,
)

# Response time inference =========================================================================

df_rt = df[
    (df["accuracy"] == 1)
    & (df["rt"].notna())
].copy()

fit_rt = smf.mixedlm(
    "rt ~ difficulty_label * condition",
    data=df_rt,
    groups=df_rt["subject"],
).fit(
    method="powell",
    reml=False,
)

print(fit_rt.summary())


b = (
    fit_rt.params["condition[T.post-flip]"]
    - fit_rt.params["condition[T.non-contingent]"]
)

cov = fit_rt.cov_params()

var = (
    cov.loc[
        "condition[T.post-flip]",
        "condition[T.post-flip]"
    ]
    + cov.loc[
        "condition[T.non-contingent]",
        "condition[T.non-contingent]"
    ]
    - 2 * cov.loc[
        "condition[T.post-flip]",
        "condition[T.non-contingent]"
    ]
)

se = np.sqrt(var)

z = b / se
p = 2 * stats.norm.sf(abs(z))

print(
    f"post-flip vs non-contingent: "
    f"b={b:.3f}, SE={se:.3f}, z={z:.3f}, p={p:.5f}"
)


# Accuracy inference =========================================================================

gee_acc = smf.gee(
    "accuracy ~ difficulty_label * condition",
    groups="subject",
    data=df,
    family=sm.families.Binomial(),
)

fit_acc = gee_acc.fit()

print(fit_acc.summary())

b = (
    fit_acc.params["condition[T.post-flip]"]
    - fit_acc.params["condition[T.non-contingent]"]
)

cov = fit_acc.cov_params()

var = (
    cov.loc["condition[T.post-flip]", "condition[T.post-flip]"]
    + cov.loc["condition[T.non-contingent]", "condition[T.non-contingent]"]
    - 2 * cov.loc["condition[T.post-flip]", "condition[T.non-contingent]"]
)

se = np.sqrt(var)
z = b / se
p = 2 * stats.norm.sf(abs(z))

print(
    f"Accuracy post-flip vs non-contingent: "
    f"b={b:.3f}, SE={se:.3f}, z={z:.3f}, p={p:.5f}"
)

# Plotting ====================================================================================

# Subject-level summaries for visualization
plot_df = (
    df.groupby(
        ["subject", "difficulty_label", "condition"],
        observed=True,
        as_index=False,
    )
    .agg(
        rt=("rt", lambda x: x[df.loc[x.index, "accuracy"] == 1].mean()),
        accuracy=("accuracy", "mean"),
    )
)

plot_df["difficulty_label"] = pd.Categorical(
    plot_df["difficulty_label"],
    categories=["easy", "hard"],
    ordered=True,
)

plot_df["condition"] = pd.Categorical(
    plot_df["condition"],
    categories=["contingent", "non-contingent", "post-flip"],
    ordered=True,
)

# Accuracy in %
plot_df["accuracy_percent"] = (
    plot_df["accuracy"] * 100
)

# Create figure
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4),
    sharex=True,
)

# RT panel
handles = plot_behavior_measure(
    plot_df,
    y="rt",
    ylabel="ms",
    ax=axes[0],
)

# Accuracy panel
plot_behavior_measure(
    plot_df,
    y="accuracy_percent",
    ylabel="% correct",
    ax=axes[1],
    ylim=(40, 100),
)

# Panel labels
axes[0].set_title("Response time")
axes[1].set_title("Accuracy")

# Shared legend
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

# descriptives =====================================================================
desc = (
    plot_df.groupby(
        ["difficulty_label", "condition"],
        observed=True,
    )
    .agg(
        rt_mean=("rt", "mean"),
        rt_sd=("rt", "std"),
        acc_mean=("accuracy_percent", "mean"),
        acc_sd=("accuracy_percent", "std"),
    )
)

print(desc)