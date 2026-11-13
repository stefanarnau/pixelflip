# =============================================================================
# PixelFlip 10: simulation-based sensitivity for CNV–RT coupling
# =============================================================================
# Two targets (two-sided Wald tests, alpha=.05):
#   (1) Sustained CNV slope moderation: non-contingent minus contingent
#   (2) Transient CNV slope moderation: post-flip minus non-contingent
#
# Preserve observed CNV values, participant and condition structure, and
# empirical nuisance effects/variance components. Impose one moderation at a
# time, then fit the SAME full interaction model as in PixelFlip 08.
# The transient effect is tested with a covariance-aware linear contrast.
# Conditional simulation-based power does NOT establish equivalence.
# =============================================================================

import glob
import warnings
import numpy as np
import pandas as pd
import mne
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend

PATH_IN = "/mnt/data_dump/pixelflip/2_cleaned/"
N_SIMULATIONS = 500     # development; use 2000+ for final estimates
N_JOBS = -1
ALPHA = 0.05
TARGET_POWER = 0.80
RANDOM_SEED = 42
MAXITER = 1000
EMPIRICAL_FIT_METHOD = "powell"
SIMULATION_FIT_METHOD = "lbfgs"
SAVE_RESULTS = True
RESULTS_CSV = "pixelflip_cnv_rt_sensitivity_results.csv"
FIGURE_FILE = "pixelflip_cnv_rt_sensitivity_analysis.pdf"
# Adjust after examining the observed coefficients and CNV scale.
INTERACTION_EFFECT_SIZES = np.arange(0, 6.01, 0.25)  # absolute ms/µV; refine as needed

# Load and prepare data identically to PixelFlip 08.
datasets = glob.glob(f"{PATH_IN}/*cue_erp.set")
if not datasets:
    raise FileNotFoundError(f"No cue ERP datasets found under {PATH_IN}")
all_trial_rows = []
roi_space = ["FCz", "Fz", "Cz", "FC1", "FC2"]
roi_time = (0.7, 1.2)

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



# Fit empirical full interaction model, matching script 08.
FULL_FORMULA = (
    "rt ~ difficulty_label + flip * cnv_value_c "
    "+ cnv_value_subject_mean_c"
)
fit_empirical = smf.mixedlm(
    FULL_FORMULA, data=df, groups=df["subject"]
).fit(method=EMPIRICAL_FIT_METHOD, reml=False, maxiter=MAXITER, disp=False)
print("\nEmpirical full model:")
print(fit_empirical.summary())

MAIN_TERM = "cnv_value_c"
MOD_TERM = "flip[T.non-contingent]:cnv_value_c"
PF_TERM = "flip[T.post-flip]:cnv_value_c"
observed_main = float(fit_empirical.params[MAIN_TERM])
observed_sustained = float(fit_empirical.params[MOD_TERM])
observed_transient = float(fit_empirical.params[PF_TERM] - fit_empirical.params[MOD_TERM])
print(f"\nObserved contingent-reference CNV slope: {observed_main:.4f} ms/µV")
print(f"Observed sustained moderation (NC - contingent): {observed_sustained:.4f} ms/µV")
print(f"Observed transient moderation (post-flip - NC): {observed_transient:.4f} ms/µV")

# Retain nuisance parameters and observed design, not empirical CNV effects.
params = fit_empirical.params
intercept = float(params["Intercept"])
difficulty_effect = float(params["difficulty_label[T.hard]"])
noncontingent_effect = float(params["flip[T.non-contingent]"])
postflip_effect = float(params["flip[T.post-flip]"])
between_cnv_effect = float(params["cnv_value_subject_mean_c"])
random_sd = float(np.sqrt(fit_empirical.cov_re.iloc[0, 0]))
residual_sd = float(np.sqrt(fit_empirical.scale))

subjects, subject_idx = np.unique(df["subject"].to_numpy(), return_inverse=True)
is_hard = df["difficulty_label"].astype(str).eq("hard").to_numpy(dtype=float)
is_nc = df["flip"].astype(str).eq("non-contingent").to_numpy(dtype=float)
is_pf = df["flip"].astype(str).eq("post-flip").to_numpy(dtype=float)
cnv = df["cnv_value_c"].to_numpy(dtype=float)
between_cnv = df["cnv_value_subject_mean_c"].to_numpy(dtype=float)
base_mu = (intercept + difficulty_effect * is_hard
           + noncontingent_effect * is_nc + postflip_effect * is_pf
           + between_cnv_effect * between_cnv)
fit_template = df[["subject", "difficulty_label", "flip",
                   "cnv_value_c", "cnv_value_subject_mean_c"]].copy()

def transient_contrast_pvalue(fit):
    """Two-sided Wald test for post-flip minus non-contingent CNV slope."""
    b = float(fit.params[PF_TERM] - fit.params[MOD_TERM])
    cov = fit.cov_params()
    var = float(cov.loc[PF_TERM, PF_TERM]
                + cov.loc[MOD_TERM, MOD_TERM]
                - 2 * cov.loc[PF_TERM, MOD_TERM])
    if not np.isfinite(var) or var <= 0:
        return np.nan
    from scipy.stats import norm
    return float(2 * norm.sf(abs(b / np.sqrt(var))))


def run_one(effect, target, seed):
    rng = np.random.default_rng(seed)
    if target == "sustained":
        # Both unreliable conditions share the imposed slope difference.
        # True post-flip minus NC slope difference = 0.
        mu = base_mu + observed_main * cnv + effect * cnv * (is_nc + is_pf)
    elif target == "transient":
        # No sustained moderation; only post-flip changes slope.
        # True post-flip minus NC slope difference = effect.
        mu = base_mu + observed_main * cnv + effect * cnv * is_pf
    else:
        raise ValueError(target)

    y = (mu + rng.normal(0, random_sd, len(subjects))[subject_idx]
         + rng.normal(0, residual_sd, len(df)))
    tmp = fit_template.copy()
    tmp["rt"] = y
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(FULL_FORMULA, data=tmp, groups=tmp["subject"]).fit(
                method=SIMULATION_FIT_METHOD, reml=False,
                maxiter=MAXITER, disp=False)
        if not fit.converged:
            return (np.nan, False, False)
        if target == "sustained":
            p = float(fit.pvalues[MOD_TERM])
        else:
            p = transient_contrast_pvalue(fit)
        return (p, True, not np.isfinite(p))
    except Exception:
        return (np.nan, False, True)


def power_curve(target, effects, seed):
    rows = []
    seeds_by_effect = np.random.SeedSequence(seed).spawn(len(effects))
    for effect, seed_seq in zip(effects, seeds_by_effect):
        seeds = [int(s.generate_state(1)[0])
                 for s in seed_seq.spawn(N_SIMULATIONS)]
        with parallel_backend("loky", inner_max_num_threads=1):
            out = Parallel(n_jobs=N_JOBS)(
                delayed(run_one)(effect, target, seed) for seed in seeds)
        p = np.asarray([r[0] for r in out], dtype=float)
        valid = np.isfinite(p)
        n_valid = int(valid.sum())
        n_sig = int(np.sum(p[valid] < ALPHA))
        power = n_sig / n_valid if n_valid else np.nan
        rows.append(dict(target=target, effect_ms_per_uv=effect, power=power,
                         n_significant=n_sig, n_successful=n_valid,
                         n_nonconverged=sum(not r[1] for r in out),
                         n_failed=sum(r[2] for r in out)))
        print(f"{target:10s} {effect:6.2f} ms/µV: power={power:.3f} "
              f"valid={n_valid}/{N_SIMULATIONS}")
    return pd.DataFrame(rows)


def threshold_at_power(results):
    a = results.sort_values("effect_ms_per_uv")
    x = a["effect_ms_per_uv"].to_numpy()
    y = a["power"].to_numpy()
    valid = np.isfinite(y)
    x, y = x[valid], y[valid]
    idx = np.flatnonzero(y >= TARGET_POWER)
    if not len(idx):
        return np.nan
    k = idx[0]
    if k == 0:
        return float(x[0])
    return float(np.interp(TARGET_POWER, y[k-1:k+1], x[k-1:k+1])) if y[k] > y[k-1] else float(x[k])


sustained = power_curve("sustained", INTERACTION_EFFECT_SIZES, RANDOM_SEED)
transient = power_curve("transient", INTERACTION_EFFECT_SIZES, RANDOM_SEED + 1)
results = pd.concat([sustained, transient], ignore_index=True)
if SAVE_RESULTS:
    results.to_csv(RESULTS_CSV, index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, result, title, observed in zip(
    axes, [sustained, transient],
    ["Sustained contingency moderation", "Transient post-flip moderation"],
    [observed_sustained, observed_transient],
):
    mde = threshold_at_power(result)
    ax.plot(result["effect_ms_per_uv"], result["power"], "o-",
            linewidth=2, label="Estimated power")
    ax.axhline(TARGET_POWER, color="0.4", linestyle="--", label="80% power")
    if np.isfinite(mde):
        ax.axvline(mde, color="tab:orange", linestyle=":",
                   label=f"80% sensitivity ({mde:.2f})")
    ax.axvline(abs(observed), color="tab:green", linestyle="-.",
               label=f"Observed |effect| ({abs(observed):.2f})")
    ax.set_title(title)
    ax.set_xlabel("Absolute true slope difference (ms/µV)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, INTERACTION_EFFECT_SIZES.max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    print(f"{title}: 80% sensitivity = {mde:.3f} ms/µV")
    zero = result.loc[np.isclose(result["effect_ms_per_uv"], 0), "power"]
    if len(zero):
        print(f"  Null-effect rejection rate: {zero.iloc[0]:.3f} (nominal {ALPHA:.2f})")
axes[0].set_ylabel("Estimated power")
fig.tight_layout()
if SAVE_RESULTS:
    fig.savefig(FIGURE_FILE, bbox_inches="tight")
plt.show()
