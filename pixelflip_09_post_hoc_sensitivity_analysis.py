# =============================================================================
# PixelFlip
# Simulation-based sensitivity analysis for behavioral RT effects
#
# Uses the actual trial structure and variance components of the manuscript
# MixedLM to estimate sensitivity/power for:
#
#   1. Sustained contingency effect:
#      non-contingent vs contingent
#
#   2. Transient post-flip effect:
#      post-flip vs non-contingent
#
# Empirical manuscript model:
#     rt ~ difficulty_label * condition
#
# Simulation/sensitivity model:
#     rt ~ difficulty_label + condition
#
# The additive model is used for the sensitivity analysis because the target
# is the average condition effect across difficulty levels. The simulation
# therefore does not spend power estimating difficulty × condition
# interactions that are not part of the sensitivity question.
# =============================================================================


# Imports ======================================================================

import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from scipy import stats

from joblib import Parallel, delayed, parallel_backend


# Settings =====================================================================

PATH_IN = "/mnt/data_dump/pixelflip/2_cleaned/"

# -----------------------------------------------------------------------------
# Effect-size grid
# -----------------------------------------------------------------------------

EFFECT_MIN_MS = 0
EFFECT_MAX_MS = 25
N_EFFECT_SIZES = 26   # 1-ms steps: 0, 1, 2, ..., 25

EFFECT_SIZES_MS = np.linspace(
    EFFECT_MIN_MS,
    EFFECT_MAX_MS,
    N_EFFECT_SIZES,
)

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

# Development:
#   100-500 simulations per effect size
#
# Final:
#   2000+ simulations per effect size,
#   possibly after narrowing the effect-size range.

N_SIMULATIONS = 500

ALPHA = 0.05
TARGET_POWER = 0.80

RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Model-fitting settings
# -----------------------------------------------------------------------------

# Keep the empirical manuscript model exactly as originally fitted.
EMPIRICAL_FIT_METHOD = "powell"

# Faster optimizer for repeated simulation fits.
SIMULATION_FIT_METHOD = "lbfgs"

MAXITER = 1000


# -----------------------------------------------------------------------------
# Parallelization
# -----------------------------------------------------------------------------

# -1 = use all available CPU cores.
#
# If this makes the machine too busy, use e.g.
# N_JOBS = 8

N_JOBS = -1


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

SAVE_RESULTS = True

RESULTS_CSV = "pixelflip_rt_sensitivity_results.csv"
FIGURE_FILE = "pixelflip_rt_sensitivity_analysis.pdf"


# =============================================================================
# Load behavioral data
# =============================================================================

all_rows = []

datasets = glob.glob(
    f"{PATH_IN}/*cue_erp.set"
)

for dataset in datasets:

    subject_id = int(
        dataset.split("/")[-1]
        .split("_")[0]
        .split("VP")[1]
    )

    # Manuscript exclusion
    if subject_id == 7:
        continue

    # Load corresponding trial information
    trialinfo = pd.read_csv(
        dataset.split("_cleaned_")[0]
        + "_erp_trialinfo.csv"
    )

    trialinfo = trialinfo.copy()

    trialinfo["subject"] = subject_id

    trialinfo["difficulty_label"] = np.where(
        trialinfo["difficulty"] == 0,
        "easy",
        "hard",
    )

    # -------------------------------------------------------------------------
    # Condition coding
    # -------------------------------------------------------------------------

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

    # Same inclusion criterion as ERP analyses
    keep = (
        trialinfo["prev_accuracy"] == 1
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
            ],
        ].copy()
    )


df = pd.concat(
    all_rows,
    ignore_index=True,
)

df["difficulty_label"] = pd.Categorical(
    df["difficulty_label"],
    categories=[
        "easy",
        "hard",
    ],
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


# Correct trials with valid RT
df_rt = df[
    (df["accuracy"] == 1)
    & (df["rt"].notna())
    & (df["condition"].notna())
].copy()


print("\nAnalysis dataset")
print("----------------")
print(
    f"Participants: "
    f"{df_rt['subject'].nunique()}"
)
print(
    f"Trials:       "
    f"{len(df_rt)}"
)

print("\nTrials by condition")
print("-------------------")
print(
    df_rt["condition"]
    .value_counts(sort=False)
)


# =============================================================================
# Fit empirical manuscript model
# =============================================================================

print("\nFitting empirical manuscript RT model...")
print("----------------------------------------")

fit_empirical = smf.mixedlm(
    "rt ~ difficulty_label * condition",
    data=df_rt,
    groups=df_rt["subject"],
).fit(
    method=EMPIRICAL_FIT_METHOD,
    reml=False,
    maxiter=MAXITER,
    disp=False,
)

print(
    fit_empirical.summary()
)


# =============================================================================
# Observed RT effects
# =============================================================================

OBSERVED_SUSTAINED_MS = float(
    fit_empirical.params[
        "condition[T.non-contingent]"
    ]
)

OBSERVED_TRANSIENT_MS = float(
    fit_empirical.params[
        "condition[T.post-flip]"
    ]
    - fit_empirical.params[
        "condition[T.non-contingent]"
    ]
)

print("\nObserved RT effects")
print("-------------------")
print(
    f"Sustained: {OBSERVED_SUSTAINED_MS:.3f} ms"
)
print(
    f"Transient: {OBSERVED_TRANSIENT_MS:.3f} ms"
)

# =============================================================================
# Extract empirical simulation parameters
# =============================================================================

# Condition effects are NOT taken from the empirical model.
# They will be imposed explicitly by EFFECT_SIZES_MS.
#
# We retain only:
#
#   - intercept
#   - difficulty effect
#   - participant random-intercept variance
#   - trial-level residual variance


INTERCEPT = float(
    fit_empirical.params[
        "Intercept"
    ]
)

DIFFICULTY_EFFECT = float(
    fit_empirical.params[
        "difficulty_label[T.hard]"
    ]
)

RANDOM_INTERCEPT_VAR = float(
    fit_empirical.cov_re.iloc[
        0,
        0,
    ]
)

RESIDUAL_VAR = float(
    fit_empirical.scale
)

RANDOM_INTERCEPT_SD = np.sqrt(
    RANDOM_INTERCEPT_VAR
)

RESIDUAL_SD = np.sqrt(
    RESIDUAL_VAR
)


print("\nSimulation parameters")
print("---------------------")

print(
    f"Intercept:                 "
    f"{INTERCEPT:.3f} ms"
)

print(
    f"Difficulty effect:         "
    f"{DIFFICULTY_EFFECT:.3f} ms"
)

print(
    f"Random-intercept variance: "
    f"{RANDOM_INTERCEPT_VAR:.3f}"
)

print(
    f"Random-intercept SD:       "
    f"{RANDOM_INTERCEPT_SD:.3f} ms"
)

print(
    f"Residual variance:         "
    f"{RESIDUAL_VAR:.3f}"
)

print(
    f"Residual SD:               "
    f"{RESIDUAL_SD:.3f} ms"
)


# =============================================================================
# Prepare fixed design arrays
# =============================================================================

subjects = np.sort(
    df_rt["subject"].unique()
)

subject_to_idx = {
    subject: idx
    for idx, subject
    in enumerate(subjects)
}

subject_idx = (
    df_rt["subject"]
    .map(subject_to_idx)
    .to_numpy()
)


is_hard = (
    df_rt["difficulty_label"]
    .astype(str)
    .eq("hard")
    .to_numpy()
)

is_noncontingent = (
    df_rt["condition"]
    .astype(str)
    .eq("non-contingent")
    .to_numpy()
)

is_postflip = (
    df_rt["condition"]
    .astype(str)
    .eq("post-flip")
    .to_numpy()
)


# Template reused for all simulated fits
SIM_TEMPLATE = df_rt[
    [
        "subject",
        "difficulty_label",
        "condition",
    ]
].copy()


# =============================================================================
# Simulation helpers
# =============================================================================

def simulate_rt(
    rng,
    effect_ms,
    effect_type,
):
    """
    Generate one simulated RT dataset using the actual observed trial design.

    Parameters
    ----------
    rng : numpy.random.Generator

    effect_ms : float
        True condition effect imposed in milliseconds.

    effect_type : {"sustained", "transient"}

        sustained:
            contingent       = 0
            non-contingent   = effect_ms
            post-flip        = effect_ms

        transient:
            contingent       = 0
            non-contingent   = 0
            post-flip        = effect_ms

    Difficulty × condition interactions are zero in the data-generating model.
    """

    # -------------------------------------------------------------------------
    # Fixed effects
    # -------------------------------------------------------------------------

    mu = (
        INTERCEPT
        + DIFFICULTY_EFFECT
        * is_hard
    )

    if effect_type == "sustained":

        # Reduced-contingency state affects both ordinary
        # non-contingent and post-flip trials equally.
        mu = (
            mu
            + effect_ms
            * (
                is_noncontingent
                | is_postflip
            )
        )

    elif effect_type == "transient":

        # No sustained contingency effect.
        # Only the additional post-flip cost is imposed.
        mu = (
            mu
            + effect_ms
            * is_postflip
        )

    else:

        raise ValueError(
            "effect_type must be "
            "'sustained' or 'transient'"
        )


    # -------------------------------------------------------------------------
    # Participant random intercept
    # -------------------------------------------------------------------------

    random_intercepts = rng.normal(
        loc=0,
        scale=RANDOM_INTERCEPT_SD,
        size=len(subjects),
    )


    # -------------------------------------------------------------------------
    # Trial-level residual noise
    # -------------------------------------------------------------------------

    residuals = rng.normal(
        loc=0,
        scale=RESIDUAL_SD,
        size=len(df_rt),
    )


    simulated_rt = (
        mu
        + random_intercepts[
            subject_idx
        ]
        + residuals
    )

    return simulated_rt


# =============================================================================
# Fit one simulated dataset
# =============================================================================

def fit_simulated_model(
    simulated_rt,
):
    """
    Fit the additive sensitivity-analysis MixedLM.

    This deliberately differs from the full empirical manuscript model.

    Sensitivity target:
        average condition effects across difficulty levels.
    """

    sim_df = SIM_TEMPLATE.copy()

    sim_df["rt"] = simulated_rt

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        fit = smf.mixedlm(
            "rt ~ difficulty_label + condition",
            data=sim_df,
            groups=sim_df["subject"],
        ).fit(
            method=SIMULATION_FIT_METHOD,
            reml=False,
            maxiter=MAXITER,
            disp=False,
        )

    return fit


# =============================================================================
# Extract p-values
# =============================================================================

def get_sustained_pvalue(
    fit,
):
    """
    Test:
        non-contingent vs contingent.
    """

    return float(
        fit.pvalues[
            "condition[T.non-contingent]"
        ]
    )


def get_transient_pvalue(
    fit,
):
    """
    Test:
        post-flip vs non-contingent.

    Uses the same linear contrast logic as the manuscript analysis.
    """

    term_pf = (
        "condition[T.post-flip]"
    )

    term_nc = (
        "condition[T.non-contingent]"
    )

    b = (
        fit.params[term_pf]
        - fit.params[term_nc]
    )

    cov = fit.cov_params()

    var = (
        cov.loc[
            term_pf,
            term_pf,
        ]
        + cov.loc[
            term_nc,
            term_nc,
        ]
        - 2
        * cov.loc[
            term_pf,
            term_nc,
        ]
    )

    if (
        not np.isfinite(var)
        or var <= 0
    ):
        return np.nan

    se = np.sqrt(
        var
    )

    z = (
        b
        / se
    )

    p = (
        2
        * stats.norm.sf(
            np.abs(z)
        )
    )

    return float(p)


# =============================================================================
# Run one simulation
# =============================================================================

def run_one_simulation(
    effect_ms,
    effect_type,
    random_seed,
):
    """
    Complete one independent simulation.

    Returns
    -------
    dict
        p-value, convergence status, and error status.
    """

    rng = np.random.default_rng(
        random_seed
    )

    simulated_rt = simulate_rt(
        rng=rng,
        effect_ms=effect_ms,
        effect_type=effect_type,
    )

    try:

        fit = fit_simulated_model(
            simulated_rt
        )

        converged = bool(
            fit.converged
        )

        if not converged:

            return {
                "p": np.nan,
                "converged": False,
                "failed": False,
            }

        if effect_type == "sustained":

            p = get_sustained_pvalue(
                fit
            )

        else:

            p = get_transient_pvalue(
                fit
            )

        if not np.isfinite(p):

            return {
                "p": np.nan,
                "converged": True,
                "failed": True,
            }

        return {
            "p": p,
            "converged": True,
            "failed": False,
        }


    except Exception:

        return {
            "p": np.nan,
            "converged": False,
            "failed": True,
        }


# =============================================================================
# Estimate one sensitivity curve
# =============================================================================

def run_power_curve(
    effect_type,
    effect_sizes,
    n_simulations,
    seed,
):
    """
    Estimate power across the requested true-effect grid.

    Simulations belonging to each effect size are run in parallel.
    """

    rows = []

    print(
        f"\nSensitivity analysis: "
        f"{effect_type}"
    )

    print(
        "=" * 60
    )


    # Separate deterministic seed stream for every effect size
    master_seed = np.random.SeedSequence(
        seed
    )

    effect_seed_sequences = (
        master_seed.spawn(
            len(effect_sizes)
        )
    )


    for effect_idx, effect_ms in enumerate(
        effect_sizes
    ):

        # Independent deterministic seeds
        # for every simulated dataset
        simulation_seed_sequences = (
            effect_seed_sequences[
                effect_idx
            ].spawn(
                n_simulations
            )
        )

        simulation_seeds = [
            int(
                ss.generate_state(
                    1
                )[0]
            )
            for ss
            in simulation_seed_sequences
        ]


        # ---------------------------------------------------------------------
        # Parallel simulation
        # ---------------------------------------------------------------------

        # inner_max_num_threads=1 prevents each worker process from
        # spawning additional BLAS threads, which otherwise can make
        # parallel MixedLM fitting slower rather than faster.

        with parallel_backend(
            "loky",
            inner_max_num_threads=1,
        ):

            sim_results = Parallel(
                n_jobs=N_JOBS,
            )(
                delayed(
                    run_one_simulation
                )(
                    effect_ms=effect_ms,
                    effect_type=effect_type,
                    random_seed=sim_seed,
                )
                for sim_seed
                in simulation_seeds
            )


        # ---------------------------------------------------------------------
        # Collect results
        # ---------------------------------------------------------------------

        p_values = np.array(
            [
                result["p"]
                for result
                in sim_results
            ],
            dtype=float,
        )

        converged = np.array(
            [
                result[
                    "converged"
                ]
                for result
                in sim_results
            ],
            dtype=bool,
        )

        failed = np.array(
            [
                result[
                    "failed"
                ]
                for result
                in sim_results
            ],
            dtype=bool,
        )


        valid = np.isfinite(
            p_values
        )

        n_successful = int(
            valid.sum()
        )

        n_nonconverged = int(
            (~converged).sum()
        )

        n_failed = int(
            failed.sum()
        )

        if n_successful > 0:

            n_significant = int(
                (
                    p_values[
                        valid
                    ]
                    < ALPHA
                ).sum()
            )

            power = (
                n_significant
                / n_successful
            )

        else:

            n_significant = 0
            power = np.nan


        rows.append(
            {
                "effect_type": effect_type,
                "effect_ms": effect_ms,
                "power": power,
                "n_significant": n_significant,
                "n_successful": n_successful,
                "n_nonconverged": n_nonconverged,
                "n_failed": n_failed,
            }
        )


        print(
            f"{effect_ms:6.2f} ms | "
            f"power = {power:6.3f} | "
            f"valid = {n_successful:4d}/{n_simulations} | "
            f"non-converged = {n_nonconverged:3d} | "
            f"failed = {n_failed:3d}"
        )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# Interpolate effect at target power
# =============================================================================

def interpolate_effect_at_power(
    results,
    target_power=0.80,
):
    """
    Linearly interpolate the approximate effect size where the estimated
    power curve first crosses the requested target.
    """

    results = (
        results
        .sort_values(
            "effect_ms"
        )
        .reset_index(
            drop=True
        )
    )

    x = (
        results[
            "effect_ms"
        ]
        .to_numpy()
    )

    y = (
        results[
            "power"
        ]
        .to_numpy()
    )


    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]


    if len(x) < 2:

        return np.nan


    above = np.where(
        y >= target_power
    )[0]


    if len(above) == 0:

        return np.nan


    idx_hi = above[0]


    if idx_hi == 0:

        return x[0]


    idx_lo = idx_hi - 1


    x0 = x[idx_lo]
    x1 = x[idx_hi]

    y0 = y[idx_lo]
    y1 = y[idx_hi]


    if y1 == y0:

        return x1


    x_target = (
        x0
        + (
            target_power
            - y0
        )
        * (
            x1
            - x0
        )
        / (
            y1
            - y0
        )
    )

    return float(
        x_target
    )


# =============================================================================
# Run sensitivity analyses
# =============================================================================

results_sustained = run_power_curve(
    effect_type="sustained",
    effect_sizes=EFFECT_SIZES_MS,
    n_simulations=N_SIMULATIONS,
    seed=RANDOM_SEED,
)


results_transient = run_power_curve(
    effect_type="transient",
    effect_sizes=EFFECT_SIZES_MS,
    n_simulations=N_SIMULATIONS,
    seed=RANDOM_SEED + 1,
)


results = pd.concat(
    [
        results_sustained,
        results_transient,
    ],
    ignore_index=True,
)


# =============================================================================
# Estimate target-power threshold
# =============================================================================

mde_sustained = (
    interpolate_effect_at_power(
        results_sustained,
        TARGET_POWER,
    )
)

mde_transient = (
    interpolate_effect_at_power(
        results_transient,
        TARGET_POWER,
    )
)


print("\nEstimated sensitivity")
print("---------------------")

print(
    f"Sustained effect at "
    f"{TARGET_POWER:.0%} power: "
    f"{mde_sustained:.2f} ms"
)

print(
    f"Transient effect at "
    f"{TARGET_POWER:.0%} power: "
    f"{mde_transient:.2f} ms"
)


# =============================================================================
# Type-I error check
# =============================================================================

print("\nZero-effect check")
print("-----------------")

for effect_type, tmp in [
    (
        "sustained",
        results_sustained,
    ),
    (
        "transient",
        results_transient,
    ),
]:

    zero_row = tmp.loc[
        np.isclose(
            tmp[
                "effect_ms"
            ],
            0,
        )
    ]

    if len(zero_row) == 1:

        zero_power = float(
            zero_row[
                "power"
            ].iloc[0]
        )

        print(
            f"{effect_type:10s}: "
            f"{zero_power:.3f} "
            f"(expected ≈ {ALPHA:.2f})"
        )


# =============================================================================
# Save results
# =============================================================================

if SAVE_RESULTS:

    results.to_csv(
        RESULTS_CSV,
        index=False,
    )

    print(
        f"\nSaved results to "
        f"{RESULTS_CSV}"
    )


# =============================================================================
# Plot sensitivity curves
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4),
    sharey=True,
)

# -----------------------------------------------------------------------------
# Left panel: sustained effect
# -----------------------------------------------------------------------------

ax = axes[0]

ax.plot(
    results_sustained["effect_ms"],
    results_sustained["power"],
    marker="o",
    linewidth=2,
    label="Estimated power",
)

ax.axhline(
    TARGET_POWER,
    linestyle="--",
    linewidth=1.5,
    color="0.4",
    label="80% power",
)

if np.isfinite(mde_sustained):
    ax.axvline(
        mde_sustained,
        linestyle=":",
        linewidth=1.8,
        color="tab:orange",
        label=(
            f"80% sensitivity "
            f"({mde_sustained:.1f} ms)"
        ),
    )

ax.axvline(
    OBSERVED_SUSTAINED_MS,
    linestyle="-.",
    linewidth=1.8,
    color="tab:green",
    label=(
        f"Observed effect "
        f"({OBSERVED_SUSTAINED_MS:.1f} ms)"
    ),
)

ax.set_title(
    "Sustained contingency effect"
)

ax.set_xlabel(
    "True RT effect (ms)"
)

ax.set_ylabel(
    "Estimated power"
)

ax.set_xlim(
    0,
    25,
)

ax.set_ylim(
    0,
    1.02,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# -----------------------------------------------------------------------------
# Right panel: transient effect
# -----------------------------------------------------------------------------

ax = axes[1]

ax.plot(
    results_transient["effect_ms"],
    results_transient["power"],
    marker="o",
    linewidth=2,
    label="Estimated power",
)

ax.axhline(
    TARGET_POWER,
    linestyle="--",
    linewidth=1.5,
    color="0.4",
    label="80% power",
)

if np.isfinite(mde_transient):
    ax.axvline(
        mde_transient,
        linestyle=":",
        linewidth=1.8,
        color="tab:orange",
        label=(
            f"80% sensitivity "
            f"({mde_transient:.1f} ms)"
        ),
    )

ax.axvline(
    OBSERVED_TRANSIENT_MS,
    linestyle="-.",
    linewidth=1.8,
    color="tab:green",
    label=(
        f"Observed effect "
        f"({OBSERVED_TRANSIENT_MS:.1f} ms)"
    ),
)

ax.set_title(
    "Transient post-flip effect"
)

ax.set_xlabel(
    "True RT effect (ms)"
)

ax.set_xlim(
    0,
    25,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# -----------------------------------------------------------------------------
# Shared legend below both panels
# -----------------------------------------------------------------------------

handles_left, labels_left = axes[0].get_legend_handles_labels()
handles_right, labels_right = axes[1].get_legend_handles_labels()

# Combine while avoiding duplicate "Estimated power" / "80% power" entries.
legend_items = {}

for handle, label in zip(
    handles_left + handles_right,
    labels_left + labels_right,
):
    if label not in legend_items:
        legend_items[label] = handle

fig.legend(
    legend_items.values(),
    legend_items.keys(),
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08),
)

plt.tight_layout(
    rect=[0, 0.14, 1, 1]
)

if SAVE_RESULTS:

    plt.savefig(
        FIGURE_FILE,
        bbox_inches="tight",
    )

plt.show()