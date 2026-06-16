# =============================================================================
# Imports
# =============================================================================

import glob
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fooof import FOOOF
from scipy.signal import welch
from joblib import Parallel, delayed
from mne.stats import permutation_cluster_1samp_test
import seaborn as sns

# =============================================================================
# Settings
# =============================================================================

path_in = "/mnt/data_dump/pixelflip/2_cleaned/"
path_out = "/mnt/data_dump/pixelflip/"

datasets = glob.glob(f"{path_in}/*cue_erp.set")

# PSD window
tmin_psd, tmax_psd = 0.0, 1.2
fmin, fmax = 1, 30

# Welch settings
welch_window = "hann"
welch_nperseg = None   # None = full epoch length
welch_noverlap = 0
welch_nfft_factor = 1  # use 4 for zero-padded display grid, 1 for no padding

# Slow-wave / CNV window
tmin_slow, tmax_slow = 0.7, 1.2

# Parallel FOOOF
n_jobs_fooof = -1

bands = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta":  (18, 30),
}

fooof_settings = dict(
    peak_width_limits=[1, 8],
    max_n_peaks=8,
    min_peak_height=0.05,
    aperiodic_mode="fixed",
    verbose=False,
)


# =============================================================================
# Containers
# =============================================================================

all_fooof_rows = []
all_psd_rows = []
all_cnv_rows = []


# =============================================================================
# Helpers
# =============================================================================

def plot_state_cluster_overview(
    results,
    measures,
    contrasts=("state", "difficulty", "interaction"),
    alpha=0.05,
    figsize=None,
    cmap="RdBu_r",
    symmetric_vlim=True,
):
    """
    Rows = measures
    Columns = contrasts
    Significant cluster electrodes are marked.

    Uses one global color scale across all panels by default.
    """

    n_rows = len(measures)
    n_cols = len(contrasts)

    if figsize is None:
        figsize = (3.2 * n_cols, 2.6 * n_rows)

    # Global vlim
    all_t = []
    for key, res in results.items():
        all_t.append(res["T_obs"])

    all_t = np.concatenate(all_t)

    if symmetric_vlim:
        vmax = np.nanmax(np.abs(all_t))
        vlim = (-vmax, vmax)
    else:
        vlim = (np.nanmin(all_t), np.nanmax(all_t))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = axes[None, :]

    if n_cols == 1:
        axes = axes[:, None]

    for r, measure in enumerate(measures):

        for c, contrast in enumerate(contrasts):

            ax = axes[r, c]
            res = results[(measure, contrast)]

            sig_mask = np.zeros(len(res["electrodes"]), dtype=bool)

            sig_ps = []

            for cluster, p in zip(
                res["clusters"],
                res["cluster_p_values"],
            ):
                if p < alpha:
                    sig_mask |= cluster
                    sig_ps.append(p)

            mne.viz.plot_topomap(
                res["T_obs"],
                res["info"],
                axes=ax,
                show=False,
                cmap=cmap,
                vlim=vlim,
                contours=0,
                mask=sig_mask,
                mask_params=dict(
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="k",
                    linewidth=1.2,
                    markersize=5,
                ),
            )

            if r == 0:
                ax.set_title(contrast, fontsize=11)

            if c == 0:
                ax.text(
                    -0.25,
                    0.5,
                    measure,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                )

            if len(sig_ps):
                ax.set_xlabel(
                    "min p = {:.4f}".format(min(sig_ps)),
                    fontsize=8,
                )
            else:
                ax.set_xlabel("n.s.", fontsize=8)

    fig.suptitle(
        "Topographic cluster tests",
        fontsize=14,
    )

    return fig

def run_all_state_cluster_tests(
    df,
    info,
    measures,
    contrasts=("state", "difficulty", "interaction"),
    n_permutations=5000,
    threshold=None,
    tail=0,
    seed=42,
):
    """
    Run cluster tests for all measures × contrasts.

    Returns
    -------
    results : dict
        keys: (measure, contrast)
    """

    results = {}

    for measure in measures:
        for contrast in contrasts:

            print(f"Running {measure} | {contrast}")

            results[(measure, contrast)] = run_state_topo_cluster_test(
                df=df,
                measure=measure,
                info=info,
                contrast_type=contrast,
                n_permutations=n_permutations,
                threshold=threshold,
                tail=tail,
                seed=seed,
            )

    return results

def run_state_topo_cluster_test(
    df,
    measure,
    info,
    contrast_type="state",
    state_levels=("0", "1"),
    difficulty_levels=("easy", "hard"),
    n_permutations=5000,
    threshold=None,
    tail=0,
    seed=42,
):
    """
    Topographic cluster test for state-coded dataframe.
    """

    X, subjects, electrodes = _make_state_contrast_matrix(
        df=df,
        measure=measure,
        contrast_type=contrast_type,
        state_levels=state_levels,
        difficulty_levels=difficulty_levels,
    )

    picks = mne.pick_channels(
        info["ch_names"],
        include=electrodes,
    )

    info_use = mne.pick_info(info, picks)

    if info_use["ch_names"] != electrodes:
        raise RuntimeError("Channel order mismatch between X and info.")

    adjacency, _ = mne.channels.find_ch_adjacency(
        info_use,
        ch_type="eeg",
    )

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        out_type="mask",
        seed=seed,
        n_jobs=1,
    )

    return {
        "measure": measure,
        "contrast_type": contrast_type,
        "state_levels": state_levels,
        "X": X,
        "subjects": subjects,
        "electrodes": electrodes,
        "info": info_use,
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "H0": H0,
    }

def _make_state_contrast_matrix(
    df,
    measure,
    contrast_type="state",
    state_levels=("0", "1"),
    difficulty_levels=("easy", "hard"),
):
    """
    Returns X, subjects, electrodes.
    X shape: subjects × electrodes

    contrast_type:
        "state"       = ((easy_1 + hard_1) / 2) - ((easy_0 + hard_0) / 2)
        "difficulty"  = ((hard_0 + hard_1) / 2) - ((easy_0 + easy_1) / 2)
        "interaction" = (hard_1 - hard_0) - (easy_1 - easy_0)
    """

    s0, s1 = state_levels
    d0, d1 = difficulty_levels

    wide = df.pivot_table(
        index=["subject", "electrode"],
        columns=["difficulty", "state"],
        values=measure,
        aggfunc="mean",
        observed=True,
    )

    if contrast_type == "state":
        contrast = (
            (wide[(d0, s1)] + wide[(d1, s1)]) / 2
            - (wide[(d0, s0)] + wide[(d1, s0)]) / 2
        )

    elif contrast_type == "difficulty":
        contrast = (
            (wide[(d1, s0)] + wide[(d1, s1)]) / 2
            - (wide[(d0, s0)] + wide[(d0, s1)]) / 2
        )

    elif contrast_type == "interaction":
        contrast = (
            wide[(d1, s1)] - wide[(d1, s0)]
            - (wide[(d0, s1)] - wide[(d0, s0)])
        )

    else:
        raise ValueError(
            "contrast_type must be 'state', 'difficulty', or 'interaction'"
        )

    contrast = contrast.dropna()
    contrast_df = contrast.reset_index(name="contrast")

    X_df = contrast_df.pivot(
        index="subject",
        columns="electrode",
        values="contrast",
    )

    X_df = X_df.dropna(axis=0, how="any")

    return X_df.values, X_df.index.to_numpy(), X_df.columns.to_list()

def compute_trialwise_welch_psd(
    data,
    sfreq,
    fmin=1,
    fmax=30,
    window="hann",
    nperseg=None,
    noverlap=0,
    nfft_factor=1,
):
    """
    Compute Welch PSD separately for each epoch, then average across trials.

    Parameters
    ----------
    data : ndarray
        Shape: trials × channels × times.

    Returns
    -------
    psd_mean : ndarray
        Shape: channels × freqs.
    freqs : ndarray
        Frequency vector.
    """

    n_trials, n_channels, n_times = data.shape

    if nperseg is None:
        nperseg = n_times

    nfft = int(nfft_factor * nperseg)

    freqs, psd = welch(
        data,
        fs=sfreq,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        axis=-1,
        scaling="density",
        average="mean",
    )

    # psd shape: trials × channels × freqs
    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    psd = psd[:, :, freq_mask]
    freqs = freqs[freq_mask]

    psd_mean = psd.mean(axis=0)  # channels × freqs

    return psd_mean, freqs


def fit_fooof_one_channel(
    ch_idx,
    ch_name,
    spectrum,
    freqs,
    subject_id,
    cond_name,
    n_trials,
    bands,
    fooof_settings,
    fmin,
    fmax,
):
    """Fit FOOOF to one electrode spectrum."""

    if np.any(~np.isfinite(spectrum)) or np.any(spectrum <= 0):
        return None

    fm = FOOOF(**fooof_settings)
    fm.fit(freqs, spectrum, [fmin, fmax])

    offset, exponent = fm.aperiodic_params_

    log_power = np.log10(spectrum)
    ap_fit = offset - exponent * np.log10(freqs)
    flat_power = log_power - ap_fit

    row = {
        "subject": subject_id,
        "condition": cond_name,
        "electrode": ch_name,
        "n_trials": n_trials,
        "fooof_offset": offset,
        "fooof_exponent": exponent,
        "fooof_r_squared": fm.r_squared_,
        "fooof_error": fm.error_,
        "fooof_n_peaks": fm.n_peaks_,
    }

    for band_name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs <= hi)
        row[f"{band_name}_flat_power"] = (
            np.nan if band_mask.sum() == 0 else flat_power[band_mask].mean()
        )

    return row


def add_condition_factors(df):
    """
    For conditions like easy_0, easy_1, hard_0, hard_1.
    state:
        0 = flips impossible / reliable
        1 = flips possible / unreliable
    """

    df["difficulty"] = df["condition"].str.split("_").str[0]
    df["state"] = df["condition"].str.split("_").str[1]

    df["difficulty"] = pd.Categorical(
        df["difficulty"],
        categories=["easy", "hard"],
        ordered=True,
    )

    df["state"] = pd.Categorical(
        df["state"],
        categories=["0", "1"],
        ordered=True,
    )

    return df

def make_psd_band_df(psd_df, bands, value_col="log10_psd"):
    rows = []

    for band_name, (lo, hi) in bands.items():

        tmp = (
            psd_df[
                (psd_df["freq"] >= lo)
                & (psd_df["freq"] <= hi)
            ]
            .groupby(
                ["subject", "condition", "electrode", "difficulty", "state"],
                observed=True,
                as_index=False,
            )[value_col]
            .mean()
        )

        tmp = tmp.rename(columns={value_col: f"{band_name}_{value_col}"})
        rows.append(tmp)

    out = rows[0]

    for tmp in rows[1:]:
        out = out.merge(
            tmp,
            on=["subject", "condition", "electrode", "difficulty", "state"],
            how="outer",
        )

    return out

def plot_cluster_psd(
    psd_df,
    electrodes,
    value_col="log10_psd",
):
    
    plot_df = (
        psd_df[
            psd_df["electrode"].isin(electrodes)
        ]
        .groupby(
            ["subject", "difficulty", "state", "freq"],
            observed=True,
            as_index=False,
        )[value_col]
        .mean()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        sharey=True,
        constrained_layout=True,
    )

    for ax, difficulty in zip(
        axes,
        ["easy", "hard"],
    ):

        tmp = plot_df.query(
            "difficulty == @difficulty"
        )

        sns.lineplot(
            data=tmp,
            x="freq",
            y=value_col,
            hue="state",
            errorbar="se",
            ax=ax,
        )

        ax.axvspan(
            8,
            12,
            color="grey",
            alpha=.15,
        )

        ax.set_title(difficulty)

    fig.suptitle(
        f"Cluster-average PSD ({len(electrodes)} electrodes)"
    )

    return fig


def plot_cluster_difference(
    psd_df,
    electrodes,
):
    
    tmp = (
        psd_df[
            psd_df["electrode"].isin(electrodes)
        ]
        .groupby(
            ["subject", "difficulty", "state", "freq"],
            observed=True,
            as_index=False,
        )["log10_psd"]
        .mean()
    )

    wide = tmp.pivot_table(
        index=["subject", "difficulty", "freq"],
        columns="state",
        values="log10_psd",
        observed=True,
    )

    wide["diff"] = wide["1"] - wide["0"]

    diff_df = (
        wide["diff"]
        .reset_index()
    )

    plt.figure(figsize=(8, 4))

    sns.lineplot(
        data=diff_df,
        x="freq",
        y="diff",
        hue="difficulty",
        errorbar="se",
    )

    plt.axhline(
        0,
        color="k",
        lw=1,
    )

    plt.axvspan(
        8,
        12,
        color="grey",
        alpha=.15,
    )

    plt.ylabel("state1 - state0")
    plt.title("Cluster-average PSD difference")

    plt.show()

# =============================================================================
# Main loop
# =============================================================================

info = None

for dataset in datasets:

    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])

    if subject_id == 7:
        continue

    print(f"Processing subject {subject_id}")

    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(-0.2, 0))
        .crop(tmin=-0.2, tmax=1.8)
    )

    if info is None:
        info = eeg_epochs.info.copy()

    trialinfo = pd.read_csv(dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv")

    idx_easy_0 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 1)
    )[0]

    idx_easy_1 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 0)
    )[0]

    idx_hard_0 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 1)
    )[0]

    idx_hard_1 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 0)
    )[0]

    conditions = {
        "easy_0": idx_easy_0,
        "easy_1": idx_easy_1,
        "hard_0": idx_hard_0,
        "hard_1": idx_hard_1,
    }

    epochs_psd = eeg_epochs.copy().crop(tmin=tmin_psd, tmax=tmax_psd)

    sfreq = epochs_psd.info["sfreq"]
    ch_names = epochs_psd.ch_names

    for cond_name, idx in conditions.items():

        if len(idx) == 0:
            print(f"Skipping {subject_id} {cond_name}: no trials")
            continue

        print(f"  {cond_name}: {len(idx)} trials")

        # ---------------------------------------------------------------------
        # CNV / slow wave
        # ---------------------------------------------------------------------

        evoked = eeg_epochs[idx].average()

        slow_wave_uv = (
            evoked.copy()
            .crop(tmin=tmin_slow, tmax=tmax_slow)
            .data
            .mean(axis=1)
            * 1e6
        )

        for ch_idx, ch_name in enumerate(ch_names):
            all_cnv_rows.append({
                "subject": subject_id,
                "condition": cond_name,
                "electrode": ch_name,
                "n_trials": len(idx),
                "slow_wave_uv": slow_wave_uv[ch_idx],
            })

        # ---------------------------------------------------------------------
        # PSD
        # ---------------------------------------------------------------------

        data = epochs_psd[idx].get_data()  # trials × channels × times

        psd_mean, freqs = compute_trialwise_welch_psd(
            data=data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            window=welch_window,
            nperseg=welch_nperseg,
            noverlap=welch_noverlap,
            nfft_factor=welch_nfft_factor,
        )

        log10_psd = np.log10(psd_mean)

        # Save PSD long-format
        for ch_idx, ch_name in enumerate(ch_names):
            for fi, freq in enumerate(freqs):
                all_psd_rows.append({
                    "subject": subject_id,
                    "condition": cond_name,
                    "electrode": ch_name,
                    "n_trials": len(idx),
                    "freq": freq,
                    "psd": psd_mean[ch_idx, fi],
                    "log10_psd": log10_psd[ch_idx, fi],
                })

        # ---------------------------------------------------------------------
        # FOOOF
        # ---------------------------------------------------------------------

        rows = Parallel(n_jobs=n_jobs_fooof, prefer="processes")(
            delayed(fit_fooof_one_channel)(
                ch_idx=ch_idx,
                ch_name=ch_name,
                spectrum=psd_mean[ch_idx, :],
                freqs=freqs,
                subject_id=subject_id,
                cond_name=cond_name,
                n_trials=len(idx),
                bands=bands,
                fooof_settings=fooof_settings,
                fmin=fmin,
                fmax=fmax,
            )
            for ch_idx, ch_name in enumerate(ch_names)
        )

        all_fooof_rows.extend([r for r in rows if r is not None])


# =============================================================================
# Dataframes
# =============================================================================

fooof_df = pd.DataFrame(all_fooof_rows)
psd_df = pd.DataFrame(all_psd_rows)
cnv_df = pd.DataFrame(all_cnv_rows)

fooof_df = add_condition_factors(fooof_df)
psd_df = add_condition_factors(psd_df)
cnv_df = add_condition_factors(cnv_df)


# =============================================================================
# Save
# =============================================================================

fooof_df.to_pickle(f"{path_out}/fooof_state_df.pkl")
psd_df.to_pickle(f"{path_out}/psd_state_df.pkl")
cnv_df.to_pickle(f"{path_out}/cnv_state_df.pkl")




fooof_measures = [
    "fooof_offset",
    "fooof_exponent",
    "delta_flat_power",
    "theta_flat_power",
    "alpha_flat_power",
    "beta_flat_power",
]

cnv_measures = [
    "slow_wave_uv",
]

psd_measures = [
    # define these only if you add band averages to psd_df
]


cnv_results = run_all_state_cluster_tests(
    df=cnv_df,
    info=info,
    measures=cnv_measures,
    contrasts=("state", "difficulty", "interaction"),
    n_permutations=5000,
)

fig = plot_state_cluster_overview(
    results=cnv_results,
    measures=cnv_measures,
    contrasts=("state", "difficulty", "interaction"),
)

plt.show()




fooof_results = run_all_state_cluster_tests(
    df=fooof_df,
    info=info,
    measures=fooof_measures,
    contrasts=("state", "difficulty", "interaction"),
    n_permutations=5000,
)

fig = plot_state_cluster_overview(
    results=fooof_results,
    measures=fooof_measures,
    contrasts=("state", "difficulty", "interaction"),
)

plt.show()



psd_band_df = make_psd_band_df(
    psd_df,
    bands=bands,
    value_col="log10_psd",
)

psd_measures = [
    "delta_log10_psd",
    "theta_log10_psd",
    "alpha_log10_psd",
    "beta_log10_psd",
]

psd_results = run_all_state_cluster_tests(
    df=psd_band_df,
    info=info,
    measures=psd_measures,
    contrasts=("state", "difficulty", "interaction"),
    n_permutations=5000,
)

fig = plot_state_cluster_overview(
    results=psd_results,
    measures=psd_measures,
    contrasts=("state", "difficulty", "interaction"),
)

plt.show()




alpha_res = psd_results[("alpha_log10_psd", "state")]

sig_idx = np.where(alpha_res["cluster_p_values"] < 0.05)[0][0]
cluster_mask = alpha_res["clusters"][sig_idx]

cluster_electrodes = np.array(
    alpha_res["electrodes"]
)[cluster_mask]

print(cluster_electrodes)


plot_cluster_psd(
    psd_df,
    cluster_electrodes,
)

plot_cluster_difference(
    psd_df,
    cluster_electrodes,
)