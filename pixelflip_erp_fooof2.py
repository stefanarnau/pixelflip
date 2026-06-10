# =============================================================================
# Imports
# =============================================================================

import glob
import pickle

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fooof import FOOOF
from scipy.signal import welch, get_window
from joblib import Parallel, delayed
from mne.stats import permutation_cluster_1samp_test


# =============================================================================
# Settings
# =============================================================================

path_in = "/mnt/data_dump/pixelflip/2_cleaned/"
path_out = "/mnt/data_dump/pixelflip/"

datasets = glob.glob(f"{path_in}/*cue_erp.set")

# PSD window
tmin_psd, tmax_psd = 0, 1.2
fmin, fmax = 1, 30

# Concatenated Welch PSD settings
welch_trials_per_segment = 4
welch_overlap = 0.80

# Slow-wave / CNV window
tmin_slow, tmax_slow = 0.7, 1.2

# Parallel FOOOF
n_jobs_fooof = -1  # use all cores; set e.g. 8 if memory gets high

# Freq bands on flattened spectrum
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
all_erp_arrays = []
all_psd_arrays = []


# =============================================================================
# Helpers
# =============================================================================

def concatenated_epoch_welch(
    data,
    sfreq,
    n_trials_per_segment=4,
    overlap=0.80,
    fmin=1,
    fmax=30,
):
    """
    Hann-window epochs, concatenate trials, then compute Welch PSD.

    data shape: trials × channels × times
    returns: psd channels × freqs, freqs
    """

    n_trials, n_channels, n_times = data.shape

    if n_trials < n_trials_per_segment:
        raise ValueError(
            f"Need at least {n_trials_per_segment} trials, got {n_trials}"
        )

    epoch_window = get_window("hann", n_times)
    data_win = data * epoch_window[None, None, :]

    concat = np.concatenate(
        [data_win[i] for i in range(n_trials)],
        axis=1,
    )  # channels × concatenated_time

    nperseg = n_trials_per_segment * n_times
    noverlap = int(overlap * nperseg)

    freqs, psd = welch(
        concat,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend="constant",
        axis=-1,
        scaling="density",
        average="mean",
    )

    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    return psd[:, freq_mask], freqs[freq_mask]


def fit_fooof_one_channel(
    ch_idx,
    ch_name,
    spectrum,
    freqs,
    subject_id,
    cond_name,
    n_trials,
    slow_wave_uv,
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
        "slow_wave_uv": slow_wave_uv[ch_idx],
    }

    for band_name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs <= hi)
        row[f"{band_name}_flat_power"] = (
            np.nan if band_mask.sum() == 0 else flat_power[band_mask].mean()
        )

    return row


def add_condition_factors(df):
    """Add difficulty and flip columns from condition."""

    df["difficulty"] = df["condition"].str.split("_").str[0]
    df["flip"] = df["condition"].str.split("_").str[1]

    df["difficulty"] = pd.Categorical(
        df["difficulty"],
        categories=["easy", "hard"],
        ordered=True,
    )

    df["flip"] = pd.Categorical(
        df["flip"],
        categories=["00", "10", "11"],
        ordered=True,
    )

    return df


def plot_interaction_measures_by_electrode(
    df,
    electrode,
    measures=None,
    flip_order=None,
    difficulty_order=None,
    figsize=(14, 10),
    errorbar="se",
):
    """Multi-panel flip × difficulty interaction plot for one electrode."""

    if measures is None:
        measures = [
            "fooof_offset",
            "fooof_exponent",
            "fooof_r_squared",
            "fooof_error",
            "fooof_n_peaks",
            "delta_flat_power",
            "theta_flat_power",
            "alpha_flat_power",
            "beta_flat_power",
            "slow_wave_uv",
        ]

    if difficulty_order is None:
        difficulty_order = ["easy", "hard"]

    if flip_order is None:
        flip_order = list(df["flip"].cat.categories)
        flip_order = [f for f in flip_order if f in df["flip"].astype(str).unique()]

    df_elec = df.query("electrode == @electrode").copy()

    if df_elec.empty:
        raise ValueError(f"No data found for electrode: {electrode}")

    measures = [m for m in measures if m in df_elec.columns]

    df_long = df_elec.melt(
        id_vars=["subject", "condition", "electrode", "difficulty", "flip"],
        value_vars=measures,
        var_name="measure",
        value_name="value",
    )

    df_long["difficulty"] = pd.Categorical(
        df_long["difficulty"],
        categories=difficulty_order,
        ordered=True,
    )

    df_long["flip"] = pd.Categorical(
        df_long["flip"].astype(str),
        categories=flip_order,
        ordered=True,
    )

    n_measures = len(measures)
    n_cols = 3
    n_rows = int(np.ceil(n_measures / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, measure in zip(axes, measures):

        plot_df = df_long[df_long["measure"] == measure].dropna()

        sns.pointplot(
            data=plot_df,
            x="flip",
            y="value",
            hue="difficulty",
            order=flip_order,
            hue_order=difficulty_order,
            errorbar=errorbar,
            dodge=0.08,
            markers="o",
            linestyles="-",
            ax=ax,
        )

        ax.set_title(measure)
        ax.set_xlabel("Flip condition")
        ax.set_ylabel("")

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for ax in axes[n_measures:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Difficulty",
        loc="upper right",
        bbox_to_anchor=(1.02, 1.0),
    )

    fig.suptitle(f"Interaction plots at {electrode}", fontsize=16)

    return fig


def arrays_to_plot_df(arrays, kind, electrodes, value_kind="erp_uv"):
    """
    Convert compact ERP/PSD arrays to plotting dataframe.

    kind: "erp" or "psd"
    electrodes: str or list[str]
    value_kind for psd: "psd", "log10_psd"
    """

    if isinstance(electrodes, str):
        electrodes = [electrodes]

    rows = []

    for item in arrays:

        ch_names = item["ch_names"]
        picks = [ch_names.index(ch) for ch in electrodes if ch in ch_names]

        if len(picks) == 0:
            continue

        if kind == "erp":
            x = item["times"]
            y = item["erp_uv"][picks, :].mean(axis=0)
            x_name = "time"
            y_name = "erp_uv"

        elif kind == "psd":
            x = item["freqs"]
            y = item[value_kind][picks, :].mean(axis=0)
            x_name = "freq"
            y_name = value_kind

        else:
            raise ValueError("kind must be 'erp' or 'psd'")

        tmp = pd.DataFrame({
            "subject": item["subject"],
            "condition": item["condition"],
            "n_trials": item["n_trials"],
            x_name: x,
            y_name: y,
        })

        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out = add_condition_factors(out)

    return out


def plot_erp_psd_4panel_from_arrays(
    all_erp_arrays,
    all_psd_arrays,
    electrode,
    conditions=("00", "10"),
    psd_value_kind="log10_psd",
    time_window=(-0.2, 1.2),
    freq_window=(1, 30),
    errorbar="se",
    figsize=(12, 8),
):
    """
    Four-panel plot:
    upper row ERP, lower row PSD.
    easy left, hard right.
    """

    erp_plot = arrays_to_plot_df(
        all_erp_arrays,
        kind="erp",
        electrodes=electrode,
    )

    psd_plot = arrays_to_plot_df(
        all_psd_arrays,
        kind="psd",
        electrodes=electrode,
        value_kind=psd_value_kind,
    )

    erp_plot = erp_plot[
        (erp_plot["time"] >= time_window[0])
        & (erp_plot["time"] <= time_window[1])
        & (erp_plot["flip"].astype(str).isin(conditions))
    ].copy()

    psd_plot = psd_plot[
        (psd_plot["freq"] >= freq_window[0])
        & (psd_plot["freq"] <= freq_window[1])
        & (psd_plot["flip"].astype(str).isin(conditions))
    ].copy()

    if isinstance(electrode, str):
        elec_label = electrode
    else:
        elec_label = f"{len(electrode)}-electrode average"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        constrained_layout=True,
    )

    for col, difficulty in enumerate(["easy", "hard"]):

        erp_d = erp_plot[erp_plot["difficulty"].astype(str) == difficulty]
        psd_d = psd_plot[psd_plot["difficulty"].astype(str) == difficulty]

        sns.lineplot(
            data=erp_d,
            x="time",
            y="erp_uv",
            hue="flip",
            hue_order=list(conditions),
            estimator="mean",
            errorbar=errorbar,
            ax=axes[0, col],
        )

        axes[0, col].axvline(0, color="k", linewidth=0.8, linestyle="--")
        axes[0, col].axhline(0, color="k", linewidth=0.8)
        axes[0, col].set_title(f"ERP | {difficulty}")
        axes[0, col].set_xlabel("Time from cue (s)")
        axes[0, col].set_ylabel("Amplitude (µV)")

        sns.lineplot(
            data=psd_d,
            x="freq",
            y=psd_value_kind,
            hue="flip",
            hue_order=list(conditions),
            estimator="mean",
            errorbar=errorbar,
            ax=axes[1, col],
        )

        axes[1, col].set_title(f"PSD | {difficulty}")
        axes[1, col].set_xlabel("Frequency (Hz)")
        axes[1, col].set_ylabel(psd_value_kind)

        if col == 0:
            for ax in [axes[0, col], axes[1, col]]:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        else:
            for ax in [axes[0, col], axes[1, col]]:
                if ax.get_legend() is not None:
                    ax.get_legend().set_title("Flip")

    fig.suptitle(f"ERP and PSD | {elec_label}", fontsize=16)

    return fig


def _make_contrast_matrix(
    df,
    measure,
    contrast_type,
    flip_levels=None,
    difficulty_levels=("easy", "hard"),
):
    """Return X matrix: subjects × electrodes."""

    if flip_levels is None:
        flip_levels = list(df["flip"].cat.categories)
        flip_levels = [f for f in flip_levels if f in df["flip"].astype(str).unique()]

    if len(flip_levels) != 2:
        raise ValueError(f"Expected exactly two flip levels, got {flip_levels}")

    f0, f1 = flip_levels
    d0, d1 = difficulty_levels

    wide = df.pivot_table(
        index=["subject", "electrode"],
        columns=["difficulty", "flip"],
        values=measure,
        aggfunc="mean",
        observed=True,
    )

    if contrast_type == "flip":
        contrast = (
            (wide[(d0, f1)] + wide[(d1, f1)]) / 2
            - (wide[(d0, f0)] + wide[(d1, f0)]) / 2
        )

    elif contrast_type == "difficulty":
        contrast = (
            (wide[(d1, f0)] + wide[(d1, f1)]) / 2
            - (wide[(d0, f0)] + wide[(d0, f1)]) / 2
        )

    elif contrast_type == "interaction":
        contrast = (
            wide[(d1, f1)] - wide[(d1, f0)]
            - (wide[(d0, f1)] - wide[(d0, f0)])
        )

    else:
        raise ValueError("contrast_type must be 'flip', 'difficulty', or 'interaction'")

    contrast = contrast.dropna()
    contrast_df = contrast.reset_index(name="contrast")

    X_df = contrast_df.pivot(
        index="subject",
        columns="electrode",
        values="contrast",
    )

    X_df = X_df.dropna(axis=0, how="any")

    return X_df.values, X_df.index.to_numpy(), X_df.columns.to_list()


def run_topo_cluster_test(
    df,
    measure,
    info,
    contrast_type="flip",
    flip_levels=None,
    difficulty_levels=("easy", "hard"),
    n_permutations=5000,
    threshold=None,
    tail=0,
    seed=42,
):
    """Topographic cluster test across electrodes."""

    X, subjects, electrodes = _make_contrast_matrix(
        df=df,
        measure=measure,
        contrast_type=contrast_type,
        flip_levels=flip_levels,
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
        "flip_levels": flip_levels,
        "X": X,
        "subjects": subjects,
        "electrodes": electrodes,
        "info": info_use,
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "H0": H0,
    }


def print_cluster_results(res, alpha=0.05):
    for i, p in enumerate(res["cluster_p_values"]):
        if p < alpha:
            chans = np.array(res["electrodes"])[res["clusters"][i]]
            print(f"Cluster {i}: p = {p:.4f}")
            print(list(chans))


def plot_cluster_topomap(res, alpha=0.05):
    sig_mask = np.zeros(len(res["electrodes"]), dtype=bool)

    for cluster, p in zip(res["clusters"], res["cluster_p_values"]):
        if p < alpha:
            sig_mask |= cluster

    fig, ax = plt.subplots(figsize=(5, 4))

    mne.viz.plot_topomap(
        res["T_obs"],
        res["info"],
        axes=ax,
        show=False,
        mask=sig_mask,
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            linewidth=1.5,
            markersize=8,
        ),
        contours=0,
    )

    ax.set_title(f"{res['measure']} | {res['contrast_type']}")
    plt.show()

    return fig

def run_all_cluster_tests(
    df_state,
    df_sequence,
    info,
    measures,
    n_permutations=5000,
):
    results = {}

    for measure in measures:

        results[(measure, "state_flip")] = run_topo_cluster_test(
            df=df_state,
            measure=measure,
            info=info,
            contrast_type="flip",
            flip_levels=["00", "10"],
            n_permutations=n_permutations,
        )

        results[(measure, "sequence_flip")] = run_topo_cluster_test(
            df=df_sequence,
            measure=measure,
            info=info,
            contrast_type="flip",
            flip_levels=["10", "11"],
            n_permutations=n_permutations,
        )

        results[(measure, "state_interaction")] = run_topo_cluster_test(
            df=df_state,
            measure=measure,
            info=info,
            contrast_type="interaction",
            flip_levels=["00", "10"],
            n_permutations=n_permutations,
        )

        results[(measure, "sequence_interaction")] = run_topo_cluster_test(
            df=df_sequence,
            measure=measure,
            info=info,
            contrast_type="interaction",
            flip_levels=["10", "11"],
            n_permutations=n_permutations,
        )

    return results

def plot_cluster_overview(
    results,
    measures,
    effects=(
        "state_flip",
        "sequence_flip",
        "state_interaction",
        "sequence_interaction",
    ),
    alpha=0.05,
    figsize=(12, 14),
):
    """
    Rows = measures
    Columns = effects
    """

    n_rows = len(measures)
    n_cols = len(effects)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = axes[None, :]

    for r, measure in enumerate(measures):

        for c, effect in enumerate(effects):

            ax = axes[r, c]

            res = results[(measure, effect)]

            sig_mask = np.zeros(
                len(res["electrodes"]),
                dtype=bool,
            )

            for cluster, p in zip(
                res["clusters"],
                res["cluster_p_values"],
            ):
                if p < alpha:
                    sig_mask |= cluster

            mne.viz.plot_topomap(
                res["T_obs"],
                res["info"],
                axes=ax,
                show=False,
                mask=sig_mask,
                mask_params=dict(
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="k",
                    linewidth=1.5,
                    markersize=6,
                ),
                contours=0,
            )

            if r == 0:
                ax.set_title(effect)

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

            sig_ps = [
                p
                for p in res["cluster_p_values"]
                if p < alpha
            ]

            if len(sig_ps):
                ax.set_xlabel(
                    f"p={min(sig_ps):.3f}",
                    fontsize=8,
                )

    return fig

def inspect_cluster(
    df,
    cluster_result,
    measure,
    cluster_idx=0,
):
    """
    Extract significant cluster electrodes and plot cluster-average data.

    Parameters
    ----------
    df : dataframe
        df_state or df_sequence
    cluster_result : dict
        Output from run_topo_cluster_test()
    measure : str
        e.g. "beta_flat_power"
    cluster_idx : int
        Which cluster to inspect.
    """

    mask = cluster_result["clusters"][cluster_idx]

    electrodes = np.array(
        cluster_result["electrodes"]
    )[mask]

    print("\nCluster electrodes:")
    print(list(electrodes))

    plot_df = (
        df[df["electrode"].isin(electrodes)]
        .groupby(
            ["subject", "difficulty", "flip"],
            observed=True
        )[measure]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(6, 4))

    sns.pointplot(
        data=plot_df,
        x="flip",
        y=measure,
        hue="difficulty",
        errorbar="se",
        dodge=0.1,
    )

    plt.title(
        f"{measure}\nCluster {cluster_idx} "
        f"(p={cluster_result['cluster_p_values'][cluster_idx]:.4f})"
    )

    plt.show()

    return electrodes, plot_df


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

    idx_easy_00 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 1)
        & (trialinfo.prev_accuracy == 1)
    )[0]

    idx_easy_10 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 0)
    )[0]

    idx_easy_11 = np.where(
        (trialinfo.difficulty == 0)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 1)
    )[0]

    idx_hard_00 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 1)
        & (trialinfo.prev_accuracy == 1)
    )[0]

    idx_hard_10 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 0)
    )[0]

    idx_hard_11 = np.where(
        (trialinfo.difficulty == 1)
        & (trialinfo.reliability == 0)
        & (trialinfo.prev_accuracy == 1)
        & (trialinfo.prev_flipped == 1)
    )[0]

    conditions = {
        "easy_00": idx_easy_00,
        "easy_10": idx_easy_10,
        "easy_11": idx_easy_11,
        "hard_00": idx_hard_00,
        "hard_10": idx_hard_10,
        "hard_11": idx_hard_11,
    }

    epochs_psd = eeg_epochs.copy().crop(tmin=tmin_psd, tmax=tmax_psd)

    sfreq = epochs_psd.info["sfreq"]
    ch_names = epochs_psd.ch_names

    for cond_name, idx in conditions.items():

        if len(idx) == 0:
            print(f"Skipping {subject_id} {cond_name}: no trials")
            continue

        print(f"  {cond_name}: {len(idx)} trials")

        # ERP
        evoked = eeg_epochs[idx].average()

        all_erp_arrays.append({
            "subject": subject_id,
            "condition": cond_name,
            "n_trials": len(idx),
            "ch_names": ch_names.copy(),
            "times": evoked.times.astype(np.float32),
            "erp_uv": (evoked.data * 1e6).astype(np.float32),
        })

        # Slow-wave / CNV
        slow_wave_uv = (
            evoked.copy()
            .crop(tmin=tmin_slow, tmax=tmax_slow)
            .data
            .mean(axis=1)
            * 1e6
        )

        # PSD
        data = epochs_psd[idx].get_data()  # trials × channels × times

        try:
            psd_mean, freqs = concatenated_epoch_welch(
                data=data,
                sfreq=sfreq,
                n_trials_per_segment=welch_trials_per_segment,
                overlap=welch_overlap,
                fmin=fmin,
                fmax=fmax,
            )
        except ValueError as err:
            print(f"Skipping PSD/FOOOF {subject_id} {cond_name}: {err}")
            continue

        log10_psd = np.log10(psd_mean)

        all_psd_arrays.append({
            "subject": subject_id,
            "condition": cond_name,
            "n_trials": len(idx),
            "ch_names": ch_names.copy(),
            "freqs": freqs.astype(np.float32),
            "psd": psd_mean.astype(np.float32),
            "log10_psd": log10_psd.astype(np.float32),
        })

        # FOOOF in parallel over channels
        rows = Parallel(n_jobs=n_jobs_fooof, prefer="processes")(
            delayed(fit_fooof_one_channel)(
                ch_idx=ch_idx,
                ch_name=ch_name,
                spectrum=psd_mean[ch_idx, :],
                freqs=freqs,
                subject_id=subject_id,
                cond_name=cond_name,
                n_trials=len(idx),
                slow_wave_uv=slow_wave_uv,
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
fooof_df = add_condition_factors(fooof_df)

df_state = fooof_df[fooof_df["flip"].isin(["00", "10"])].copy()
df_sequence = fooof_df[fooof_df["flip"].isin(["10", "11"])].copy()


# =============================================================================
# Save compact outputs
# =============================================================================

fooof_df.to_pickle(
    f"{path_out}/fooof_measures_concatwelch.pkl"
)

with open(f"{path_out}/erp_arrays.pkl", "wb") as f:
    pickle.dump(all_erp_arrays, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{path_out}/psd_arrays.pkl", "wb") as f:
    pickle.dump(all_psd_arrays, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved:")
print(f"{path_out}/fooof_measures_concatwelch.pkl")
print(f"{path_out}/erp_arrays.pkl")
print(f"{path_out}/psd_arrays.pkl")


# =============================================================================
# Example plots
# =============================================================================

fig = plot_interaction_measures_by_electrode(
    df_state,
    electrode="FCz",
    flip_order=["00", "10"],
)
plt.show()

fig = plot_interaction_measures_by_electrode(
    df_sequence,
    electrode="FCz",
    flip_order=["10", "11"],
)
plt.show()

fig = plot_erp_psd_4panel_from_arrays(
    all_erp_arrays=all_erp_arrays,
    all_psd_arrays=all_psd_arrays,
    electrode="Cz",
    conditions=("00", "10"),
    psd_value_kind="log10_psd",
)
plt.show()

fig = plot_erp_psd_4panel_from_arrays(
    all_erp_arrays=all_erp_arrays,
    all_psd_arrays=all_psd_arrays,
    electrode="Cz",
    conditions=("10", "11"),
    psd_value_kind="log10_psd",
)
plt.show()


# =============================================================================
# Example cluster tests
# =============================================================================

measures = [
    "slow_wave_uv",
    "fooof_exponent",
    "fooof_offset",
    "delta_flat_power",
    "theta_flat_power",
    "alpha_flat_power",
    "beta_flat_power",
]

results = run_all_cluster_tests(
    df_state=df_state,
    df_sequence=df_sequence,
    info=info,
    measures=measures,
)

fig = plot_cluster_overview(
    results,
    measures,
    alpha=0.05,
)

plt.show()





cluster_beta = [
    "FC1","FCz","FC2",
    "C1","Cz","C2",
    "CP1","CPz","CP2"
]

cluster_df = (
    df_sequence
    .query("electrode in @cluster_beta")
    .groupby(
        ["subject","difficulty","flip"],
        observed=True
    )["beta_flat_power"]
    .mean()
    .reset_index()
)