# Imports
import mne
import glob
import pandas as pd
import numpy as np
import mne.stats 
from fooof import FOOOF
from mne.time_frequency import psd_array_multitaper
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne.stats import permutation_cluster_1samp_test
from mne.time_frequency import psd_array_multitaper, psd_array_welch

# Settings
tmin_psd, tmax_psd = 0.0, 1.2
fmin, fmax = 1, 30
mt_bandwidth = 1.5

# Slow-wave / CNV window
tmin_slow, tmax_slow = 0.7, 1.2

# Freqbands
bands = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta":  (18, 30),
}

fooof_settings = dict(
    peak_width_limits=[1, 8],
    max_n_peaks=6,
    min_peak_height=0.0,
    peak_threshold=2.0,
    aperiodic_mode="fixed",
    verbose=False,
)

all_fooof_rows = []
all_erp_rows = []
all_psd_rows = []

# Define paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")

# Helpers
def compute_psd(data, sfreq, fmin=1, fmax=30, method="multitaper", bandwidth=1.5):
    if method == "multitaper":
        psds, freqs = psd_array_multitaper(
            data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            bandwidth=bandwidth,
            adaptive=False,
            low_bias=True,
            normalization="full",
            output="power",
            n_jobs=1,
            verbose=False,
        )

    elif method == "welch":
        psds, freqs = psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=data.shape[-1],
            n_per_seg=data.shape[-1],
            n_overlap=0,
            window="hann",
            average="mean",
            verbose=False,
        )

    else:
        raise ValueError("method must be 'multitaper' or 'welch'")

    return psds, freqs

def plot_interaction_measures_by_electrode(
    df,
    electrode,
    measures=None,
    flip_order=None,
    difficulty_order=None,
    figsize=(14, 10),
    errorbar="se",
):
    """
    Multi-panel interaction plots for one electrode.

    Parameters
    ----------
    df : pandas.DataFrame
        Use df_state or df_sequence.
    electrode : str
        Electrode name, e.g. "FCz".
    measures : list[str] | None
        Columns to plot.
    flip_order : list[str] | None
        ["00", "10"] for state, or ["10", "11"] for sequence.
    difficulty_order : list[str] | None
        Usually ["easy", "hard"].
    errorbar : str or tuple
        Seaborn errorbar argument, e.g. "se", "sd", or ("ci", 95).
    """

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
        sharex=False,
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
        ax.tick_params(axis="x", rotation=0)

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

    fig.suptitle(
        f"Interaction plots at {electrode}",
        fontsize=16,
    )

    return fig

def _make_contrast_matrix(
    df,
    measure,
    contrast_type,
    flip_levels=None,
    difficulty_levels=("easy", "hard"),
):
    """
    Returns X, subjects, electrodes.
    X shape: subjects × electrodes
    """

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
    )

    if contrast_type == "flip":
        # average across difficulty first:
        # ((easy_f1 + hard_f1) / 2) - ((easy_f0 + hard_f0) / 2)
        contrast = (
            (wide[(d0, f1)] + wide[(d1, f1)]) / 2
            - (wide[(d0, f0)] + wide[(d1, f0)]) / 2
        )

    elif contrast_type == "difficulty":
        # average across flip first:
        # ((hard_f0 + hard_f1) / 2) - ((easy_f0 + easy_f1) / 2)
        contrast = (
            (wide[(d1, f0)] + wide[(d1, f1)]) / 2
            - (wide[(d0, f0)] + wide[(d0, f1)]) / 2
        )

    elif contrast_type == "interaction":
        # difficulty × flip interaction:
        # (hard_f1 - hard_f0) - (easy_f1 - easy_f0)
        contrast = (
            wide[(d1, f1)] - wide[(d1, f0)]
            - (wide[(d0, f1)] - wide[(d0, f0)])
        )

    else:
        raise ValueError("contrast_type must be 'flip', 'difficulty', or 'interaction'")

    contrast = contrast.dropna()

    contrast_df = contrast.reset_index(name="contrast")

    X = contrast_df.pivot(
        index="subject",
        columns="electrode",
        values="contrast",
    )

    X = X.dropna(axis=0, how="any")

    return X.values, X.index.to_numpy(), X.columns.to_list()

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
    """
    Topographic cluster test across electrodes.

    Parameters
    ----------
    df : DataFrame
        df_state or df_sequence.
    measure : str
        Dependent variable, e.g. "slow_wave_uv", "fooof_exponent".
    info : mne.Info
        EEG info object containing channel locations.
    contrast_type : str
        "flip", "difficulty", or "interaction".
    flip_levels : list[str] | None
        ["00", "10"] for state, ["10", "11"] for sequence.
    """

    X, subjects, electrodes = _make_contrast_matrix(
        df=df,
        measure=measure,
        contrast_type=contrast_type,
        flip_levels=flip_levels,
        difficulty_levels=difficulty_levels,
    )

    # restrict info to electrodes in X, same order
    picks = mne.pick_channels(
        info["ch_names"],
        include=electrodes,
    )
    
    info_use = mne.pick_info(
        info,
        picks,
    )

    adjacency, ch_names = mne.channels.find_ch_adjacency(
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

    result = {
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

    return result

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

    ax.set_title(
        f"{res['measure']} | {res['contrast_type']}"
    )

    plt.show()

    return fig

def plot_erp_psd_4panel(
    erp_df,
    psd_df,
    electrode,
    conditions=("00", "10", "11"),
    difficulties=("easy", "hard"),
    erp_time_col="time",
    erp_value_col="erp_uv",
    psd_freq_col="freq",
    psd_value_col="log10_psd",
    time_window=(-0.2, 1.2),
    freq_window=(1, 30),
    errorbar="se",
    figsize=(12, 8),
):
    """
    Four-panel figure:
        upper left  = ERP easy
        upper right = ERP hard
        lower left  = PSD easy
        lower right = PSD hard

    electrode can be:
        str       -> one electrode, e.g. "FCz"
        list[str] -> average over electrodes, e.g. cluster channels
    """

    if isinstance(electrode, str):
        electrodes = [electrode]
        elec_label = electrode
    else:
        electrodes = list(electrode)
        elec_label = f"{len(electrodes)}-electrode average"

    erp_use = erp_df[erp_df["electrode"].isin(electrodes)].copy()
    psd_use = psd_df[psd_df["electrode"].isin(electrodes)].copy()

    if erp_use.empty:
        raise ValueError("No ERP data found for requested electrode(s).")
    if psd_use.empty:
        raise ValueError("No PSD data found for requested electrode(s).")

    # Average electrodes first within subject-condition-time/freq
    erp_use = (
        erp_use
        .groupby(
            ["subject", "condition", "difficulty", "flip", erp_time_col],
            observed=True,
            as_index=False,
        )[erp_value_col]
        .mean()
    )

    psd_use = (
        psd_use
        .groupby(
            ["subject", "condition", "difficulty", "flip", psd_freq_col],
            observed=True,
            as_index=False,
        )[psd_value_col]
        .mean()
    )

    erp_use = erp_use[
        (erp_use[erp_time_col] >= time_window[0])
        & (erp_use[erp_time_col] <= time_window[1])
        & (erp_use["flip"].astype(str).isin(conditions))
    ].copy()

    psd_use = psd_use[
        (psd_use[psd_freq_col] >= freq_window[0])
        & (psd_use[psd_freq_col] <= freq_window[1])
        & (psd_use["flip"].astype(str).isin(conditions))
    ].copy()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex=False,
        constrained_layout=True,
    )

    for col, difficulty in enumerate(difficulties):

        erp_plot = erp_use[erp_use["difficulty"].astype(str) == difficulty]
        psd_plot = psd_use[psd_use["difficulty"].astype(str) == difficulty]

        ax_erp = axes[0, col]
        ax_psd = axes[1, col]

        sns.lineplot(
            data=erp_plot,
            x=erp_time_col,
            y=erp_value_col,
            hue="flip",
            hue_order=list(conditions),
            estimator="mean",
            errorbar=errorbar,
            ax=ax_erp,
        )

        ax_erp.axvline(0, color="k", linewidth=0.8, linestyle="--")
        ax_erp.axhline(0, color="k", linewidth=0.8)
        ax_erp.set_title(f"ERP | {difficulty}")
        ax_erp.set_xlabel("Time from cue (s)")
        ax_erp.set_ylabel("Amplitude (µV)")

        sns.lineplot(
            data=psd_plot,
            x=psd_freq_col,
            y=psd_value_col,
            hue="flip",
            hue_order=list(conditions),
            estimator="mean",
            errorbar=errorbar,
            ax=ax_psd,
        )

        ax_psd.set_title(f"PSD | {difficulty}")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("log10 power")

        # Keep legend only on right panels
        if col == 0:
            if ax_erp.get_legend() is not None:
                ax_erp.get_legend().remove()
            if ax_psd.get_legend() is not None:
                ax_psd.get_legend().remove()
        else:
            if ax_erp.get_legend() is not None:
                ax_erp.get_legend().set_title("Flip")
            if ax_psd.get_legend() is not None:
                ax_psd.get_legend().set_title("Flip")

    fig.suptitle(f"ERP and PSD | {elec_label}", fontsize=16)

    return fig

# Loop datasets
for dataset in datasets:

    # Get id
    subject_id = int(dataset.split("/")[-1].split("_")[0].split("VP")[1])

    # Skip VP 07 (age outlier)
    if subject_id == 7:
        continue

    # Load a dataset
    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(-0.2, 0))
        .crop(tmin=-0.2, tmax=1.8)
    )

    # Save times
    erp_times = eeg_epochs.times

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("_cleaned_")[0] + "_erp_trialinfo.csv")

    # Get trial indices
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

    # Get evokeds
    evoked_easy_00 = eeg_epochs[idx_easy_00].average()
    evoked_easy_10 = eeg_epochs[idx_easy_10].average()
    evoked_easy_11 = eeg_epochs[idx_easy_11].average()
    evoked_hard_00 = eeg_epochs[idx_hard_00].average()
    evoked_hard_10 = eeg_epochs[idx_hard_10].average()
    evoked_hard_11 = eeg_epochs[idx_hard_11].average()
    
    conditions = {
        "easy_00": idx_easy_00,
        "easy_10": idx_easy_10,
        "easy_11": idx_easy_11,
        "hard_00": idx_hard_00,
        "hard_10": idx_hard_10,
        "hard_11": idx_hard_11,
    }
    
    # Crop once for PSD
    epochs_psd = eeg_epochs.copy().crop(tmin=tmin_psd, tmax=tmax_psd)

    # Crop once for slow-wave / CNV voltage
    epochs_slow = eeg_epochs.copy().crop(tmin=tmin_slow, tmax=tmax_slow)
    
    sfreq = epochs_psd.info["sfreq"]
    ch_names = epochs_psd.ch_names
    
    for cond_name, idx in conditions.items():
    
        if len(idx) == 0:
            continue
        
        # Calculate slow wave average for trials
        evoked = eeg_epochs[idx].average()
        slow_wave_uv = (
            evoked.copy()
            .crop(tmin=tmin_slow, tmax=tmax_slow)
            .data
            .mean(axis=1)
            * 1e6
        )
        
        # Save ERP: subject × condition × electrode × time
        erp_data_uv = evoked.data * 1e6  # channels × times
        times = evoked.times
        
        erp_df = pd.DataFrame(
            erp_data_uv,
            index=ch_names,
            columns=times,
        )
        
        erp_df = (
            erp_df
            .reset_index(names="electrode")
            .melt(
                id_vars="electrode",
                var_name="time",
                value_name="erp_uv",
            )
        )
        
        erp_df["subject"] = subject_id
        erp_df["condition"] = cond_name
        erp_df["n_trials"] = len(idx)
        
        all_erp_rows.append(erp_df)
        
        # Calculate fooof parameters
        data = epochs_psd[idx].get_data()       # trials × channels × times
            
        psds, freqs = compute_psd(
            data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            method="multitaper",
            bandwidth=1.0,
        )
    
        # Average PSD over trials: channels × freqs
        psd_mean = psds.mean(axis=0)
        
        # Save PSD: subject × condition × electrode × frequency
        psd_df = pd.DataFrame(
            psd_mean,
            index=ch_names,
            columns=freqs,
        )
        
        psd_df = (
            psd_df
            .reset_index(names="electrode")
            .melt(
                id_vars="electrode",
                var_name="freq",
                value_name="psd",
            )
        )
        
        psd_df["log10_psd"] = np.log10(psd_df["psd"])
        psd_df["subject"] = subject_id
        psd_df["condition"] = cond_name
        psd_df["n_trials"] = len(idx)
        
        all_psd_rows.append(psd_df)

    
        for ch_idx, ch_name in enumerate(ch_names):
    
            spectrum = psd_mean[ch_idx, :]
    
            # FOOOF expects linear power, not log power
            fm = FOOOF(**fooof_settings)
            fm.fit(freqs, spectrum, [fmin, fmax])
    
            offset, exponent = fm.aperiodic_params_
    
            # Flatten spectrum manually in log10 space:
            # log10(power) - log10(aperiodic fit)
            log_power = np.log10(spectrum)
            ap_fit = offset - exponent * np.log10(freqs)
            flat_power = log_power - ap_fit
    
            row = {
                "subject": subject_id,
                "condition": cond_name,
                "electrode": ch_name,
                "n_trials": len(idx),
                "fooof_offset": offset,
                "fooof_exponent": exponent,
                "fooof_r_squared": fm.r_squared_,
                "fooof_error": fm.error_,
                "fooof_n_peaks": fm.n_peaks_,
                "slow_wave_uv": slow_wave_uv[ch_idx],
            }
    
            for band_name, (lo, hi) in bands.items():
                band_mask = (freqs >= lo) & (freqs <= hi)
    
                if band_mask.sum() == 0:
                    row[f"{band_name}_flat_power"] = np.nan
                else:
                    row[f"{band_name}_flat_power"] = flat_power[band_mask].mean()
    
            all_fooof_rows.append(row)

fooof_df = pd.DataFrame(all_fooof_rows)
erp_df = pd.concat(all_erp_rows, ignore_index=True)
psd_df = pd.concat(all_psd_rows, ignore_index=True)

for df in [fooof_df, erp_df, psd_df]:
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

# State effect: flip possible vs impossible, excluding post-flip trials
df_state = fooof_df[fooof_df["flip"].isin(["00", "10"])].copy()

# Sequence effect: just flipped vs not flipped, only within flip-possible blocks
df_sequence = fooof_df[fooof_df["flip"].isin(["10", "11"])].copy()


# Plot interactions at electrode
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


# get info object
info = eeg_epochs.info.copy()

# =============================================================================================



# State contrast: 10 - 00 for slow wave
res_state_cnv = run_topo_cluster_test(
    df=df_state,
    measure="slow_wave_uv",
    info=info,
    contrast_type="flip",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_cnv)
plot_cluster_topomap(res_state_cnv)

# State contrast: 10 - 00 for exponent
res_state_expo = run_topo_cluster_test(
    df=df_state,
    measure="fooof_exponent",
    info=info,
    contrast_type="flip",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_expo)
plot_cluster_topomap(res_state_expo)

# State contrast: 10 - 00 for delta
res_state_delta = run_topo_cluster_test(
    df=df_state,
    measure="delta_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_delta)
plot_cluster_topomap(res_state_delta)

# State contrast: 10 - 00 for theta
res_state_theta = run_topo_cluster_test(
    df=df_state,
    measure="theta_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_theta)
plot_cluster_topomap(res_state_theta)

# State contrast: 10 - 00 for alpha
res_state_alpha = run_topo_cluster_test(
    df=df_state,
    measure="alpha_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_alpha)
plot_cluster_topomap(res_state_alpha)

# State contrast: 10 - 00 for beta
res_state_beta = run_topo_cluster_test(
    df=df_state,
    measure="beta_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_beta)
plot_cluster_topomap(res_state_beta)

# =============================================================================================

# sequence contrast: 10 - 00 for slow wave
res_sequence_cnv = run_topo_cluster_test(
    df=df_sequence,
    measure="slow_wave_uv",
    info=info,
    contrast_type="flip",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_cnv)
plot_cluster_topomap(res_sequence_cnv)

# sequence contrast: 10 - 00 for exponent
res_sequence_expo = run_topo_cluster_test(
    df=df_sequence,
    measure="fooof_exponent",
    info=info,
    contrast_type="flip",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_expo)
plot_cluster_topomap(res_sequence_expo)

# sequence contrast: 10 - 00 for delta
res_sequence_delta = run_topo_cluster_test(
    df=df_sequence,
    measure="delta_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_delta)
plot_cluster_topomap(res_sequence_delta)

# sequence contrast: 10 - 00 for theta
res_sequence_theta = run_topo_cluster_test(
    df=df_sequence,
    measure="theta_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_theta)
plot_cluster_topomap(res_sequence_theta)

# sequence contrast: 10 - 00 for alpha
res_sequence_alpha = run_topo_cluster_test(
    df=df_sequence,
    measure="alpha_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_alpha)
plot_cluster_topomap(res_sequence_alpha)

# sequence contrast: 10 - 00 for beta
res_sequence_beta = run_topo_cluster_test(
    df=df_sequence,
    measure="beta_flat_power",
    info=info,
    contrast_type="flip",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_beta)
plot_cluster_topomap(res_sequence_beta)







# =============================================================================================



# State contrast: 10 - 00 for slow wave
res_state_cnv = run_topo_cluster_test(
    df=df_state,
    measure="slow_wave_uv",
    info=info,
    contrast_type="interaction",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_cnv)
plot_cluster_topomap(res_state_cnv)

# State contrast: 10 - 00 for exponent
res_state_expo = run_topo_cluster_test(
    df=df_state,
    measure="fooof_exponent",
    info=info,
    contrast_type="interaction",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_expo)
plot_cluster_topomap(res_state_expo)

# State contrast: 10 - 00 for delta
res_state_delta = run_topo_cluster_test(
    df=df_state,
    measure="delta_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_delta)
plot_cluster_topomap(res_state_delta)

# State contrast: 10 - 00 for theta
res_state_theta = run_topo_cluster_test(
    df=df_state,
    measure="theta_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_theta)
plot_cluster_topomap(res_state_theta)

# State contrast: 10 - 00 for alpha
res_state_alpha = run_topo_cluster_test(
    df=df_state,
    measure="alpha_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_alpha)
plot_cluster_topomap(res_state_alpha)

# State contrast: 10 - 00 for beta
res_state_beta = run_topo_cluster_test(
    df=df_state,
    measure="beta_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["00", "10"],
)

print_cluster_results(res_state_beta)
plot_cluster_topomap(res_state_beta)

# =============================================================================================

# sequence contrast: 10 - 00 for slow wave
res_sequence_cnv = run_topo_cluster_test(
    df=df_sequence,
    measure="slow_wave_uv",
    info=info,
    contrast_type="interaction",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_cnv)
plot_cluster_topomap(res_sequence_cnv)

# sequence contrast: 10 - 00 for exponent
res_sequence_expo = run_topo_cluster_test(
    df=df_sequence,
    measure="fooof_exponent",
    info=info,
    contrast_type="interaction",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_expo)
plot_cluster_topomap(res_sequence_expo)

# sequence contrast: 10 - 00 for delta
res_sequence_delta = run_topo_cluster_test(
    df=df_sequence,
    measure="delta_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_delta)
plot_cluster_topomap(res_sequence_delta)

# sequence contrast: 10 - 00 for theta
res_sequence_theta = run_topo_cluster_test(
    df=df_sequence,
    measure="theta_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_theta)
plot_cluster_topomap(res_sequence_theta)

# sequence contrast: 10 - 00 for alpha
res_sequence_alpha = run_topo_cluster_test(
    df=df_sequence,
    measure="alpha_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_alpha)
plot_cluster_topomap(res_sequence_alpha)

# sequence contrast: 10 - 00 for beta
res_sequence_beta = run_topo_cluster_test(
    df=df_sequence,
    measure="beta_flat_power",
    info=info,
    contrast_type="interaction",
    flip_levels=["10", "11"],
)

print_cluster_results(res_sequence_beta)
plot_cluster_topomap(res_sequence_beta)

fig = plot_erp_psd_4panel(
    erp_df=erp_df,
    psd_df=psd_df,
    electrode="FCz",
    conditions=("00", "10"),
)

plt.show()