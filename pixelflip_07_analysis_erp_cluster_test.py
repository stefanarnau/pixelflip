# Imports
import mne
import glob
import pandas as pd
import numpy as np
import mne.stats 
import mne
from scipy import stats
from mne.stats import permutation_cluster_1samp_test, combine_adjacency
from mne.channels import find_ch_adjacency
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Bins
all_erp_rows = []

# Define paths
path_in = "/mnt/data_dump/pixelflip/2_cleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*cue_erp.set")

# Helpers ========================================================================================

def run_erp_cluster_test(
    df,
    effect,
    times,
    info,
    tmin,
    tmax,
    n_permutations=5000,
    threshold=None,
    tail=0,
    seed=42,
):
    """
    Cluster test for 2 x 2 within-subject ERP effects in time x channel space.

    Parameters
    ----------
    df : DataFrame
        Either df_state or df_sequence.
        Must contain: subject, condition, difficulty, flip, slow_wave_uv.
        slow_wave_uv must be array-shaped: n_channels x n_times.
    effect : {"difficulty", "flip", "interaction"}
        Effect to test.
    times : array
        ERP time vector.
    info : mne.Info
        MNE info object containing channel locations.
    tmin, tmax : float
        Time window included in cluster test.
    """

    assert effect in ["difficulty", "flip", "interaction"]

    times = np.asarray(times)
    time_mask = (times >= tmin) & (times <= tmax)
    times_sel = times[time_mask]

    flip_levels = list(df["flip"].cat.categories[df["flip"].cat.categories.isin(df["flip"].unique())])
    diff_levels = ["easy", "hard"]

    if len(flip_levels) != 2:
        raise ValueError(
            f"Cluster test requires exactly 2 flip levels, got: {flip_levels}"
        )

    flip_a, flip_b = flip_levels[0], flip_levels[1]

    subject_contrasts = []
    used_subjects = []

    for subject, sub_df in df.groupby("subject"):

        cells = {}

        for diff in diff_levels:
            for flip in [flip_a, flip_b]:
                row = sub_df[
                    (sub_df["difficulty"] == diff)
                    & (sub_df["flip"] == flip)
                ]

                if len(row) != 1:
                    continue

                arr = row.iloc[0]["slow_wave_uv"]  # channels x times
                arr = arr[:, time_mask].T         # times x channels
                cells[(diff, flip)] = arr

        if len(cells) != 4:
            continue

        easy_a = cells[("easy", flip_a)]
        easy_b = cells[("easy", flip_b)]
        hard_a = cells[("hard", flip_a)]
        hard_b = cells[("hard", flip_b)]

        if effect == "difficulty":
            contrast = ((hard_a + hard_b) / 2) - ((easy_a + easy_b) / 2)

        elif effect == "flip":
            contrast = ((easy_b + hard_b) / 2) - ((easy_a + hard_a) / 2)

        elif effect == "interaction":
            contrast = (hard_b - hard_a) - (easy_b - easy_a)

        subject_contrasts.append(contrast)
        used_subjects.append(subject)

    X = np.stack(subject_contrasts, axis=0)  # subjects x times x channels

    ch_adjacency, ch_names = find_ch_adjacency(info, ch_type="eeg")
    adjacency = combine_adjacency(len(times_sel), ch_adjacency)

    # Cluster-forming threshold
    df_t = X.shape[0] - 1
    
    if threshold is None:
        threshold = 0.05
    
    if tail == 0:
        threshold_t = stats.t.ppf(1 - threshold / 2, df_t)
    elif tail == 1:
        threshold_t = stats.t.ppf(1 - threshold, df_t)
    elif tail == -1:
        threshold_t = stats.t.ppf(threshold, df_t)
    else:
        raise ValueError("tail must be -1, 0, or 1")

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X,
        threshold=threshold_t,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        seed=seed,
        n_jobs=-1,
    )

    return {
        "effect": effect,
        "X": X,
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "H0": H0,
        "times": times_sel,
        "full_times": times,
        "time_mask": time_mask,
        "subjects": used_subjects,
        "ch_names": info["ch_names"],
    }

def run_global_difficulty_cluster_test(
    erp_df,
    times,
    info,
    tmin,
    tmax,
    threshold=0.05,
    tail=0,
    n_permutations=5000,
    seed=42,
):

    times = np.asarray(times)
    time_mask = (times >= tmin) & (times <= tmax)
    times_sel = times[time_mask]

    subject_contrasts = []
    used_subjects = []

    for subject, sub_df in erp_df.groupby("subject"):

        easy_rows = sub_df[sub_df["difficulty"] == "easy"]
        hard_rows = sub_df[sub_df["difficulty"] == "hard"]

        # Require all three cells
        if len(easy_rows) != 3 or len(hard_rows) != 3:
            continue

        easy = np.mean(
            np.stack(easy_rows["slow_wave_uv"].values, axis=0),
            axis=0,
        )

        hard = np.mean(
            np.stack(hard_rows["slow_wave_uv"].values, axis=0),
            axis=0,
        )

        contrast = hard - easy

        # times x channels
        contrast = contrast[:, time_mask].T

        subject_contrasts.append(contrast)
        used_subjects.append(subject)

    X = np.stack(subject_contrasts, axis=0)

    ch_adjacency, _ = find_ch_adjacency(info, ch_type="eeg")
    adjacency = combine_adjacency(len(times_sel), ch_adjacency)

    df_t = X.shape[0] - 1

    if tail == 0:
        threshold_t = stats.t.ppf(
            1 - threshold / 2,
            df_t,
        )
    elif tail == 1:
        threshold_t = stats.t.ppf(
            1 - threshold,
            df_t,
        )
    else:
        threshold_t = stats.t.ppf(
            threshold,
            df_t,
        )

    T_obs, clusters, cluster_p_values, H0 = (
        permutation_cluster_1samp_test(
            X,
            threshold=threshold_t,
            adjacency=adjacency,
            tail=tail,
            n_permutations=n_permutations,
            out_type="mask",
            seed=seed,
            n_jobs=-1,
        )
    )

    return {
        "effect": "difficulty",
        "X": X,
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "H0": H0,
        "times": times_sel,
        "full_times": times,
        "time_mask": time_mask,
        "subjects": used_subjects,
        "ch_names": info["ch_names"],
    }

def plot_erp_cluster(
    res,
    df,
    info,
    cluster_idx=0,
    alpha=0.05,
    figsize=(10, 7),
    cmap="RdBu_r",
    line_width=2.5,
    target_onset=1.2,
    colorbar=True,
    fig=None,
    outer_spec=None,
    panel_title=None,
    show_suptitle=True,
    show_erp_title=True,
    line_colors=None,
    topo_vmax=None,
    erp_ylim=None,
    show_legend=False,
):

    effect = res["effect"]
    times = res["times"]
    time_mask = res["time_mask"]
    ch_names = np.array(res["ch_names"])

    cluster_p = res["cluster_p_values"][cluster_idx]
    cluster_mask = res["clusters"][cluster_idx]

    if cluster_p >= alpha:
        print(
            f"Warning: selected cluster is not significant, "
            f"p = {cluster_p:.4f}"
        )

    t_inds, ch_inds = np.where(cluster_mask)
    cluster_times = times[np.unique(t_inds)]
    cluster_chs = np.unique(ch_inds)

    tmin_clu = cluster_times.min()
    tmax_clu = cluster_times.max()

    flip_levels = list(
        df["flip"].cat.categories[
            df["flip"].cat.categories.isin(df["flip"].unique())
        ]
    )

    all_a = []
    all_b = []

    for subject, sub_df in df.groupby("subject"):

        cells = {}

        for _, row in sub_df.iterrows():
            cells[(row["difficulty"], row["flip"])] = row["slow_wave_uv"]

        if effect == "difficulty":

            easy_keys = [k for k in cells if k[0] == "easy"]
            hard_keys = [k for k in cells if k[0] == "hard"]

            if len(easy_keys) == 0 or len(hard_keys) == 0:
                continue

            arr_a = np.mean(
                np.stack([cells[k] for k in easy_keys], axis=0),
                axis=0,
            )

            arr_b = np.mean(
                np.stack([cells[k] for k in hard_keys], axis=0),
                axis=0,
            )

            cond_a_label = "easy"
            cond_b_label = "hard"

        elif effect == "flip":

            flip_a, flip_b = flip_levels

            needed = [
                ("easy", flip_a),
                ("easy", flip_b),
                ("hard", flip_a),
                ("hard", flip_b),
            ]

            if not all(k in cells for k in needed):
                continue

            arr_a = (
                cells[("easy", flip_a)]
                + cells[("hard", flip_a)]
            ) / 2

            arr_b = (
                cells[("easy", flip_b)]
                + cells[("hard", flip_b)]
            ) / 2

            cond_a_label = str(flip_a)
            cond_b_label = str(flip_b)

        elif effect == "interaction":

            flip_a, flip_b = flip_levels

            needed = [
                ("easy", flip_a),
                ("easy", flip_b),
                ("hard", flip_a),
                ("hard", flip_b),
            ]

            if not all(k in cells for k in needed):
                continue

            arr_a = (
                cells[("easy", flip_b)]
                - cells[("easy", flip_a)]
            )

            arr_b = (
                cells[("hard", flip_b)]
                - cells[("hard", flip_a)]
            )

            cond_a_label = f"Easy: {flip_b} - {flip_a}"
            cond_b_label = f"Hard: {flip_b} - {flip_a}"

        else:
            raise ValueError("Unknown effect")

        all_a.append(arr_a[:, time_mask])
        all_b.append(arr_b[:, time_mask])

    all_a = np.stack(all_a, axis=0)
    all_b = np.stack(all_b, axis=0)

    time_mask_cluster = (
        (times >= tmin_clu)
        & (times <= tmax_clu)
    )

    topo_a = all_a[:, :, time_mask_cluster].mean(axis=(0, 2))
    topo_b = all_b[:, :, time_mask_cluster].mean(axis=(0, 2))

    erp_a_sub = all_a[:, cluster_chs, :].mean(axis=1)
    erp_b_sub = all_b[:, cluster_chs, :].mean(axis=1)

    erp_a = erp_a_sub.mean(axis=0)
    erp_b = erp_b_sub.mean(axis=0)

    sem_a = (
        erp_a_sub.std(axis=0, ddof=1)
        / np.sqrt(erp_a_sub.shape[0])
    )

    sem_b = (
        erp_b_sub.std(axis=0, ddof=1)
        / np.sqrt(erp_b_sub.shape[0])
    )

    topo_mask = np.zeros(len(ch_names), dtype=bool)
    topo_mask[cluster_chs] = True

    if line_colors is None:
        color_a = "#1b9e77"
        color_b = "#d81b60"
    else:
        color_a, color_b = line_colors

    if topo_vmax is None:
        vmax = np.max(np.abs([topo_a, topo_b]))
    else:
        vmax = topo_vmax
    
    vmax = vmax if vmax > 0 else 1e-12

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    standalone = fig is None or outer_spec is None

    if standalone:
        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[1, 1, 0.05],
            height_ratios=[1, 1.15],
            wspace=0.15,
            hspace=0.20,
        )

    else:
        gs = gridspec.GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=outer_spec,
            width_ratios=[1, 1, 0.06],
            height_ratios=[1, 1.15],
            wspace=0.12,
            hspace=0.22,
        )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])

    # --------------------------------------------------
    # Topographies
    # --------------------------------------------------

    mask_params = dict(
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        markeredgewidth=1.5,
    )

    im1, _ = mne.viz.plot_topomap(
        topo_a,
        info,
        axes=ax1,
        show=False,
        cmap=cmap,
        vlim=(-vmax, vmax),
        mask=topo_mask,
        mask_params=mask_params,
        contours=0,
    )

    ax1.set_title(cond_a_label, fontsize=16)

    im2, _ = mne.viz.plot_topomap(
        topo_b,
        info,
        axes=ax2,
        show=False,
        cmap=cmap,
        vlim=(-vmax, vmax),
        mask=topo_mask,
        mask_params=mask_params,
        contours=0,
    )

    ax2.set_title(cond_b_label, fontsize=16)

    if colorbar:
        cbar = fig.colorbar(
            im2,
            cax=cax,
        )

        cbar.set_label(
            "Amplitude (µV)",
            fontsize=12,
        )

    else:
        cax.set_visible(False)

    # --------------------------------------------------
    # ERP
    # --------------------------------------------------

    ax3.plot(
        times,
        erp_a,
        color=color_a,
        linewidth=line_width,
        label=cond_a_label,
    )

    ax3.fill_between(
        times,
        erp_a - sem_a,
        erp_a + sem_a,
        color=color_a,
        alpha=0.20,
    )

    ax3.plot(
        times,
        erp_b,
        color=color_b,
        linewidth=line_width,
        label=cond_b_label,
    )

    ax3.fill_between(
        times,
        erp_b - sem_b,
        erp_b + sem_b,
        color=color_b,
        alpha=0.20,
    )

    ax3.axvspan(
        tmin_clu,
        tmax_clu,
        color="grey",
        alpha=0.18,
        linewidth=0,
    )

    if target_onset is not None:
        ax3.axvline(
            target_onset,
            color="black",
            linestyle=":",
            linewidth=2,
        )

    ax3.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude (µV)")
    
    if erp_ylim is not None:
        ax3.set_ylim(erp_ylim)

    if show_erp_title:
        ax3.set_title("ERP over cluster electrodes")

    if show_legend:
        ax3.legend(frameon=False)

    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    if panel_title is None:
        panel_title = f"{effect.capitalize()} effect"

    title_text = (
        f"{panel_title} "
        f"(p = {cluster_p:.3f})\n"
        f"{tmin_clu:.3f}–{tmax_clu:.3f} s"
    )

    if standalone and show_suptitle:
        fig.suptitle(title_text)
        fig.subplots_adjust(top=0.90)

    elif not standalone:
        # Put title above the two topomaps inside this panel
        ax1.text(
            1.1,
            1.25,
            title_text,
            transform=ax1.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
        )

    return fig

# ============================================================================================

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
    
    conditions = {
        "easy_00": idx_easy_00,
        "easy_10": idx_easy_10,
        "easy_11": idx_easy_11,
        "hard_00": idx_hard_00,
        "hard_10": idx_hard_10,
        "hard_11": idx_hard_11,
    }
    
    # Get fs
    sfreq = eeg_epochs.info["sfreq"]
    
    # Get channel labels
    ch_names = eeg_epochs.ch_names
    
    for cond_name, idx in conditions.items():
    
        if len(idx) == 0:
            continue
        
        # Calculate evoked for condition
        evoked = eeg_epochs[idx].average()
        
        # Get erp times
        erp_times = evoked.times
        
        # Append erp waveforms
        all_erp_rows.append({
            "subject": subject_id,
            "condition": cond_name,
            "n_trials": len(idx),
            "slow_wave_uv": evoked.data * 1e6,
        })
            
# Create erp df
erp_df = pd.DataFrame(all_erp_rows)

# Recode
erp_df["difficulty"] = erp_df["condition"].str.split("_").str[0]

erp_df["difficulty"] = pd.Categorical(
    erp_df["difficulty"],
    categories=["easy", "hard"],
    ordered=True,
)

erp_df["flip"] = (
    erp_df["condition"]
    .str.split("_")
    .str[1]
    .map({
        "00": "contingent",
        "10": "non-contingent",
        "11": "post-flip",
    })
)

erp_df["flip"] = pd.Categorical(
    erp_df["flip"],
    categories=["contingent", "non-contingent", "post-flip"],
    ordered=True,
)

df_state = erp_df[
    erp_df["flip"].isin(["contingent", "non-contingent"])
].copy()

df_sequence = erp_df[
    erp_df["flip"].isin(["non-contingent", "post-flip"])
].copy()

res_global_difficulty = run_global_difficulty_cluster_test(
    erp_df=erp_df,
    times=erp_times,
    info=eeg_epochs.info,
    tmin=0,
    tmax=1.2,
)

res_state_flip = run_erp_cluster_test(
    df=df_state,
    effect="flip",
    times=erp_times,
    info=eeg_epochs.info,
    tmin=0,
    tmax=1.2,
)

res_sequence_flip = run_erp_cluster_test(
    df=df_sequence,
    effect="flip",
    times=erp_times,
    info=eeg_epochs.info,
    tmin=0,
    tmax=1.2,
)

res_state_interaction = run_erp_cluster_test(
    df=df_state,
    effect="interaction",
    times=erp_times,
    info=eeg_epochs.info,
    tmin=0,
    tmax=1.2,
)

res_sequence_interaction = run_erp_cluster_test(
    df=df_sequence,
    effect="interaction",
    times=erp_times,
    info=eeg_epochs.info,
    tmin=0,
    tmax=1.2,
)


# Collect results
results = {
    "Difficulty": (res_global_difficulty, erp_df),
    "State": (res_state_flip, df_state),
    "Sequence": (res_sequence_flip, df_sequence),
    "Difficulty × State": (res_state_interaction, df_state),
    "Difficulty × Sequence": (res_sequence_interaction, df_sequence),
}

alpha = 0.05

best_diff_cluster = np.where(
    res_global_difficulty["cluster_p_values"] < alpha
)[0][0]

best_state_cluster = np.where(
    res_state_flip["cluster_p_values"] < alpha
)[0][0]

# Use one common topomap scale
shared_topo_vmax = 2.5

fig = plt.figure(figsize=(16, 7))

outer = gridspec.GridSpec(
    1,
    2,
    figure=fig,
    wspace=0.20,
)

plot_erp_cluster(
    res=res_global_difficulty,
    df=erp_df,
    info=eeg_epochs.info,
    cluster_idx=best_diff_cluster,
    fig=fig,
    outer_spec=outer[0],
    panel_title="Difficulty effect",
    colorbar=False,
    show_erp_title=False,
    show_legend=True,
    topo_vmax=shared_topo_vmax,
    erp_ylim=(-1.7, 0.4),
)

plot_erp_cluster(
    res=res_state_flip,
    df=df_state,
    info=eeg_epochs.info,
    cluster_idx=best_state_cluster,
    fig=fig,
    outer_spec=outer[1],
    panel_title="Contingency effect",
    colorbar=True,
    show_erp_title=False,
    show_legend=True,
    topo_vmax=shared_topo_vmax,
    erp_ylim=(-1.7, 0.4),
)

plt.savefig(
    "erp_difficulty_contingency.pdf",
    bbox_inches="tight",
)

plt.show()