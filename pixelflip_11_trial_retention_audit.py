"""PixelFlip trial-count audit using the condition and selection logic of the RT script.

No EEG data are loaded. The list of *cue_erp.set files identifies participants;
corresponding *_erp_trialinfo.csv files supply all trial information.

Outputs (in OUTPUT_DIR): participant_condition_counts.csv,
participant_condition_difficulty_counts.csv, condition_count_summary.csv,
condition_assignment_audit.csv, trial_count_distribution.png.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH_IN = Path('/mnt/data_dump/pixelflip/2_cleaned/')
OUTPUT_DIR = Path(__file__).resolve().parent / 'trial_retention_audit'
EXCLUDED_SUBJECTS = {7}
CONDITIONS = ['contingent', 'non-contingent', 'post-flip']
DIFFICULTIES = ['easy', 'hard']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_rows = []
files = sorted(glob.glob(str(PATH_IN / '*cue_erp.set')))
if not files:
    raise FileNotFoundError(f'No *cue_erp.set files found in {PATH_IN}')

for dataset_str in files:
    dataset = Path(dataset_str)
    subject_id = int(dataset.name.split('_')[0].split('VP')[1])
    if subject_id in EXCLUDED_SUBJECTS:
        continue

    # Exactly the metadata path construction in the supplied RT script.
    trialinfo_path = Path(dataset_str.split('_cleaned_')[0] + '_erp_trialinfo.csv')
    if not trialinfo_path.exists():
        raise FileNotFoundError(f'Missing trial information for VP{subject_id}: {trialinfo_path}')
    trialinfo = pd.read_csv(trialinfo_path).copy()
    required = ['difficulty', 'reliability', 'prev_flipped', 'prev_accuracy', 'accuracy', 'rt']
    missing = [c for c in required if c not in trialinfo]
    if missing:
        raise KeyError(f'VP{subject_id}: missing columns {missing}')

    trialinfo['subject'] = subject_id
    trialinfo['difficulty_label'] = np.where(trialinfo['difficulty'] == 0, 'easy', 'hard')

    # Exactly the three condition definitions from the supplied RT script.
    # Use object dtype to permit strings and missing values reliably.
    trialinfo['condition'] = pd.Series(pd.NA, index=trialinfo.index, dtype='object')
    trialinfo.loc[trialinfo['reliability'] == 1, 'condition'] = 'contingent'
    trialinfo.loc[
        (trialinfo['reliability'] == 0) & (trialinfo['prev_flipped'] == 0),
        'condition',
    ] = 'non-contingent'
    trialinfo.loc[
        (trialinfo['reliability'] == 0) & (trialinfo['prev_flipped'] == 1),
        'condition',
    ] = 'post-flip'

    # RT script: keep prev_accuracy == 1 for the ERP/accuracy data frame;
    # then additionally require accuracy == 1 and nonmissing RT for RT.
    trialinfo['erp_selected'] = trialinfo['prev_accuracy'].eq(1)
    trialinfo['rt_selected'] = (
        trialinfo['erp_selected']
        & trialinfo['accuracy'].eq(1)
        & trialinfo['rt'].notna()
    )
    # Optional stricter finite-RT check for transparency, not used in RT counts.
    trialinfo['rt_finite'] = pd.to_numeric(trialinfo['rt'], errors='coerce').apply(np.isfinite)
    trialinfo['rt_finite_selected'] = (
        trialinfo['erp_selected'] & trialinfo['accuracy'].eq(1) & trialinfo['rt_finite']
    )
    all_rows.append(trialinfo)

trials = pd.concat(all_rows, ignore_index=True)
subjects = sorted(trials['subject'].unique())

# Audit unclassified trials; do not silently omit them from the total.
audit = (
    trials.assign(assignment=np.where(trials['condition'].isna(), 'unclassified', 'classified'))
    .groupby(['subject', 'assignment'], observed=True).size()
    .unstack(fill_value=0).reindex(columns=['classified', 'unclassified'], fill_value=0)
)
audit['total_cleaned_epochs'] = audit.sum(axis=1)
audit.to_csv(OUTPUT_DIR / 'condition_assignment_audit.csv')

classified = trials[trials['condition'].notna()].copy()

# Include zero-count subject-condition combinations explicitly.
idx = pd.MultiIndex.from_product([subjects, CONDITIONS], names=['subject', 'condition'])
counts = (
    classified.groupby(['subject', 'condition'], observed=True)
    .agg(cleaned_eeg=('condition', 'size'),
         erp_selected=('erp_selected', 'sum'),
         rt_selected=('rt_selected', 'sum'),
         rt_finite_selected=('rt_finite_selected', 'sum'))
    .reindex(idx, fill_value=0)
    .reset_index()
)
counts['excluded_by_prev_accuracy'] = counts['cleaned_eeg'] - counts['erp_selected']
counts['excluded_from_rt_after_erp'] = counts['erp_selected'] - counts['rt_selected']
counts.to_csv(OUTPUT_DIR / 'participant_condition_counts.csv', index=False)

idx_diff = pd.MultiIndex.from_product(
    [subjects, CONDITIONS, DIFFICULTIES],
    names=['subject', 'condition', 'difficulty_label'],
)
counts_diff = (
    classified.groupby(['subject', 'condition', 'difficulty_label'], observed=True)
    .agg(cleaned_eeg=('condition', 'size'),
         erp_selected=('erp_selected', 'sum'),
         rt_selected=('rt_selected', 'sum'))
    .reindex(idx_diff, fill_value=0)
    .reset_index()
)
counts_diff.to_csv(OUTPUT_DIR / 'participant_condition_difficulty_counts.csv', index=False)

summary = (
    counts.groupby('condition', observed=True)[['cleaned_eeg', 'erp_selected', 'rt_selected']]
    .agg(['min', 'median', 'mean', 'max', 'sum'])
    .reindex(CONDITIONS)
)
summary.to_csv(OUTPUT_DIR / 'condition_count_summary.csv')

print(f'Participants: {len(subjects)}')
print(f'Cleaned EEG epochs (all metadata rows): {len(trials):,}')
print(f'Unclassified epochs: {int(trials["condition"].isna().sum()):,}')
print(f'ERP-selected (prev_accuracy == 1), classified: {int(classified["erp_selected"].sum()):,}')
print(f'RT-selected (ERP + accuracy == 1 + RT notna), classified: {int(classified["rt_selected"].sum()):,}')
print('\nCounts by condition (per participant):')
print(summary.round(1).to_string())
print('\nParticipants with fewest ERP-selected trials per condition:')
for condition in CONDITIONS:
    subset = counts[counts['condition'] == condition].nsmallest(5, 'erp_selected')
    print(f'\n{condition}:')
    print(subset[['subject', 'cleaned_eeg', 'erp_selected', 'rt_selected']].to_string(index=False))

# Visualization: each point is one participant, and boxes show participant distribution.
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharex=True)
metrics = [
    ('cleaned_eeg', 'Cleaned EEG epochs'),
    ('erp_selected', 'ERP-selected epochs\n(previous trial correct)'),
    ('rt_selected', 'RT-eligible trials\n(previous and current correct)'),
]
colors = ['#526577', '#b23b68', '#718f8d']
rng = np.random.default_rng(19)
for ax, (metric, title) in zip(axes, metrics):
    data = [counts.loc[counts['condition'] == c, metric].to_numpy() for c in CONDITIONS]
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.48,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': '#222222', 'linewidth': 1.7},
                    boxprops={'edgecolor': '#4b5563'},
                    whiskerprops={'color': '#4b5563'},
                    capprops={'color': '#4b5563'})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
    for j, values in enumerate(data):
        ax.scatter(j + rng.uniform(-0.14, 0.14, size=len(values)), values,
                   s=17, color=colors[j], alpha=0.65, linewidths=0, zorder=3)
    ax.set_xticks([0, 1, 2], ['Contingent', 'Non-contingent', 'Post-flip'])
    ax.tick_params(axis='x', labelrotation=20)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Trials per participant' if ax is axes[0] else '')
fig.suptitle(f'PixelFlip trial counts by condition (N = {len(subjects)})', fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'trial_count_distribution.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'\nSaved outputs to: {OUTPUT_DIR}')
