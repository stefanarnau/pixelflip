# PixelFlip

Analysis code for the **PixelFlip** study investigating sustained and transient consequences of disrupted response–outcome contingency on behavior, subjective task engagement, and preparatory neural activity.

The experiment used a cued perceptual discrimination task in which response–outcome contingency was manipulated blockwise. In non-contingent blocks, a subset of executed responses was replaced by response flips, allowing sustained effects of reduced contingency to be distinguished from the sequential consequences of individual contingency violations.

The repository contains the scripts used to generate the experimental control files, preprocess the EEG data, and perform the behavioral, subjective, and electrophysiological analyses reported in the manuscript.

## Repository structure

The numbered files approximately follow the order of the analysis pipeline.

### `pixelflip_01_create_control_files.py`

Generates the experimental control files used to define trial and block structure.

The script specifies:

* reliable and non-reliable blocks,
* the occurrence of response flips within non-reliable blocks,
* easy and hard perceptual discrimination trials,
* trial-specific stimulus parameters,
* block and trial event information.

The generated files are stored in `control_files/`.

### `pixelflip_02_preprocessing.m`

EEG preprocessing pipeline.

This script processes the raw EEG recordings and creates the cleaned, cue-locked EEG datasets used for the subsequent ERP analyses.

### `pixelflip_03_preprostats.m`

Calculates summary statistics and quality-control information from the EEG preprocessing pipeline.

### `pixelflip_04_write_erp_trialinfo.m`

Creates trial-level metadata corresponding to the preprocessed cue-locked EEG epochs.

These trial-information files provide the behavioral and experimental variables required to assign individual EEG epochs to task conditions in the subsequent Python analyses.

### `pixelflip_05_analysis_behavior.py`

Performs the behavioral analyses.

The script analyzes response time and accuracy as a function of:

* task difficulty,
* response–outcome contingency state, and
* whether a trial followed an individual response flip.

The analyses distinguish sustained behavioral consequences of reduced response–outcome contingency from the additional sequential effects following individual contingency violations.

### `pixelflip_06_analysis_subjective_ratings.py`

Analyzes subjective ratings of task engagement.

The script tests changes in self-reported measures including task focus and motivation between contingency conditions.

### `pixelflip_07_analysis_erp_cluster_test.py`

Performs the primary mass-univariate ERP analyses.

Cue-locked ERPs are compared across task conditions using spatiotemporal cluster-based permutation tests. The analyses test effects of:

* task difficulty,
* sustained response–outcome contingency,
* post-flip trials, and
* interactions with task difficulty.

These analyses identify the frontocentral preparatory ERP modulation subsequently characterized using the CNV region of interest.

### `pixelflip_08_analyses_post_hoc_cnv_models.py`

Performs follow-up analyses of the contingent negative variation (CNV).

Single-trial CNV amplitude is extracted from a predefined frontocentral ROI:

`FCz, Fz, Cz, FC1, FC2`

within the late cue–target interval:

`0.7–1.2 s`

The script examines:

* CNV amplitude across task difficulty and contingency conditions,
* the relationship between trial-to-trial CNV amplitude and subsequent response time,
* whether the CNV–RT relationship differs across contingency conditions, and
* whether the additional behavioral slowing following individual response flips persists after accounting for trial-to-trial variation in CNV amplitude.

The script also generates the manuscript figure showing condition-related CNV amplitudes and the trial-level relationship between CNV amplitude and response time.

### `environment.yml`

Conda environment specification for the Python analyses.

The file defines the Python version and package dependencies used to run the behavioral, subjective, and EEG analyses. The environment can be created from the repository root using:

```bash
conda env create -f environment.yml
```

and activated using:

```bash
conda activate pixelflip
```

### `standard-10-5-cap385.elp`

Electrode-location file used for EEG channel coordinates and topographic visualization.

### `control_files/`

Experimental control files defining the trial sequences presented to individual participants.

## Analysis logic

The principal EEG analysis proceeds in two stages.

First, `pixelflip_07_analysis_erp_cluster_test.py` uses spatiotemporal cluster-based permutation testing to characterize condition-related ERP differences without restricting the analysis to a predefined spatial ROI.

Second, `pixelflip_08_analyses_post_hoc_cnv_models.py` uses a predefined frontocentral CNV ROI to examine the behavioral relevance of the preparatory activity identified in the ERP analyses and to test whether the same preparatory process can account for both sustained contingency effects and the sequential consequences of individual response flips.

## Software

EEG preprocessing was performed in MATLAB using EEGLAB-related functionality. Statistical analyses and visualization were performed in Python using packages including MNE-Python, pandas, NumPy, SciPy, statsmodels, matplotlib, and seaborn.

The Python environment required for the analyses is specified in `environment.yml`.

Several scripts contain local input/output paths that must be adapted to the directory structure of the system on which the analyses are run.

## License

The analysis code in this repository is available under the MIT License.
