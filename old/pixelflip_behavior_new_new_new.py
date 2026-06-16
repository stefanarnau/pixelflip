#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import re

folder = Path("/mnt/data_dump/pixelflip/2_cleaned/")
csv_files = sorted(folder.glob("*.csv"))

def extract_subject_id(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if not match:
        raise ValueError(f"No numeric subject ID found in filename: {path.name}")
    return int(match.group())

df = pd.concat(
    (
        pd.read_csv(file)
          .assign(subject_id=extract_subject_id(file))
        for file in csv_files
    ),
    ignore_index=True
)

# Specify facors
df_seq_rt["f"] = df_seq_rt["mean_feedback"]
df_seq_rt["f2"] = df_seq_rt["f"] ** 2
df_seq_rt["half"] = np.where(df_seq_rt["block_nr"] <= 4, "first", "second")
df_seq_rt["half"] = df_seq_rt["half"].astype("category")


# Model formula
formula = """
mean_log_rt ~ group * f + group * f2
             + mean_trial_difficulty + half
"""

# Specify model
model = smf.mixedlm(
    formula,
    df_seq_rt,
    groups=df_seq_rt["id"],
    re_formula="1 + f + f2"
)

# Fit model
fitted_model = model.fit(method="lbfgs", reml=False, maxiter=4000, disp=False)

# Plot summary