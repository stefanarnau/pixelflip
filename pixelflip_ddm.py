# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import os

# Paths
path_in = "/mnt/data_dump/pixelflip/6_behavioral_results/"

# Read data
fn = os.path.join(path_in, "pixflip_ddm_table.csv")
df = pd.read_csv(fn)

# Drop no-response trials
df = df[df['key_pressed'] != 0]