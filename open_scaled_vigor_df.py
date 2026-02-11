"""
Standalone script to open scaled vigor DataFrame from log-median pickle files.

Loads pickles the same way as 4_ScaledVigorPlotting_LogMedian.py:
    df = pd.read_pickle(path)

Usage:
    python open_scaled_vigor_df.py              # Load first available pickle
    python open_scaled_vigor_df.py path/to.pkl  # Load specific file
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add the repository root to the Python path
if "__file__" in globals():
    module_root = Path(__file__).resolve().parent
else:
    module_root = Path.cwd()

import file_utils
from experiment_configuration import ExperimentType, get_experiment_config

# --- Parameters ---
CSUS = "CS"  # "CS" or "US"
INPUT_PKL_SUFFIX = "_new_logmedian"

# Experiment config

list_of_experiments = [
    ExperimentType.ALL_3S_TRACE.value,
    ExperimentType.ALL_DELAY.value,
    ExperimentType.ALL_10S_TRACE.value,
    ]

def load_df(path: Path) -> pd.DataFrame:
    """Load DataFrame from pickle (same as 4_ScaledVigorPlotting_LogMedian.py)."""
    return pd.read_pickle(path)

for EXPERIMENT in list_of_experiments:

    config = get_experiment_config(EXPERIMENT)
    paths = file_utils.create_folders(config.path_save)

    all_paths = sorted(
        Path(paths.all_fish).glob(f"*_{CSUS}{INPUT_PKL_SUFFIX}.pkl")
    )
    if not all_paths:
        print(f"No pickles found in {paths.all_fish} for {CSUS}.")

    for path in tqdm(all_paths, desc="Loading Data"):

        df = load_df(path)

        print(f"Loaded {path.name} unique 'Exp.': {df['Exp.'].unique()}")
        if df['Exp.'].nunique() > 1:
            if EXPERIMENT == ExperimentType.ALL_DELAY.value:
                if 'delay' in df['Exp.'].unique():
                    df['Exp.'] = 'delay'
                elif 'control' in df['Exp.'].unique():
                    df['Exp.'] = 'control'
            elif EXPERIMENT == ExperimentType.ALL_3S_TRACE.value:
                if 'trace' in df['Exp.'].unique():
                    df['Exp.'] = '3sTrace'
                elif 'control' in df['Exp.'].unique():
                    df['Exp.'] = 'control'
            elif EXPERIMENT == ExperimentType.ALL_10S_TRACE.value:
                if 'trace' in df['Exp.'].unique():
                    df['Exp.'] = '10sTrace'
                elif 'control' in df['Exp.'].unique():
                    df['Exp.'] = 'control'


        print(df['Exp.'].unique())
        # break

        df.to_pickle(path)

    #     break
    # break

