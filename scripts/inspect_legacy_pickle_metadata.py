"""Print pickle metadata only. Never print row values or serialize the object."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd


def inspect_pickle(path: Path) -> None:
    frame = pd.read_pickle(path, compression="gzip")
    print(f"path_name={path.name}")
    print(f"python_type={type(frame).__name__}")
    if not isinstance(frame, pd.DataFrame):
        print("not_a_dataframe=True")
        del frame
        gc.collect()
        return
    print(f"row_count={len(frame)}")
    print(f"column_count={len(frame.columns)}")
    print(f"index_names={list(frame.index.names)}")
    print(f"columns={list(frame.columns)}")
    print(
        "dtypes="
        + ",".join(f"{name}:{dtype}" for name, dtype in frame.dtypes.items())
    )
    if "Trial type" in frame.columns:
        counts = frame["Trial type"].astype(str).value_counts()
        print(
            "trial_type_counts="
            + ",".join(f"{key}:{int(value)}" for key, value in counts.items())
        )
    if "Trial number" in frame.columns:
        print(f"unique_trial_numbers={int(frame['Trial number'].nunique())}")
    if "Block name" in frame.columns:
        print(
            "unique_block_names="
            + ",".join(sorted(frame["Block name"].astype(str).unique()))
        )
    memory_mb = float(frame.memory_usage(deep=True).sum()) / (1024 * 1024)
    print(f"approx_memory_mb={memory_mb:.1f}")
    del frame
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_path", type=Path)
    args = parser.parse_args()
    inspect_pickle(args.pickle_path.resolve())


if __name__ == "__main__":
    main()
