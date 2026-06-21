"""
Split helper dedup-aware — grouping berbasis hash supaya near-duplicate
tidak terpisah ke split berbeda (train/val/test).
"""
import hashlib
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SEED
from utils.preprocessing import sequence_to_texts

def _rebuild_tool_calls_text(tool_calls_json: str) -> str:
    try:
        seq = json.loads(tool_calls_json)
    except Exception:
        return str(tool_calls_json)
    return " ".join(sequence_to_texts(seq))

def make_group_key(df: pd.DataFrame) -> pd.Series:
    tc_text = df["tool_calls_json"].astype(str).apply(_rebuild_tool_calls_text)
    combined = (
        tc_text + "|"
        + df["context_text"].astype(str) + "|"
        + df["label"].astype(str)
    )
    return combined.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())


# Split train/val/test berbasis group, satu group tidak terpecah
def group_aware_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
):
    groups = make_group_key(df)

    splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    train_idx, temp_idx = next(splitter1.split(df, groups=groups))

    temp_df = df.iloc[temp_idx]
    temp_groups = groups.iloc[temp_idx]
    rel_test_size = test_size / (test_size + val_size)

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=rel_test_size, random_state=seed)
    val_idx, test_idx = next(splitter2.split(temp_df, groups=temp_groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df  = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df
