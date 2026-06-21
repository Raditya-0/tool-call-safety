"""
Generate split kanonik sekali — dipakai semua notebook model & ablasi
"""
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SEED
from data.split_utils import make_group_key


# Bangun mapping row_index ke split, simpan ke json
def generate_split_file(csv_path: str, out_path: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = SEED):
    df = pd.read_csv(csv_path)
    groups = make_group_key(df)

    splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    train_idx, temp_idx = next(splitter1.split(df, groups=groups))

    temp_groups = groups.iloc[temp_idx]
    rel_test_size = test_size / (test_size + val_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=rel_test_size, random_state=seed)
    val_rel_idx, test_rel_idx = next(splitter2.split(df.iloc[temp_idx], groups=temp_groups))

    val_idx  = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    split_map = {
        "seed": seed,
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_idx": sorted(int(i) for i in train_idx),
        "val_idx": sorted(int(i) for i in val_idx),
        "test_idx": sorted(int(i) for i in test_idx),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=1)

    return split_map


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base, "data", "tcssc_dataset.csv")
    out_path = os.path.join(base, "data", "splits", "tcssc_split.json")

    split_map = generate_split_file(csv_path, out_path)
    print(f"Split tersimpan: {out_path}")
    print(f"train={split_map['n_train']} val={split_map['n_val']} test={split_map['n_test']}")
