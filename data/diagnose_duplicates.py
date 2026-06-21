"""
Diagnostik duplikat tool_calls_text — bukan dedup, cuma analisis
"""
import json
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Normalisasi json supaya beda urutan key/whitespace tidak dianggap beda
def normalize_json(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    return json.dumps(obj, sort_keys=True)


# Tandai grup duplikat sebagai true duplicate atau benign overlap
def classify_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df.duplicated(subset=["tool_calls_text"], keep=False)
    dup_df = df[dup_mask].copy()

    results = []
    for text, group in dup_df.groupby("tool_calls_text"):
        ctx_unique = group["context_text"].nunique()
        category = "true_duplicate" if ctx_unique == 1 else "benign_overlap"
        for idx in group.index:
            results.append((idx, category))

    cat_series = pd.Series(dict(results), name="dup_category")
    dup_df = dup_df.join(cat_series)
    return dup_df


# Cek apakah true duplicate sebenarnya beda cuma di urutan key/whitespace json
def check_json_normalization(dup_df: pd.DataFrame) -> pd.DataFrame:
    true_dup = dup_df[dup_df["dup_category"] == "true_duplicate"].copy()
    rows = []
    for text, group in true_dup.groupby("tool_calls_text"):
        raw_jsons = group["tool_calls_json"].tolist()
        norm_jsons = [normalize_json(j) for j in raw_jsons]
        same_raw = len(set(raw_jsons)) == 1
        same_norm = len(set(norm_jsons)) == 1
        rows.append({
            "tool_calls_text": text[:80],
            "n_rows": len(group),
            "same_raw_json": same_raw,
            "same_norm_json": same_norm,
            "would_collapse_if_normalized": (not same_raw) and same_norm,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "tcssc_dataset.csv"))

    dup_df = classify_duplicates(df)
    n_true = (dup_df["dup_category"] == "true_duplicate").sum()
    n_benign = (dup_df["dup_category"] == "benign_overlap").sum()
    total_dup_rows = len(dup_df)

    print(f"Total baris terlibat duplikat tool_calls_text: {total_dup_rows}")
    print(f"True duplicate (tool_calls_text & context_text identik): {n_true} ({n_true/total_dup_rows*100:.1f}%)")
    print(f"Benign overlap (tool_calls_text sama, context_text beda): {n_benign} ({n_benign/total_dup_rows*100:.1f}%)")

    print("\nBreakdown per label:")
    print(dup_df.groupby(["dup_category", "label"]).size().unstack(fill_value=0))

    print("\nBreakdown per source:")
    print(dup_df.groupby(["dup_category", "source"]).size().unstack(fill_value=0))

    norm_check = check_json_normalization(dup_df)
    n_would_collapse = norm_check["would_collapse_if_normalized"].sum()
    print(f"\nGrup true duplicate yang tool_calls_json beda raw tapi sama setelah normalize: {n_would_collapse} / {len(norm_check)}")
    if n_would_collapse > 0:
        print(norm_check[norm_check["would_collapse_if_normalized"]].head(10).to_string())

    norm_check.to_csv(os.path.join(os.path.dirname(__file__), "diag_true_dup_json_check.csv"), index=False)
    dup_df.to_csv(os.path.join(os.path.dirname(__file__), "diag_duplicate_rows.csv"), index=False)
