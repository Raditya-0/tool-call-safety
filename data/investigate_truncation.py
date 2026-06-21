"""
Investigasi dampak truncation flatten_params lintas source — bukan fix
"""
import json
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessing import normalize_tool_call


# Ambil semua value param tanpa truncation, return list panjang char
def param_value_lengths(tool_calls_json: str) -> list:
    try:
        seq = json.loads(tool_calls_json)
    except Exception:
        return []
    lengths = []
    for tc in seq:
        norm = normalize_tool_call(tc)
        params = norm["params"]
        if not isinstance(params, dict):
            lengths.append(len(str(params)))
            continue
        for v in params.values():
            lengths.append(len(str(v)))
    return lengths


# Bangun tool_calls_text versi tanpa cap 50 char per value (full param)
def flatten_params_full(params, max_keys: int = 10) -> str:
    if not isinstance(params, dict):
        return str(params)
    parts = []
    for i, (k, v) in enumerate(params.items()):
        if i >= max_keys:
            break
        parts.append(f"{k}={str(v)}")
    return " | ".join(parts)


def tool_call_to_text_full(tc: dict) -> str:
    norm = normalize_tool_call(tc)
    return f"[FUNC] {norm['name']} [ARGS] {flatten_params_full(norm['params'])}"


def sequence_to_text_full(tool_calls_json: str) -> str:
    try:
        seq = json.loads(tool_calls_json)
    except Exception:
        return ""
    return " ".join(tool_call_to_text_full(tc) for tc in seq)


# Cek apakah ada value param yang > 50 char (truncation kena)
def pct_samples_truncated(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, label), group in df.groupby(["source", "label"]):
        n = len(group)
        n_truncated = 0
        for tcj in group["tool_calls_json"]:
            lens = param_value_lengths(tcj)
            if any(l > 50 for l in lens):
                n_truncated += 1
        rows.append({
            "source": source,
            "label": label,
            "n_samples": n,
            "n_value_gt_50char": n_truncated,
            "pct_truncated": round(n_truncated / n * 100, 1) if n else 0.0,
        })
    return pd.DataFrame(rows)


# Percentile panjang char param value per label (kelas attack vs benign)
def char_length_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby("label"):
        all_lens = []
        for tcj in group["tool_calls_json"]:
            all_lens.extend(param_value_lengths(tcj))
        if not all_lens:
            continue
        arr = np.array(all_lens)
        rows.append({
            "label": label,
            "n_values": len(arr),
            "p50": np.percentile(arr, 50),
            "p75": np.percentile(arr, 75),
            "p90": np.percentile(arr, 90),
            "p95": np.percentile(arr, 95),
            "max": arr.max(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "tcssc_dataset.csv"))
    attack_df = df[df["label"] != "benign"]

    print("=== Persentase sample yang punya param value > 50 char (truncation kena) ===")
    trunc_report = pct_samples_truncated(attack_df)
    print(trunc_report.sort_values(["source", "label"]).to_string(index=False))

    print("\n=== Percentile panjang char param value per label ===")
    pct_report = char_length_percentiles(df)
    print(pct_report.to_string(index=False))

    trunc_report.to_csv(os.path.join(os.path.dirname(__file__), "diag_truncation_by_source.csv"), index=False)
    pct_report.to_csv(os.path.join(os.path.dirname(__file__), "diag_char_percentiles.csv"), index=False)
