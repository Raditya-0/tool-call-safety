"""
Utilitas preprocessing tool call sequence
"""
import json
import re
from typing import List, Dict, Optional


# Normalisasi satu tool call
def normalize_tool_call(tool_call: Dict) -> Dict:
    name   = str(tool_call.get("name", tool_call.get("function", "unknown"))).lower().strip()
    params = tool_call.get("parameters", tool_call.get("arguments", tool_call.get("args", tool_call.get("params", {}))))

    # konversi params ke dict jika masih string
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {"raw": params}

    return {"name": name, "params": params}


# Flatten parameter JSON ke string
def flatten_params(params: Dict, max_keys: int = 10) -> str:
    if not isinstance(params, dict):
        return str(params)[:200]

    parts = []
    for i, (k, v) in enumerate(params.items()):
        if i >= max_keys:
            break
        parts.append(f"{k}={str(v)[:50]}")

    return " | ".join(parts)


# Tool call ke teks representasi
def tool_call_to_text(tool_call: Dict) -> str:
    norm   = normalize_tool_call(tool_call)
    params = flatten_params(norm["params"])
    return f"[FUNC] {norm['name']} [ARGS] {params}"


# Sequence tool calls ke list teks
def sequence_to_texts(sequence: List[Dict]) -> List[str]:
    return [tool_call_to_text(tc) for tc in sequence]


# Ekstrak konteks percakapan
def extract_context(conversation: List[Dict], n_tokens: int = 64) -> str:
    texts = []
    for turn in conversation[-3:]:  # ambil 3 turn terakhir
        role    = turn.get("role", "")
        content = str(turn.get("content", ""))[:100]
        texts.append(f"{role}: {content}")
    ctx = " ".join(texts)
    return ctx[:n_tokens * 4]  # estimasi token


# Validasi format satu sample
def validate_sample(sample: Dict) -> bool:
    has_sequence = "tool_call_sequence" in sample or "sequence" in sample
    has_label    = "label" in sample
    return has_sequence and has_label
