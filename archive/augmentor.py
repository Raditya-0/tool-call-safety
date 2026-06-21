"""
Data augmentor TCSSC.
Augmentasi tool call sequences per kelas untuk balance distribusi.
Hasil disimpan ke data/processed/augmented.csv.
Jalankan: python data/augmentor.py
"""
import os
import sys
import json
import copy
import random
from collections import Counter
from typing import List, Dict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DIR, STAC_PATH,
    TOOLSAFE_AGENTHARM_DIR, TOOLSAFE_SAFETYBENCH_PATH,
    WILDJAILBREAK_PATH, SCRAPED_LABELED_PATH,
    AUGMENTED_PATH, LABEL2ID,
)
from data.dataset import (
    load_safetoolbench_dir, load_stac,
    load_toolsafe_agentharm, load_toolsafe_safetybench,
    load_wildjailbreak, load_scraped_labeled,
    generate_benign_samples,
)

SEED = 42

TARGETS = {
    "sequential_attack":   2500,   # sudah 2827 di base → skip otomatis
    "direct_attack":       2500,
    "benign":              2500,
    "parameter_injection": 2500,   # naik dari 2200
}

# Sinonim verb untuk substitusi nama API
VERB_SYNONYMS = {
    "delete":   ["remove", "erase", "clear", "purge"],
    "remove":   ["delete", "erase", "clear"],
    "get":      ["fetch", "retrieve", "read", "load"],
    "fetch":    ["get", "retrieve", "read"],
    "retrieve": ["get", "fetch", "load"],
    "set":      ["update", "write", "store", "assign"],
    "update":   ["set", "write", "modify", "store"],
    "create":   ["make", "add", "generate", "build"],
    "make":     ["create", "generate", "build"],
    "add":      ["create", "insert", "append", "push"],
    "check":    ["verify", "validate", "inspect", "confirm"],
    "verify":   ["check", "validate", "inspect"],
    "send":     ["submit", "post", "dispatch", "transmit"],
    "post":     ["send", "submit", "dispatch"],
    "submit":   ["send", "post", "dispatch"],
    "search":   ["find", "query", "lookup", "scan"],
    "find":     ["search", "query", "lookup"],
    "query":    ["search", "find", "lookup"],
    "list":     ["enumerate", "getAll", "showAll", "index"],
    "upload":   ["push", "submit", "transfer"],
    "download": ["pull", "fetch", "retrieve"],
    "run":      ["execute", "launch", "invoke", "call"],
    "execute":  ["run", "launch", "invoke"],
    "stop":     ["halt", "kill", "terminate", "cancel"],
    "cancel":   ["stop", "abort", "terminate"],
    "read":     ["load", "fetch", "get"],
    "write":    ["save", "store", "put"],
    "save":     ["write", "store", "persist"],
    "edit":     ["update", "modify", "change"],
    "modify":   ["edit", "update", "change", "alter"],
    "share":    ["send", "broadcast", "distribute"],
    "manage":   ["control", "handle", "administer"],
    "control":  ["manage", "handle", "operate"],
    "monitor":  ["watch", "track", "observe"],
    "track":    ["monitor", "follow", "observe"],
    "rate":     ["score", "evaluate", "rank"],
    "book":     ["reserve", "schedule", "register"],
    "reserve":  ["book", "schedule", "register"],
    "pay":      ["charge", "transfer", "process"],
    "transfer": ["send", "move", "forward"],
    "append":   ["add", "insert", "attach"],
    "inject":   ["insert", "embed", "introduce"],
    "scan":     ["search", "check", "inspect"],
    "open":     ["access", "load", "start"],
    "close":    ["stop", "end", "terminate"],
    "move":     ["transfer", "relocate", "shift"],
    "copy":     ["duplicate", "clone", "replicate"],
    "reset":    ["clear", "restore", "reinitialize"],
}

# Pool nilai untuk perturbasi
_FAKE_IDS = [
    "user001", "user042", "user099", "usr789", "account42",
    "member55", "id_x01", "uid_2024", "abc_999", "def_777",
    "user_88", "client_12", "acct_305",
]
_FAKE_DATES = [
    "2024-01-15", "2024-03-22", "2024-07-08",
    "2023-12-01", "2025-02-14", "2024-09-30",
    "2024-06-18", "2024-11-05",
]
_FAKE_LOCATIONS = [
    "Jakarta", "Surabaya", "Bandung", "Medan", "Bali",
    "Yogyakarta", "Semarang", "Makassar", "Palembang",
]
_OPTIONAL_PARAMS = [
    ("timeout",  [30, 60, 120, 300]),
    ("retry",    [1, 2, 3]),
    ("format",   ["json", "xml", "csv", "text"]),
    ("version",  ["v1", "v2", "v3"]),
    ("limit",    [10, 20, 50, 100]),
    ("verbose",  [True, False]),
    ("dry_run",  [True]),
    ("async",    [True, False]),
]


# Augmentasi nama tool call
def _augment_name(name: str) -> str:
    low = name.lower()

    # snake_case: cek token pertama
    parts = name.split("_")
    if len(parts) > 1 and parts[0].lower() in VERB_SYNONYMS:
        syns = VERB_SYNONYMS[parts[0].lower()]
        chosen = random.choice(syns)
        if chosen != parts[0].lower():
            parts[0] = chosen
            return "_".join(parts)

    # camelCase / PascalCase: cek prefix
    for verb, syns in VERB_SYNONYMS.items():
        if low.startswith(verb):
            chosen = random.choice(syns)
            if chosen.lower() != verb:
                rest = name[len(verb):]
                if name[0].isupper():
                    chosen = chosen[0].upper() + chosen[1:]
                return chosen + rest

    return name


# Perturbasi nilai parameter
def _perturb_value(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v + random.choice([-3, -2, -1, 1, 2, 3])
    if isinstance(v, float):
        return round(v * random.uniform(0.85, 1.15), 2)
    if isinstance(v, str):
        low = v.lower()
        if any(k in low for k in ["user", "uid", "account", "member", "client", "id_"]):
            return random.choice(_FAKE_IDS)
        if any(k in low for k in ["ride", "trip", "journey", "order", "booking"]):
            prefix = random.choice(["ride", "trip", "order", "booking"])
            return f"{prefix}_{random.randint(100, 999)}"
        if any(c in v for c in ["-01-", "-02-", "-03-", "-04-", "-05-",
                                  "-06-", "-07-", "-08-", "-09-", "-10-",
                                  "-11-", "-12-"]):
            return random.choice(_FAKE_DATES)
        if any(k in low for k in ["jakarta", "bandung", "surabaya",
                                   "location", "city", "place", "address"]):
            return random.choice(_FAKE_LOCATIONS)
        if v.isdigit():
            return str(int(v) + random.randint(1, 99))
        if len(v) > 2:
            return v + str(random.randint(1, 9))
    if isinstance(v, list) and v:
        return v
    return v


# Augmentasi parameter dict
def _augment_params(params: dict) -> dict:
    if not isinstance(params, dict) or not params:
        return params

    new_params = {}
    for k, v in params.items():
        new_params[k] = _perturb_value(v) if random.random() < 0.65 else v

    # 20% buang satu param (jika lebih dari 1)
    if len(new_params) > 1 and random.random() < 0.20:
        del new_params[random.choice(list(new_params.keys()))]

    # 25% tambah satu optional param
    if random.random() < 0.25:
        opt_key, opt_vals = random.choice(_OPTIONAL_PARAMS)
        if opt_key not in new_params:
            new_params[opt_key] = random.choice(opt_vals)

    # 20% reorder keys
    if random.random() < 0.20:
        keys = list(new_params.keys())
        random.shuffle(keys)
        new_params = {k: new_params[k] for k in keys}

    return new_params


# Augmentasi satu tool call
def _augment_tool_call(tc: dict) -> dict:
    new_tc = copy.deepcopy(tc)
    if random.random() < 0.55:
        new_tc["name"] = _augment_name(tc.get("name", ""))
    params = tc.get("parameters", {})
    if params:
        new_tc["parameters"] = _augment_params(params)
    return new_tc


# Augmentasi satu sample
def _augment_sample(sample: dict) -> dict:
    new_s = copy.deepcopy(sample)
    new_s["tool_call_sequence"] = [
        _augment_tool_call(tc) for tc in sample["tool_call_sequence"]
    ]
    new_s["source"] = "augmented"
    return new_s


# Load semua base samples
def _load_base_samples() -> List[Dict]:
    all_samples = []

    stb_dir = os.path.join(PROCESSED_DIR, "safetoolbench")
    if os.path.isdir(stb_dir):
        all_samples.extend(load_safetoolbench_dir(stb_dir))
    if os.path.exists(STAC_PATH):
        all_samples.extend(load_stac(STAC_PATH))
    if os.path.isdir(TOOLSAFE_AGENTHARM_DIR):
        all_samples.extend(load_toolsafe_agentharm(TOOLSAFE_AGENTHARM_DIR))
    if os.path.exists(TOOLSAFE_SAFETYBENCH_PATH):
        all_samples.extend(load_toolsafe_safetybench(TOOLSAFE_SAFETYBENCH_PATH))
    if os.path.exists(WILDJAILBREAK_PATH):
        all_samples.extend(load_wildjailbreak(WILDJAILBREAK_PATH))
    if os.path.exists(SCRAPED_LABELED_PATH):
        all_samples.extend(load_scraped_labeled(SCRAPED_LABELED_PATH))

    all_samples.extend(generate_benign_samples(100))
    return all_samples


# Main
def main():
    random.seed(SEED)

    print("Loading base samples...")
    base = _load_base_samples()
    counts = Counter(s["label"] for s in base)

    print(f"\nDistribusi saat ini ({sum(counts.values())} total):")
    for label in sorted(TARGETS):
        cur = counts.get(label, 0)
        tgt = TARGETS[label]
        need = max(0, tgt - cur)
        print(f"  {label:<25}: {cur:>5}  →  target {tgt}  (perlu +{need})")

    # kelompokkan per label
    by_label: Dict[str, List[Dict]] = {}
    for s in base:
        lbl = s["label"]
        if lbl in TARGETS:
            by_label.setdefault(lbl, []).append(s)

    augmented: List[Dict] = []

    for label, target in TARGETS.items():
        current  = counts.get(label, 0)
        n_needed = max(0, target - current)

        if n_needed == 0:
            print(f"\n{label}: sudah {current} ≥ {target}, skip.")
            continue

        pool = by_label.get(label, [])
        if not pool:
            print(f"\nWARN: tidak ada base sample untuk {label}, skip.")
            continue

        print(f"\n{label}: generate {n_needed} augmented samples...")
        for _ in range(n_needed):
            src = random.choice(pool)
            augmented.append(_augment_sample(src))
        print(f"  Selesai: {n_needed} samples.")

    if not augmented:
        print("\nTidak ada yang perlu diaugmentasi.")
        return

    # Simpan ke CSV
    records = [
        {
            "label":              s["label"],
            "tool_call_sequence": json.dumps(s["tool_call_sequence"], ensure_ascii=False),
            "conversation":       json.dumps(s.get("conversation", []), ensure_ascii=False),
            "source":             "augmented",
        }
        for s in augmented
    ]

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(AUGMENTED_PATH), exist_ok=True)
    df.to_csv(AUGMENTED_PATH, index=False)

    aug_counts = Counter(r["label"] for r in records)

    print(f"\n{'='*55}")
    print(f"Augmented disimpan → {AUGMENTED_PATH}")
    print(f"Total augmented   : {len(records)}")
    print("\nDistribusi augmented:")
    for label in sorted(aug_counts):
        print(f"  {label:<25}: {aug_counts[label]:>5}")

    print(f"\nEstimasi total setelah augmentasi:")
    grand = 0
    for label in sorted(TARGETS):
        total = counts.get(label, 0) + aug_counts.get(label, 0)
        grand += total
        print(f"  {label:<25}: {total:>5}  (target {TARGETS[label]})")
    print(f"  {'TOTAL':<25}: {grand:>5}")


if __name__ == "__main__":
    main()
