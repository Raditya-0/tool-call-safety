"""
Dataset loader — semua sumber data TCSSC
"""
import json
import os
import platform
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
from utils.preprocessing import sequence_to_texts, extract_context, validate_sample


# Dataset utama
class ToolCallDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        tokenizer: AutoTokenizer,
        max_seq_len: int    = MAX_SEQ_LEN,
        max_ctx_len: int    = MAX_SEQ_LEN_CTX,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ):
        self.samples        = samples
        self.tokenizer      = tokenizer
        self.max_seq_len    = max_seq_len
        self.max_ctx_len    = max_ctx_len
        self.max_tool_calls = max_tool_calls

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_tool_call(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length     = self.max_seq_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def _tokenize_context(self, ctx: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            ctx,
            max_length     = self.max_ctx_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample   = self.samples[idx]
        sequence = sample.get("tool_call_sequence", sample.get("sequence", []))
        label    = LABEL2ID.get(sample["label"], 0) if isinstance(sample["label"], str) else sample["label"]
        ctx_text = extract_context(sample.get("conversation", []))

        tc_texts  = sequence_to_texts(sequence)
        tc_tokens = [self._tokenize_tool_call(t) for t in tc_texts[:self.max_tool_calls]]

        pad_len = self.max_tool_calls - len(tc_tokens)
        pad_enc = self._tokenize_tool_call("")
        tc_tokens += [pad_enc] * pad_len

        input_ids      = torch.stack([t["input_ids"]      for t in tc_tokens])
        attention_mask = torch.stack([t["attention_mask"] for t in tc_tokens])

        seq_len  = min(len(sequence), self.max_tool_calls)
        seq_mask = torch.zeros(self.max_tool_calls, dtype=torch.bool)
        seq_mask[:seq_len] = True

        ctx_enc = self._tokenize_context(ctx_text)

        return {
            "input_ids":          input_ids,
            "attention_mask":     attention_mask,
            "seq_mask":           seq_mask,
            "ctx_input_ids":      ctx_enc["input_ids"],
            "ctx_attention_mask": ctx_enc["attention_mask"],
            "label":              torch.tensor(label, dtype=torch.long),
        }


# Loader dataset STAC
def load_stac(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        # Extract tool calls from attack_plan.verified_tool_chain (clean format)
        chain = item.get("attack_plan", {}).get("verified_tool_chain", [])
        tool_calls = [
            {"name": step.get("tool_name", "unknown"), "parameters": step.get("parameters", {})}
            for step in chain
        ]

        if not tool_calls:
            continue

        # Build conversation context from interaction_history user/assistant text turns
        conversation = []
        for turn in item.get("interaction_history", []):
            role    = turn.get("role", "")
            content = turn.get("content")
            if role in ("user", "assistant") and content:
                conversation.append({"role": role, "content": content})

        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              "sequential_attack",
            "conversation":       conversation,
            "source":             "stac",
        })

    return samples


# Helper SafeToolBench
def _stb_api_to_tool_call(api_entry: dict) -> Optional[dict]:
    name = next((k for k in api_entry if k != "use_times"), None)
    if name is None:
        return None
    params = api_entry.get(name, {})
    if not isinstance(params, dict):
        params = {"value": params}
    return {"name": name, "parameters": params}


# Loader SafeToolBench (direktori)
def load_safetoolbench_dir(dir_path: str) -> List[Dict]:
    # label dari prefix nama file: BO/PD → direct_attack, PI/PL → parameter_injection
    prefix_to_label = {
        "BO": "direct_attack",
        "PD": "direct_attack",
        "PI": "parameter_injection",
        "PL": "parameter_injection",
    }

    samples = []
    for filename in sorted(os.listdir(dir_path)):
        if not filename.endswith(".json"):
            continue

        parts  = filename.replace(".json", "").split("_")   # ["query", "BO", "MA"]
        prefix = parts[1] if len(parts) >= 2 else ""
        label  = prefix_to_label.get(prefix, "direct_attack")

        filepath = os.path.join(dir_path, filename)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        file_count = 0
        for item in data:
            used_api   = item.get("output", {}).get("used_api", [])
            tool_calls = [tc for tc in (_stb_api_to_tool_call(e) for e in used_api) if tc]

            if not tool_calls:
                continue

            instruction  = item.get("instruction", "")
            conversation = [{"role": "user", "content": instruction}] if instruction else []

            samples.append({
                "tool_call_sequence": tool_calls,
                "label":              label,
                "conversation":       conversation,
                "source":             "safetoolbench",
            })
            file_count += 1

        print(f"  {filename} [{label}]: {file_count} samples")

    return samples


# Loader SafeToolBench (file tunggal, legacy)
def load_safetoolbench(path: str) -> List[Dict]:
    if os.path.isdir(path):
        return load_safetoolbench_dir(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        used_api   = item.get("output", {}).get("used_api", [])
        tool_calls = [tc for tc in (_stb_api_to_tool_call(e) for e in used_api) if tc]
        if not tool_calls:
            continue

        risk_type = item.get("risk_type", item.get("Risk category", ""))
        if "parameter" in risk_type.lower() or "injection" in risk_type.lower():
            label = "parameter_injection"
        else:
            label = "direct_attack"

        instruction  = item.get("instruction", "")
        conversation = [{"role": "user", "content": instruction}] if instruction else []

        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       conversation,
            "source":             "safetoolbench",
        })

    return samples


# Loader AgentHarm (legacy, file parquet tunggal)
def load_agentharm(path: str) -> List[Dict]:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    samples = []
    for _, row in df.iterrows():
        tool_calls = json.loads(row.get("tool_calls", "[]")) if isinstance(row.get("tool_calls"), str) else []
        label = "direct_attack" if row.get("is_harmful", False) else "benign"
        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       [],
            "source":             "agentharm",
        })
    return samples


# Loader ToolSafe AgentHarm (dari direktori)
def load_toolsafe_agentharm(dir_path: str) -> List[Dict]:
    """
    Load harmful dan benign behaviors dari ToolSafe AgentHarm dataset.
    target_functions → tool call sequence.
    harmful: sequential_attack (>1 tool) atau direct_attack (1 tool).
    """
    file_splits = [
        ("harmful_behaviors_test_public.json",  "harmful"),
        ("harmful_behaviors_validation.json",   "harmful"),
        ("benign_behaviors_test_public.json",   "benign"),
        ("benign_behaviors_validation.json",    "benign"),
    ]
    samples = []
    for fname, split_type in file_splits:
        fpath = os.path.join(dir_path, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        behaviors = data.get("behaviors", [])
        for item in behaviors:
            fns = item.get("target_functions", [])
            if not fns:
                continue
            tool_calls = [{"name": fn, "parameters": {}} for fn in fns]
            if split_type == "benign":
                label = "benign"
            else:
                label = "sequential_attack" if len(tool_calls) > 1 else "direct_attack"
            prompt = str(item.get("prompt", ""))[:500]
            samples.append({
                "tool_call_sequence": tool_calls,
                "label":              label,
                "conversation":       [{"role": "user", "content": prompt}] if prompt else [],
                "source":             "toolsafe_agentharm",
            })
        print(f"  {fname} [{split_type}]: {len(behaviors)} items")
    return samples


# Loader ToolSafe AgentSafeBench
def load_toolsafe_safetybench(path: str) -> List[Dict]:
    """
    Load released_data.json dari ToolSafe AgentSafeBench.
    environments[].tools → tool call sequence.
    fulfillable=1 → benign, fulfillable=0 → harmful.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        tool_names = []
        for env in item.get("environments", []):
            tool_names.extend(env.get("tools", []))
        if not tool_names:
            continue
        tool_calls = [{"name": fn, "parameters": {}} for fn in tool_names[:MAX_TOOL_CALLS]]
        fulfillable = item.get("fulfillable", 0)
        if fulfillable == 1:
            label = "benign"
        else:
            label = "sequential_attack" if len(tool_calls) > 2 else "direct_attack"
        instruction = str(item.get("instruction", ""))[:500]
        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       [{"role": "user", "content": instruction}] if instruction else [],
            "source":             "toolsafe_safetybench",
        })
    return samples


# Loader WildJailbreak
def load_wildjailbreak(path: str) -> List[Dict]:
    """
    Load WildJailbreak eval parquet.
    Kolom: adversarial (teks), label (0=benign/1=harmful), data_type (string).
    Tool call diekstrak dari teks dengan regex; fallback ke generik.
    """
    import re
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)

    # Mapping label dari kolom data_type (paling reliable)
    # Fallback: label int 0→benign, 1→harmful
    def _get_label(row) -> str:
        dt = str(row.get("data_type", "")).lower()
        if "benign" in dt:
            return "benign"
        if "harmful" in dt:
            return "sequential_attack"   # adversarial = multi-step jailbreak
        # fallback ke int label
        lbl = row.get("label", 0)
        return "benign" if int(lbl) == 0 else "sequential_attack"

    # Deteksi kolom teks prompt
    text_priority = ["adversarial", "vanilla", "prompt", "text", "query"]
    text_col = next((c for c in text_priority if c in df.columns), None)
    if text_col is None:
        text_col = df.columns[0]

    tc_pattern = re.compile(
        r'\b([a-z][a-z0-9]*_[a-z][a-z0-9_]*)\s*\('
        r'|"(?:name|function)"\s*:\s*"([^"]{2,40})"'
        r'|`([a-z][a-z0-9_]{2,})\s*\(',
        re.IGNORECASE,
    )

    samples = []
    for _, row in df.iterrows():
        prompt_text = str(row.get(text_col, "")).strip()[:500]
        if not prompt_text:
            continue

        label = _get_label(row)

        # ekstrak tool call dari teks
        matches = tc_pattern.findall(prompt_text)
        found   = [next(g for g in m if g) for m in matches if any(m)]
        found   = list(dict.fromkeys(fn.lower() for fn in found))[:5]

        if found:
            tool_calls = [{"name": fn, "parameters": {}} for fn in found]
        else:
            generic    = "benign_action" if label == "benign" else "harmful_action"
            tool_calls = [{"name": generic, "parameters": {"context": prompt_text[:80]}}]

        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       [{"role": "user", "content": prompt_text[:300]}],
            "source":             "wildjailbreak",
        })
    return samples


# Loader scraped (raw, tanpa label)
def load_scraped(path: str) -> List[Dict]:
    df = pd.read_csv(path)
    samples = []
    for _, row in df.iterrows():
        tool_calls = json.loads(row.get("tool_calls", "[]")) if isinstance(row.get("tool_calls"), str) else []
        label      = row.get("pseudo_label", "benign")
        if not label or str(label).strip() == "":
            label = "benign"
        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       [],
            "source":             "scraped",
        })
    return samples


# Loader augmented data
def load_augmented(path: str) -> List[Dict]:
    """Load augmented.csv yang dihasilkan oleh augmentor.py."""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    valid_labels = set(LABEL2ID.keys())
    samples = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        if label not in valid_labels:
            continue
        try:
            tc = json.loads(row.get("tool_call_sequence", "[]"))
        except Exception:
            tc = []
        if not tc:
            continue
        try:
            conv = json.loads(row.get("conversation", "[]"))
        except Exception:
            conv = []
        samples.append({
            "tool_call_sequence": tc,
            "label":              label,
            "conversation":       conv,
            "source":             "augmented",
        })
    return samples


# Loader scraped labeled (hasil pseudolabeling)
def load_scraped_labeled(path: str) -> List[Dict]:
    df = pd.read_csv(path)
    valid_labels = set(LABEL2ID.keys())
    samples = []
    for _, row in df.iterrows():
        label = str(row.get("pseudo_label", "benign")).strip()
        if label not in valid_labels:
            label = "benign"
        tool_calls_raw = row.get("tool_calls", "[]")
        try:
            tool_calls = json.loads(tool_calls_raw) if isinstance(tool_calls_raw, str) else []
        except Exception:
            tool_calls = []
        if not tool_calls:
            continue
        title = str(row.get("title", ""))[:200]
        body  = str(row.get("body", ""))[:300]
        conversation = [{"role": "user", "content": (title + " " + body).strip()}] if (title or body) else []
        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       conversation,
            "source":             "scraped_labeled",
        })
    return samples


# Generate synthetic benign samples
def generate_benign_samples(n: int = 100) -> List[Dict]:
    templates = [
        {"name": "get_weather",        "parameters": {"location": "Jakarta", "units": "metric"}},
        {"name": "search_web",         "parameters": {"query": "best restaurants near me"}},
        {"name": "get_calendar_events","parameters": {"date": "2024-01-15", "user_id": "user1"}},
        {"name": "send_reminder",      "parameters": {"message": "Team meeting at 3pm", "time": "14:45"}},
        {"name": "check_stock_price",  "parameters": {"symbol": "BBCA"}},
        {"name": "translate_text",     "parameters": {"text": "Hello world", "target_lang": "id"}},
        {"name": "get_news",           "parameters": {"topic": "technology", "count": 5}},
        {"name": "calculate",          "parameters": {"expression": "15 * 23"}},
        {"name": "set_alarm",          "parameters": {"time": "07:00", "label": "Morning alarm"}},
        {"name": "get_directions",     "parameters": {"origin": "home", "destination": "office"}},
    ]
    samples = []
    for i in range(n):
        tc = templates[i % len(templates)]
        samples.append({
            "tool_call_sequence": [tc],
            "label":              "benign",
            "conversation":       [{"role": "user", "content": f"Please {tc['name'].replace('_', ' ')}"}],
            "source":             "synthetic",
        })
    return samples


# Gabungkan semua dataset
def load_all_datasets(processed_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_samples = []

    # 1. SafeToolBench
    stb_dir = os.path.join(processed_dir, "safetoolbench")
    if os.path.isdir(stb_dir):
        loaded = load_safetoolbench_dir(stb_dir)
        all_samples.extend(loaded)
        print(f"SafeToolBench: {len(loaded)} samples")
    else:
        print(f"Skip SafeToolBench — tidak ditemukan: {stb_dir}")

    # 2. STAC
    if os.path.exists(STAC_PATH):
        loaded = load_stac(STAC_PATH)
        all_samples.extend(loaded)
        print(f"STAC: {len(loaded)} samples")
    else:
        print(f"Skip STAC — tidak ditemukan: {STAC_PATH}")

    # 3. ToolSafe AgentHarm
    if os.path.isdir(TOOLSAFE_AGENTHARM_DIR):
        loaded = load_toolsafe_agentharm(TOOLSAFE_AGENTHARM_DIR)
        all_samples.extend(loaded)
        print(f"ToolSafe AgentHarm: {len(loaded)} samples")
    else:
        print(f"Skip ToolSafe AgentHarm — tidak ditemukan: {TOOLSAFE_AGENTHARM_DIR}")

    # 4. ToolSafe AgentSafeBench
    if os.path.exists(TOOLSAFE_SAFETYBENCH_PATH):
        loaded = load_toolsafe_safetybench(TOOLSAFE_SAFETYBENCH_PATH)
        all_samples.extend(loaded)
        print(f"ToolSafe SafetyBench: {len(loaded)} samples")
    else:
        print(f"Skip ToolSafe SafetyBench — tidak ditemukan: {TOOLSAFE_SAFETYBENCH_PATH}")

    # 5. WildJailbreak (jika sudah didownload)
    if os.path.exists(WILDJAILBREAK_PATH):
        loaded = load_wildjailbreak(WILDJAILBREAK_PATH)
        all_samples.extend(loaded)
        print(f"WildJailbreak: {len(loaded)} samples")
    else:
        print(f"Skip WildJailbreak — belum didownload: {WILDJAILBREAK_PATH}")

    # 6. Scraped labeled (hasil pseudolabeling)
    if os.path.exists(SCRAPED_LABELED_PATH):
        loaded = load_scraped_labeled(SCRAPED_LABELED_PATH)
        all_samples.extend(loaded)
        print(f"Scraped labeled: {len(loaded)} samples")
    else:
        print(f"Skip scraped labeled — belum ada: {SCRAPED_LABELED_PATH}")

    # 7. Augmented data (hasil augmentor.py)
    if os.path.exists(AUGMENTED_PATH):
        loaded = load_augmented(AUGMENTED_PATH)
        all_samples.extend(loaded)
        print(f"Augmented: {len(loaded)} samples")
    else:
        print(f"Skip augmented — belum ada: {AUGMENTED_PATH}")

    # 8. Legacy scraped raw (jika ada)
    for path in [os.path.join(RAW_DIR, "agentharm.parquet")]:
        if os.path.exists(path):
            loaded = load_agentharm(path)
            all_samples.extend(loaded)
            print(f"AgentHarm legacy: {len(loaded)} dari {os.path.basename(path)}")

    # 9. Synthetic benign minimal (100 saja — balance diserahkan ke augmentor)
    benign = generate_benign_samples(100)
    all_samples.extend(benign)
    print(f"Synthetic benign: {len(benign)} samples")

    print(f"\nTotal: {len(all_samples)} samples")
    dist = Counter(s["label"] for s in all_samples)
    print("Distribusi label:", dict(dist))

    # Split 70/15/15 stratified
    labels = [s["label"] for s in all_samples]
    train, temp = train_test_split(all_samples, test_size=0.30, random_state=SEED, stratify=labels)
    temp_labels = [s["label"] for s in temp]
    val, test   = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=temp_labels)

    return train, val, test


# Buat DataLoader
def build_dataloaders(
    train_samples: List[Dict],
    val_samples:   List[Dict],
    test_samples:  List[Dict],
    tokenizer:     AutoTokenizer,
    batch_size:    int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = ToolCallDataset(train_samples, tokenizer)
    val_ds   = ToolCallDataset(val_samples,   tokenizer)
    test_ds  = ToolCallDataset(test_samples,  tokenizer)

    # num_workers=0 on Windows to avoid multiprocessing spawn issues
    nw = 0 if platform.system() == "Windows" else 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=False)

    return train_loader, val_loader, test_loader
