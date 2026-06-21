"""
Dataset loader — semua sumber data TCSSC
"""
import json
import os
import platform
import re
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


# Helper R-Judge: parse string action (format tidak konsisten antar file)
def _rjudge_parse_action(action) -> Dict:
    """
    Format action di R-Judge bervariasi per file:
      - ToolName{'key': 'val'}        (python dict repr, dh_*/ds_* files)
      - ToolName: {"key": "val"}      (json, kebanyakan file)
      - {ToolName: {"key": "val"}}    (dibungkus kurung kurawal, socialapp)
      - ```bash\ncommand\n```         (code fence shell command, terminal.json)
      - <Tag>-<Tag> / SET <X>: <Y>    (DSL UI action, phone*/iot files)
      - teks bebas tanpa tool call    (attack_type unintended non-tool, chatbot/mail)
    """
    import ast

    if action is None:
        return {"name": "unknown", "parameters": {}}
    if isinstance(action, dict):
        return {"name": str(action.get("name", "unknown")), "parameters": action.get("parameters", action)}

    text = str(action).strip()
    if not text:
        return {"name": "unknown", "parameters": {}}

    # code fence shell command
    fence = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if fence:
        return {"name": "shell_command", "parameters": {"command": fence.group(1).strip()[:300]}}

    # buang pembungkus { } luar (mis. "{FacebookManagerCreatePost: {...}}")
    stripped = text
    if stripped.startswith("{") and stripped.endswith("}"):
        inner = stripped[1:-1].strip()
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*:', inner):
            stripped = inner

    # ToolName diikuti blob { ... } (dict repr ATAU json)
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*:?\s*(\{.*\})\s*$', stripped, re.DOTALL)
    if m:
        name, blob = m.group(1), m.group(2)
        params = None
        try:
            params = json.loads(blob)
        except Exception:
            try:
                params = ast.literal_eval(blob)
            except Exception:
                params = {"raw": blob[:300]}
        if not isinstance(params, dict):
            params = {"value": params}
        return {"name": name, "parameters": params}

    # DSL UI action: "SET <X>: <Y>" atau "Click <A>-<B>" — tanpa parameter JSON
    if "<" in text and ">" in text:
        return {"name": "ui_action", "parameters": {"raw": text[:300]}}

    # teks bebas (tidak ada tool call — respons natural language)
    return {"name": "respond", "parameters": {"text": text[:300]}}


# Loader R-Judge
def load_rjudge(json_path: str) -> List[Dict]:
    """
    Load satu file R-Judge. Setiap item punya 'contents' = list of round,
    tiap round [user, agent, environment]. agent.action → tool call.
    Label: attack_type=='injection' & label==1 → parameter_injection;
    label==1 lain → sequential_attack (>1 tool call) / direct_attack (1 tool call);
    label==0 → benign.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        rounds = item.get("contents", [])
        tool_calls  = []
        conversation = []
        for round_ in rounds:
            for turn in round_:
                role = turn.get("role", "")
                if role == "user" and turn.get("content"):
                    conversation.append({"role": "user", "content": str(turn["content"])[:300]})
                elif role == "environment" and turn.get("content"):
                    conversation.append({"role": "environment", "content": str(turn["content"])[:300]})
                elif role == "agent" and turn.get("action") is not None:
                    tool_calls.append(_rjudge_parse_action(turn["action"]))

        if not tool_calls:
            continue

        is_unsafe   = item.get("label", 0) == 1
        attack_type = item.get("attack_type", "")

        if not is_unsafe:
            label = "benign"
        elif attack_type == "injection":
            label = "parameter_injection"
        else:
            label = "sequential_attack" if len(tool_calls) > 1 else "direct_attack"

        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       conversation,
            "source":             "rjudge",
        })

    return samples


# Loader R-Judge (semua file di direktori data/, rekursif per kategori)
def load_rjudge_dir(dir_path: str) -> List[Dict]:
    import glob
    samples = []
    for fpath in sorted(glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True)):
        try:
            loaded = load_rjudge(fpath)
        except Exception as e:
            print(f"  Skip {fpath} (AV-quarantined or unreadable): {e}")
            continue
        samples.extend(loaded)
    return samples


# Helper InjecAgent: parse "Tool Parameters" (python dict repr string)
def _injecagent_parse_params(raw) -> Dict:
    import ast
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        params = ast.literal_eval(text)
    except Exception:
        try:
            params = json.loads(text)
        except Exception:
            return {"raw": text[:300]}
    return params if isinstance(params, dict) else {"value": params}


# Loader InjecAgent (satu file test_cases_*.json)
def load_injecagent(json_path: str, label: str) -> List[Dict]:
    """
    Setiap item = 1 skenario injection: agent memanggil 'User Tool' (legit),
    'Tool Response' (environment) membawa instruksi tersembunyi attacker,
    lalu 'Attacker Tools' adalah tool yang dipanggil agent kalau termakan injeksi.
    label diberikan oleh caller berdasar kategori file (dh → direct_attack, ds → parameter_injection).
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        tool_calls = []

        user_tool = item.get("User Tool")
        if user_tool:
            tool_calls.append({
                "name":       user_tool,
                "parameters": _injecagent_parse_params(item.get("Tool Parameters", "{}")),
            })

        for attacker_tool in item.get("Attacker Tools", []):
            tool_calls.append({"name": attacker_tool, "parameters": {}})

        if not tool_calls:
            continue

        conversation = []
        user_instruction = item.get("User Instruction", "")
        if user_instruction:
            conversation.append({"role": "user", "content": str(user_instruction)[:300]})
        tool_response = item.get("Tool Response", "")
        if tool_response:
            conversation.append({"role": "environment", "content": str(tool_response)[:300]})

        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              label,
            "conversation":       conversation,
            "source":             "injecagent",
        })

    return samples


# Loader InjecAgent (semua file test_cases di direktori data/)
def load_injecagent_dir(dir_path: str) -> List[Dict]:
    file_labels = [
        ("test_cases_dh_base.json",     "direct_attack"),
        ("test_cases_dh_enhanced.json", "direct_attack"),
        ("test_cases_ds_base.json",     "parameter_injection"),
        ("test_cases_ds_enhanced.json", "parameter_injection"),
    ]
    samples = []
    for fname, label in file_labels:
        fpath = os.path.join(dir_path, fname)
        if not os.path.exists(fpath):
            continue
        loaded = load_injecagent(fpath, label)
        samples.extend(loaded)
    return samples


# Loader AgentDojo (Debenedetti et al., NeurIPS 2024) — successful prompt injection traces
def load_agentdojo(
    runs_dir:  str = None,
    pipeline:  str = None,
    n_samples: int = 512,
    seed:      int = 42,
) -> List[Dict]:
    """
    Load recorded agent transcripts dari AgentDojo runs/<pipeline>/**/injection_task_*.json.
    Hanya ambil trace dengan security==True (injeksi berhasil membajak agent) sebagai
    contoh parameter_injection — agent termakan instruksi tersembunyi di tool response
    lalu memanggil tool tambahan/berbeda dari yang diminta user.
    """
    import glob
    import random

    runs_dir = runs_dir or AGENTDOJO_RUNS_DIR
    pipeline = pipeline or AGENTDOJO_PIPELINE
    pattern  = os.path.join(runs_dir, pipeline, "**", "injection_task_*.json")
    files    = glob.glob(pattern, recursive=True)

    candidates = []
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue
        if item.get("security") is True:
            candidates.append(item)

    random.seed(seed)
    random.shuffle(candidates)

    samples = []
    for item in candidates[:n_samples]:
        tool_calls = []
        conversation = []
        for msg in item.get("messages", []):
            role    = msg.get("role", "")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                conversation.append({"role": role, "content": str(content)[:300]})
            elif role == "tool" and content:
                conversation.append({"role": "environment", "content": str(content)[:300]})

            for tc in (msg.get("tool_calls") or []):
                tool_calls.append({
                    "name":       tc.get("function", "unknown"),
                    "parameters": tc.get("args", {}),
                })

        if not tool_calls:
            continue

        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              "parameter_injection",
            "conversation":       conversation,
            "source":             "agentdojo",
        })

    return samples


# Loader APIGen-MT-5k (Liu et al., NeurIPS 2024 — Salesforce APIGen pipeline) — benign
def load_apigen_mt_benign(n_samples: int = 4000, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset

    SUSPICIOUS_KEYWORDS = ["delete_all", "transfer_all", "format_disk", "rm -rf", "drop_table", "wipe"]

    ds = load_dataset("Salesforce/APIGen-MT-5k", split="train")
    ds = ds.shuffle(seed=seed)

    samples = []
    for item in ds:
        if len(samples) >= n_samples:
            break

        tool_calls = []
        conversation = []
        for turn in item.get("conversations", []):
            who     = turn.get("from", "")
            content = turn.get("value", "")
            if who == "human" and content:
                conversation.append({"role": "user", "content": str(content)[:300]})
            elif who == "gpt" and content:
                conversation.append({"role": "assistant", "content": str(content)[:300]})
            elif who == "function_call" and content:
                try:
                    call = json.loads(content)
                except Exception:
                    continue
                tool_calls.append({
                    "name":       call.get("name", "unknown"),
                    "parameters": call.get("arguments", {}),
                })

        if not tool_calls:
            continue

        tc_str = str(tool_calls).lower()
        if any(kw in tc_str for kw in SUSPICIOUS_KEYWORDS):
            continue

        samples.append({
            "tool_call_sequence": tool_calls,
            "label":              "benign",
            "conversation":       conversation,
            "source":             "apigen_mt",
        })

    return samples

# load BIPIA
def load_bipia(n_samples: int = 3500, seed: int = 42) -> List[Dict]:
    from datasets import load_dataset
    
    ds = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT", split="train")
    df = ds.to_pandas()
    
    df_filtered = df[df['source'] == 'BIPIA'].copy()
    df_filtered = df_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    samples = []
    for _, row in df_filtered.iterrows():
        if len(samples) >= n_samples:
            break
            
        context = str(row.get("context", "")).strip()
        user_intent = str(row.get("user_intent", "")).strip()
        
        if not context:
            continue
            
        tool_calls = [{
            "name": "process_external_content",
            "parameters": {"content": context[:500]}
        }]
        
        conversation = []
        if user_intent:
            conversation.append({"role": "user", "content": user_intent[:300]})
            
        samples.append({
            "tool_call_sequence": tool_calls,
            "label": "parameter_injection",
            "conversation": conversation,
            "source": "bipia"
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


# Loader RedCode-Exec (GitHub AI-secure/RedCode) — direct_attack
def load_redcode_exec(dir_path: str, n_log: int = 0) -> List[Dict]:
    """
    Hanya scenario index yang sudah diverifikasi manual benar-benar
    risky/malicious (exfiltrasi, backdoor, code injection, dst) yang dipakai —
    scenario bias/fairness/benign-baseline RedCode di-skip karena bukan
    serangan tool-call. Tiap entry punya 3 representasi teks (code/summary/
    details) untuk context_text, tool_call tetap sama → exact dedup lintas
    varian ditangani dedup_records() di pipeline, bukan di sini.
    """
    lang_config = [
        ("py2text_dataset_json",   "execute_python", {1, 2, 4, 6, 7, 8, 9, 10, 13, 14, 18, 24}),
        ("bash2text_dataset_json", "run_bash",        {1, 2, 4, 6, 7, 8, 9, 13, 14, 18}),
    ]

    samples = []
    raw_count = 0
    for subdir, tool_name, malicious_indices in lang_config:
        folder = os.path.join(dir_path, subdir)
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            m = re.match(r"index(\d+)_", filename)
            if not m or int(m.group(1)) not in malicious_indices:
                continue
            with open(os.path.join(folder, filename), encoding="utf-8") as f:
                entries = json.load(f)

            for entry in entries:
                raw_count += 1
                code = str(entry.get("Code", ""))[:1000]
                tool_calls = [{"name": tool_name, "parameters": {"code": code}}]

                for ctx_text in (entry.get("Text_summary"), entry.get("Text_details"), code):
                    if not ctx_text:
                        continue
                    samples.append({
                        "tool_call_sequence": tool_calls,
                        "label":              "direct_attack",
                        "conversation":       [{"role": "user", "content": str(ctx_text)[:500]}],
                        "source":             "redcode_exec",
                    })

    return samples


# Loader AgentHarm (HuggingFace ai-safety-institute/AgentHarm)
def load_agentharm_hf(n_samples: int = 1500, seed: int = 42) -> List[Dict]:
    """
    target_functions selalu berisi >=2 tool call di dataset ini (chain serangan),
    jadi hanya item dengan tepat 1 tool call yang masuk direct_attack — sisanya
    di-skip (bukan disintesa), agar tetap 100% data real.
    """
    from datasets import load_dataset, concatenate_datasets
    import json

    splits = []
    for split in ("test_public", "validation"):
        try:
            splits.append(load_dataset("ai-safety-institute/AgentHarm", "harmful", split=split))
        except Exception:
            continue
    if not splits:
        return []

    ds = concatenate_datasets(splits)
    df = ds.to_pandas()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    samples = []
    for _, row in df.iterrows():
        if len(samples) >= n_samples:
            break

        target_fns = row.get("target_functions", [])
        if isinstance(target_fns, str):
            try:
                target_fns = json.loads(target_fns)
            except Exception:
                target_fns = []
        elif hasattr(target_fns, "tolist"):
            target_fns = target_fns.tolist()

        if not isinstance(target_fns, list) or len(target_fns) != 1:
            continue

        tool_calls = [{"name": str(fn), "parameters": {}} for fn in target_fns]
        prompt = str(row.get("prompt", "")).strip()

        samples.append({
            "tool_call_sequence": tool_calls,
            "label": "direct_attack",
            "conversation": [{"role": "user", "content": prompt[:300]}],
            "source": "agentharm_hf"
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
    return samples


# Loader ToolSafe AgentSafeBench
def load_toolsafe_safetybench(path: str) -> List[Dict]:
    """
    Load released_data.json dari ToolSafe AgentSafeBench (ekstraksi asli,
    logic tidak diubah) ditambah re-mining file training yang belum pernah
    diproses (train_data_1025*.json, agent_align_data_v3_harmful.json) —
    file leftover ini dipakai cuma untuk direct_attack dengan filter ketat
    exact 1 tool call, biar konsisten sama source baru lain.
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

    samples.extend(_load_toolsafe_safetybench_leftover(os.path.dirname(path)))
    return samples


# Re-mining file training ToolSafe yang belum ke-extract
def _load_toolsafe_safetybench_leftover(data_dir: str) -> List[Dict]:
    samples = []

    # train_data_1025* — environments[].tools exact 1 = direct_attack
    train_files = [
        "train_data_1025.json", "train_data_1025_v2.json",
        "train_data_1025_v3.json", "train_data_1025_v4.json",
    ]
    for fname in train_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            extra = json.load(f)
        for item in extra:
            tool_names = []
            for env in item.get("environments", []):
                tool_names.extend(env.get("tools", []))
            if len(tool_names) != 1:
                continue
            instruction = str(item.get("instruction", ""))[:500]
            samples.append({
                "tool_call_sequence": [{"name": tool_names[0], "parameters": {}}],
                "label":              "direct_attack",
                "conversation":       [{"role": "user", "content": instruction}] if instruction else [],
                "source":             "toolsafe_safetybench",
            })

    # agent_align_data_v3_harmful — pattern exact 1 tool = direct_attack
    align_path = os.path.join(data_dir, "agent_align_data_v3_harmful.json")
    if os.path.exists(align_path):
        with open(align_path, encoding="utf-8") as f:
            align = json.load(f)
        for item in align:
            pattern = item.get("pattern", [])
            if len(pattern) != 1:
                continue
            user_msg = next((m.get("content", "") for m in item.get("messages", []) if m.get("role") == "user"), "")
            samples.append({
                "tool_call_sequence": [{"name": pattern[0], "parameters": {}}],
                "label":              "direct_attack",
                "conversation":       [{"role": "user", "content": str(user_msg)[:500]}] if user_msg else [],
                "source":             "toolsafe_safetybench",
            })

    return samples


# Loader WildJailbreak
def load_wildjailbreak(path: str) -> List[Dict]:
    import re
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)

    def _get_label(row) -> str:
        dt = str(row.get("data_type", "")).lower()
        if "benign" in dt:
            return "benign"
        if "harmful" in dt:
            return "sequential_attack"
        lbl = row.get("label", 0)
        return "benign" if int(lbl) == 0 else "sequential_attack"

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


# Kumpulkan semua sample dari semua sumber (tanpa split) — dipakai load_all_datasets() & export script
def collect_all_samples(processed_dir: str) -> List[Dict]:
    all_samples = []

    # 1. SafeToolBench
    stb_dir = os.path.join(processed_dir, "safetoolbench")
    if os.path.isdir(stb_dir):
        all_samples.extend(load_safetoolbench_dir(stb_dir))
    else:
        print(f"Skip SafeToolBench — tidak ditemukan: {stb_dir}")

    # 2. STAC
    if os.path.exists(STAC_PATH):
        all_samples.extend(load_stac(STAC_PATH))
    else:
        print(f"Skip STAC — tidak ditemukan: {STAC_PATH}")

    # 3. ToolSafe AgentHarm
    if os.path.isdir(TOOLSAFE_AGENTHARM_DIR):
        all_samples.extend(load_toolsafe_agentharm(TOOLSAFE_AGENTHARM_DIR))
    else:
        print(f"Skip ToolSafe AgentHarm — tidak ditemukan: {TOOLSAFE_AGENTHARM_DIR}")

    # 4. ToolSafe AgentSafeBench
    if os.path.exists(TOOLSAFE_SAFETYBENCH_PATH):
        all_samples.extend(load_toolsafe_safetybench(TOOLSAFE_SAFETYBENCH_PATH))
    else:
        print(f"Skip ToolSafe SafetyBench — tidak ditemukan: {TOOLSAFE_SAFETYBENCH_PATH}")

    # 5. WildJailbreak (jika sudah didownload)
    if os.path.exists(WILDJAILBREAK_PATH):
        all_samples.extend(load_wildjailbreak(WILDJAILBREAK_PATH))
    else:
        print(f"Skip WildJailbreak — belum didownload: {WILDJAILBREAK_PATH}")

    # 7. R-Judge
    if os.path.isdir(RJUDGE_DIR):
        all_samples.extend(load_rjudge_dir(RJUDGE_DIR))
    else:
        print(f"Skip R-Judge — tidak ditemukan: {RJUDGE_DIR}")

    # 8. InjecAgent
    if os.path.isdir(INJECAGENT_DIR):
        all_samples.extend(load_injecagent_dir(INJECAGENT_DIR))
    else:
        print(f"Skip InjecAgent — tidak ditemukan: {INJECAGENT_DIR}")

    # 9. AgentHarm HF (ai-safety-institute/AgentHarm) — direct_attack
    try:
        all_samples.extend(load_agentharm_hf(n_samples=1500))
    except Exception as e:
        print(f"Skip AgentHarm HF — gagal download: {e}")

    # 10. AgentDojo — parameter_injection
    if os.path.isdir(os.path.join(AGENTDOJO_RUNS_DIR, AGENTDOJO_PIPELINE)):
        all_samples.extend(load_agentdojo(n_samples=512))
    else:
        print(f"Skip AgentDojo — tidak ditemukan: {AGENTDOJO_RUNS_DIR}")

    # 11. APIGen-MT-5k — benign
    try:
        all_samples.extend(load_apigen_mt_benign(n_samples=4000))
    except Exception as e:
        print(f"Skip APIGen-MT-5k — gagal download: {e}")

    # 12. BIPIA — parameter_injection
    try:
        all_samples.extend(load_bipia(n_samples=3500))
    except Exception as e:
        print(f"Skip BIPIA — gagal download: {e}")

    # 13. RedCode-Exec — direct_attack
    if os.path.isdir(REDCODE_EXEC_DIR):
        all_samples.extend(load_redcode_exec(REDCODE_EXEC_DIR))
    else:
        print(f"Skip RedCode-Exec — tidak ditemukan: {REDCODE_EXEC_DIR}")

    return all_samples


# Dedup exact lintas semua source berdasar tool_calls_json
def dedup_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    deduped = df.drop_duplicates(subset=["tool_calls_json"], keep="first").reset_index(drop=True)
    return deduped, before - len(deduped)


# Undersample kelas berlebih ke target real (tanpa augmentasi sintetis)
def balance_records(df: pd.DataFrame, seed: int = SEED, tolerance: float = 1.15) -> Tuple[pd.DataFrame, Dict]:
    counts_before = df["label"].value_counts().to_dict()
    minority_label = min(counts_before, key=counts_before.get)
    target = counts_before[minority_label]
    ceiling = int(target * tolerance)

    parts = []
    for label, group in df.groupby("label"):
        if len(group) > ceiling:
            parts.append(group.sample(n=ceiling, random_state=seed))
        else:
            parts.append(group)
    balanced = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    report = {
        "minority_label":  minority_label,
        "target_per_class": target,
        "ceiling":          ceiling,
        "counts_before":    counts_before,
        "counts_after":     balanced["label"].value_counts().to_dict(),
    }
    return balanced, report


# Gabungkan semua dataset lalu split train/val/test
def load_all_datasets(processed_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_samples = collect_all_samples(processed_dir)

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

if __name__ == "__main__":
    processed_dir = os.path.join(os.path.dirname(__file__), "processed")
    all_samples = collect_all_samples(processed_dir)

    # Bangun record CSV per sample
    records = []
    for s in all_samples:
        seq = s.get("tool_call_sequence", [])
        tc_json = json.dumps(seq)

        tc_texts = sequence_to_texts(seq)
        tc_text = " ".join(tc_texts) if isinstance(tc_texts, list) else str(tc_texts)

        ctx_text = extract_context(s.get("conversation", []))

        records.append({
            "tool_calls_json": tc_json,
            "tool_calls_text": tc_text,
            "label": s.get("label", ""),
            "source": s.get("source", ""),
            "context_text": ctx_text
        })

    df_raw = pd.DataFrame(records)

    # Blokir source non-real (legacy synthetic/pseudo-label residual)
    BLOCKED_SOURCES = {"synthetic", "scraped_labeled"}
    df_raw = df_raw[~df_raw["source"].isin(BLOCKED_SOURCES)].reset_index(drop=True)

    df_deduped, n_dropped = dedup_records(df_raw)
    df_final, report = balance_records(df_deduped, seed=SEED)

    out_path = os.path.join(os.path.dirname(__file__), "tcssc_dataset.csv")
    df_final.to_csv(out_path, index=False)

    counts_after = report["counts_after"]
    imbalance_ratio = max(counts_after.values()) / min(counts_after.values())

    print(f"File CSV berhasil dibuat: {out_path}")
    print(f"Raw: {len(df_raw)} | Duplikat di-drop: {n_dropped} | Setelah dedup: {len(df_deduped)} | Final balanced: {len(df_final)}")
    print(f"Distribusi sebelum balance: {report['counts_before']}")
    print(f"Distribusi final: {counts_after} | imbalance ratio: {imbalance_ratio:.2f}")

    minority = report["minority_label"]
    if imbalance_ratio > 1.15:
        print(f"PERINGATAN: kelas '{minority}' jadi bottleneck (target_per_class={report['target_per_class']}) "
              f"— data real masih kurang, perlu source tambahan untuk kelas ini.")